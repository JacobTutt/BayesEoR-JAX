"""Small example inputs preserve the production visibility and beam conventions."""

import json
from pathlib import Path

import numpy as np
import pytest

from bayeseor_jax import BayesEoRParser, Beam_Model, build_sky_to_visibility_operator
from bayeseor_jax.model.observation.image_geometry import required_healpix_nside
from test_examples.generate_data import (
    BASELINES, FREQUENCIES, TIMES, MASK_NSIDE,
    build_footprint, healpix_sky, select_directions, integrate,
)


@pytest.mark.parametrize("name", ["eor", "eor_fgs"])
def test_shipped_laptop_data_match_config(name):
    config = BayesEoRParser().parse_args(["--config", f"test_examples/{name}/config.yaml"])
    instrument = Path(config.inst_model)
    metadata = json.loads((instrument / "simulation.json").read_text())
    uvw = np.load(instrument / "uvw_model.npy", allow_pickle=False)
    redundancy = np.load(instrument / "redundancy_model.npy", allow_pickle=False)
    vis = np.load(config.data_path, allow_pickle=False).reshape(config.n_t, config.n_freq, 30)
    assert vis.dtype == np.complex128 and np.all(np.isfinite(vis))
    assert vis.size == 6750 and uvw.shape == (15, 30, 3)
    np.testing.assert_array_equal(vis[..., 15:], vis[..., :15].conj())
    np.testing.assert_array_equal(uvw[:, 15:], -uvw[:, :15])
    np.testing.assert_array_equal(redundancy, np.ones((15, 30)))
    assert config.nside == 64
    np.testing.assert_array_equal(uvw[0, :15], BASELINES)
    assert np.max(np.linalg.norm(uvw, axis=-1)) < 40
    assert required_healpix_nside(np.max(np.linalg.norm(uvw, axis=-1)), FREQUENCIES.max()) == 64
    np.testing.assert_allclose(metadata["frequency_hz"],
                               1e6*(config.f0_mhz + config.df_mhz*np.arange(config.n_freq)))
    np.testing.assert_allclose(metadata["jd"], TIMES, rtol=0, atol=1e-9)
    assert config.fov_eor == metadata["field_diameter_deg"] == 10
    assert len(config.priors) == 6
    assert metadata["mask_nside"] == config.nside


@pytest.mark.parametrize("fine_nside", [128, 256])
def test_fine_sky_inherits_exactly_the_selected_coarse_pixels(fine_nside):
    import jax_healpy as hp
    from bayeseor_jax.model.observation.image_geometry import build_image_geometry, HERA_LOCATION

    mask = build_footprint()
    fine_sky = healpix_sky(fine_nside)
    selected, _ = select_directions(fine_sky, mask)
    parent = np.asarray(hp.ang2pix(MASK_NSIDE, np.pi/2-fine_sky.dec.rad, fine_sky.ra.rad))
    np.testing.assert_array_equal(np.unique(parent[selected]), np.flatnonzero(mask))
    counts = np.bincount(parent[selected], minlength=mask.size)
    np.testing.assert_array_equal(counts, (fine_nside//MASK_NSIDE)**2*mask.astype(int))

    # Cross-check the selected coarse centres against production model geometry.
    coarse_indices, coarse_lmn = select_directions(healpix_sky(64), mask)
    geometry = build_image_geometry(fov_eor=10, fov_fg=10, nside=64,
                                   telescope_latlonalt=HERA_LOCATION,
                                   jd_center=TIMES[len(TIMES)//2], n_t=len(TIMES),
                                   d_t=21.511369943618774)
    assert len(coarse_indices) == geometry.n_pix_fg
    np.testing.assert_allclose(coarse_lmn.transpose(0, 2, 1), geometry.lmn_fg, atol=1e-12)

    # Point sources use the same membership rule, with no separate circular cut.
    points = fine_sky[[selected[0], np.flatnonzero(~mask[parent])[0]]]
    point_indices, _ = select_directions(points, mask)
    np.testing.assert_array_equal(point_indices, [0])


def test_independent_simulator_matches_visibility_operator():
    # Include zenith and a below-horizon source to check limits and masking.
    directions = np.array([[0, 0, 1], [.04, .03, np.sqrt(1-.04**2-.03**2)], [0, 0, -1.]])
    lmn = np.broadcast_to(directions, (3, 3, 3))
    az = np.arctan2(lmn[..., 0], lmn[..., 1])
    za = np.arccos(lmn[..., 2])
    beam = Beam_Model("airy", diam=14.2).get_beam_vals(np.stack((az, za), axis=1), FREQUENCIES)
    uvw = np.concatenate((BASELINES, -BASELINES)).T[None, :, :] * FREQUENCIES[:, None, None]/299792458
    response = build_sky_to_visibility_operator(lmn.transpose(0, 2, 1), uvw,
                                                np.asarray(beam).transpose(1, 0, 2), .002)
    brightness = np.arange(len(FREQUENCIES)*3).reshape(-1, 3) + 1.
    expected = np.einsum("tfbp,fp->tfb", response, brightness).reshape(-1)
    np.testing.assert_allclose(integrate(lmn, brightness*.002), expected, rtol=1e-11, atol=1e-12)


def test_injected_spectrum_uses_simulation_resolution():
    from test_examples.plot_recoveries import injected_power_spectrum

    metadata = json.loads(Path("test_examples/instrument/simulation.json").read_text())
    power = injected_power_spectrum(metadata)
    assert power == pytest.approx(8.84e4, rel=.005)
    metadata["signal_nside"] *= 2
    assert injected_power_spectrum(metadata) == pytest.approx(power/4)
