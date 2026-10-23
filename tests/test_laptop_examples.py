"""Small example inputs preserve the production visibility and beam conventions."""

import json
from pathlib import Path

import numpy as np
import pytest

from bayeseor_jax import BayesEoRParser, Beam_Model, build_sky_to_visibility_operator
from bayeseor_jax.model.observation.image_geometry import required_healpix_nside
from test_examples.generate_data import BASELINES, FREQUENCIES, TIMES, integrate


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
