"""Small example inputs preserve the production visibility and beam conventions."""

import json
from pathlib import Path

import numpy as np
import pytest

from bayeseor_jax import BayesEoRParser
from bayeseor_jax.model.observation.image_geometry import required_healpix_nside


@pytest.mark.parametrize("name", ["eor", "eor_fgs"])
def test_shipped_laptop_data_match_config(name):
    config = BayesEoRParser().parse_args(["--config", f"test_examples/{name}/config.yaml"])
    instrument = Path(config.inst_model)
    assert instrument == Path(f"test_examples/{name}/data")
    assert Path(config.data_path).parent == instrument
    metadata = json.loads((instrument / "simulation.json").read_text())
    uvw = np.load(instrument / "uvw_model.npy", allow_pickle=False)
    redundancy = np.load(instrument / "redundancy_model.npy", allow_pickle=False)
    vis = np.load(config.data_path, allow_pickle=False).reshape(config.n_t, config.n_freq, 30)
    assert vis.dtype == np.complex128 and np.all(np.isfinite(vis))
    assert vis.size == 6750 and uvw.shape == (15, 30, 3)
    np.testing.assert_array_equal(vis[..., 15:], vis[..., :15].conj())
    np.testing.assert_array_equal(uvw[:, 15:], -uvw[:, :15])
    np.testing.assert_array_equal(redundancy, np.ones((15, 30)))
    assert config.nside == 128
    assert metadata["components"] == (["21cm"] if name == "eor" else ["21cm", "foregrounds"])
    assert np.max(np.linalg.norm(uvw, axis=-1)) < 40
    assert required_healpix_nside(np.max(np.linalg.norm(uvw, axis=-1)), max(metadata["frequency_hz"])) == 64
    np.testing.assert_allclose(metadata["frequency_hz"],
                               1e6*(config.f0_mhz + config.df_mhz*np.arange(config.n_freq)))
    np.testing.assert_allclose(metadata["jd"], config.jd_center + np.arange(-7, 8)*config.d_t/86400,
                               rtol=0, atol=1e-9)
    assert config.fov_eor == metadata["field_diameter_deg"] == 10
    assert len(config.priors) == 6
    assert metadata["mask_nside"] == config.footprint_nside == 64


@pytest.mark.parametrize("fine_nside", [128, 256])
def test_fine_sky_inherits_exactly_the_selected_coarse_pixels(fine_nside):
    import jax_healpy as hp
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from bayeseor_jax.model.observation.image_geometry import build_image_geometry, HERA_LOCATION

    metadata = json.loads(Path("test_examples/eor/data/simulation.json").read_text())
    theta, phi = hp.pix2ang(fine_nside, np.arange(12*fine_nside**2))
    parent = np.asarray(hp.ang2pix(64, theta, phi))
    mask = np.zeros(12*64**2, dtype=bool)
    mask[metadata["mask_ring_pixels"]] = True
    selected = np.flatnonzero(mask[parent])
    counts = np.bincount(parent[selected], minlength=mask.size)
    np.testing.assert_array_equal(counts, (fine_nside//64)**2*mask.astype(int))

    # Check model directions against the saved simulation footprint, independently.
    lat, lon, height = HERA_LOCATION
    frames = AltAz(obstime=Time(metadata["jd"], format="jd"),
                   location=EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height*u.m))
    sky = SkyCoord(ra=np.asarray(phi)[selected]*u.rad,
                   dec=(np.pi/2-np.asarray(theta)[selected])*u.rad)
    horizontal = sky[None, :].transform_to(frames[:, None])
    az, alt = horizontal.az.rad, horizontal.alt.rad
    lmn = np.stack((np.cos(alt)*np.sin(az), np.cos(alt)*np.cos(az), np.sin(alt)), axis=1)
    refined = build_image_geometry(fov_eor=10, fov_fg=10, nside=fine_nside,
                                  footprint_nside=64, telescope_latlonalt=HERA_LOCATION,
                                  jd_center=metadata["jd"][7], n_t=15,
                                  d_t=21.511369943618774)
    assert refined.n_pix_fg == len(selected)
    np.testing.assert_allclose(refined.lmn_fg, lmn, atol=1e-12)
    assert np.all(refined.eor_fg_shared_axis)
    assert refined.pixel_area_sr == pytest.approx(4*np.pi/(12*fine_nside**2))

def test_injected_spectrum_uses_simulation_resolution():
    from test_examples.plot_recoveries import injected_power_spectrum

    metadata = json.loads(Path("test_examples/eor/data/simulation.json").read_text())
    power = injected_power_spectrum(metadata)
    assert power == pytest.approx(8.84e4, rel=.005)
    metadata["signal_nside"] *= 2
    assert injected_power_spectrum(metadata) == pytest.approx(power/4)


@pytest.mark.parametrize("name", ["eor", "eor_fgs"])
def test_independent_entrypoint_selects_own_inputs(name, monkeypatch):
    import runpy
    from test_examples import run_analysis_example as example

    calls = []
    monkeypatch.setattr(example, "main", lambda *args, **kwargs: calls.append((args, kwargs)))
    runpy.run_module(f"test_examples.{name}.run", run_name="__main__")
    (config,), options = calls.pop()
    assert Path(config).resolve() == Path(f"test_examples/{name}/config.yaml").resolve()
    assert options.get("prepare_foreground", False) == (name == "eor_fgs")
    assert options["expected_ps"] == pytest.approx(8.84e4, rel=.005)
    assert not calls


def test_example_orchestrates_pipeline_and_preserves_saved_results(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import bayeseor_jax
    from test_examples.eor_fgs import prepare_kl as preparation
    from test_examples import run_analysis_example as example

    steps = []
    data = np.array([1+2j, 1-2j])
    noise = np.array([.1+.2j, .1-.2j])
    def prepare(config):
        steps.append(('kl', config))
    def setup(config):
        steps.append(('setup', config))
        return SimpleNamespace(vis_dict={'vis_noisy':data+noise, 'noise':noise})
    def sample(**kwargs):
        steps.append(('sample', kwargs))
        kwargs['out_dir'].mkdir()
        (kwargs['out_dir']/'summary.json').write_text('{"sampler":"multinest"}')
        return {'sampler':'multinest'}
    monkeypatch.setattr(preparation, 'prepare_kl', prepare)
    monkeypatch.setattr(bayeseor_jax, 'run_setup', setup)
    monkeypatch.setattr(bayeseor_jax, 'run_nestedsampling', sample)
    monkeypatch.setattr(example, 'run_output_analysis', lambda **kwargs:steps.append(('analysis', kwargs)))
    overrides = ['--output-dir', str(tmp_path), '--array-save-directory', str(tmp_path/'matrices'),
                 '--foreground-kl-path', str(tmp_path/'kl/T_fg_reduced.npy')]
    example.run_example('test_examples/eor_fgs/config.yaml', do_sampling=True,
                        prepare_foreground=True, extra_bayeseor_args=overrides)
    assert [name for name, _ in steps] == ['kl','setup','sample','analysis']
    assert steps[0][1] is steps[1][1]
    assert steps[0][1].nside == 128 and steps[0][1].footprint_nside == 64
    assert steps[2][1]['center_likelihood'] is True
    np.testing.assert_array_equal(np.load(tmp_path/'data/visibilities_noisy.npy'), data+noise)
    np.testing.assert_array_equal(np.load(tmp_path/'data/noise.npy'), noise)
    with pytest.raises(FileExistsError, match='Sampler output already exists'):
        example.run_example('test_examples/eor_fgs/config.yaml', do_sampling=True,
                            extra_bayeseor_args=overrides)
    with pytest.raises(ValueError, match='configuration changed'):
        example.run_example('test_examples/eor_fgs/config.yaml',
                            extra_bayeseor_args=overrides+['--nside','64'])
    assert len(steps) == 4
