"""Exercise setup's real matrix stages without requiring the GPU solver."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


def test_setup_refreshes_data_and_noise_without_rebuilding_components(tmp_path, monkeypatch):
    # The test stops at likelihood configuration, before any JAXMg solve.
    monkeypatch.setitem(
        sys.modules, "bayeseor_jax.inference.likelihood_marg",
        SimpleNamespace(MarginalisedLikelihood=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    spec = importlib.util.spec_from_file_location(
        "bayeseor_jax._setup_under_test",
        Path(__file__).parents[1] / "bayeseor_jax/setup.py",
    )
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    data = np.arange(6, dtype=float) + 1j
    vis = dict(
        freqs_hz=np.array([150e6, 151e6, 152e6]), df_hz=1e6,
        uvws=np.array([[[1., 0., 0.], [0., 2., 0.]]]), dt=10.,
        jds=np.array([2461040.5]), redundancy=np.array([[1., 2.]]),
        vis_noisy=data,
    )
    geometry = SimpleNamespace(
        az_za_fg=jnp.zeros((1, 2, 2)),
        lmn_fg=jnp.array([[[0., .1], [0., 0.], [1., np.sqrt(.99)]]]),
        time_ind_center=0, eor_fg_shared_axis=jnp.array([0, 1]),
        pixel_area_sr=.1, n_pix_fg=2,
        fov_l_angle_eor=.2, fov_m_angle_eor=.2,
        fov_l_angle_fg=.4, fov_m_angle_fg=.4,
    )
    grid = dict(n_u_eor=3, n_v_eor=3, du_eor=1., dv_eor=1.,
                beam_aperture_width_uv=1., u_max_inst_model=1., v_max_inst_model=1.,
                u_extent_with_buffer=1., v_extent_with_buffer=1.,
                n_u_eor_required=3, n_v_eor_required=3)
    monkeypatch.setattr(setup, "get_vis_data", lambda **kwargs: vis)
    monkeypatch.setattr(setup, "build_image_geometry", lambda **kwargs: geometry)
    monkeypatch.setattr(setup, "_resolve_uv_grid_from_extents", lambda **kwargs: grid)
    monkeypatch.setattr(setup, "write_analysis_metadata", lambda **kwargs: None)
    # One power bin over the 8 transverse x 2 line-of-sight coefficients.
    monkeypatch.setattr(
        setup, "generate_k_cube_model_spherical_binning",
        lambda **kwargs: (jnp.zeros(16, dtype=int), jnp.ones(16, dtype=bool),
                          None, None, jnp.array([16])),
    )
    arguments = dict(data_path=tmp_path / "data.npy", n_freq=3, n_eta=3,
                     array_save_directory=tmp_path, output_dir=tmp_path,
                     priors=[[-1., 1.]], nside=1)
    result = setup.run_setup(**arguments, sigma=.5)
    operator = np.load(tmp_path / "T.npy")
    inverse_noise = np.load(tmp_path / "Ninv.npy")
    assert operator.shape == (6, 22)
    assert result.likelihood_class.n_data == 6
    assert result.phi_inverse_class.n_par == 22
    np.testing.assert_allclose(np.load(tmp_path / "T_Ninv_T.npy"),
                               operator.conj().T @ (inverse_noise[:, None] * operator))
    np.testing.assert_allclose(np.load(tmp_path / "dbar.npy"),
                               operator.conj().T @ (inverse_noise * data))
    stamps = {name: (tmp_path / name).stat().st_mtime_ns
              for name in ("T_21.npy", "T_fg.npy", "T.npy", "T_Ninv_T.npy")}

    vis["vis_noisy"] = 2 * data
    setup.run_setup(**arguments, sigma=.5)
    assert all((tmp_path / name).stat().st_mtime_ns == stamp for name, stamp in stamps.items())
    np.testing.assert_allclose(np.load(tmp_path / "dbar.npy"),
                               2 * operator.conj().T @ (inverse_noise * data))
    np.testing.assert_allclose(np.load(tmp_path / "d_ninv_d.npy"),
                               4 * np.vdot(data, inverse_noise * data).real)

    setup.run_setup(**arguments, sigma=1.)
    assert (tmp_path / "T.npy").stat().st_mtime_ns == stamps["T.npy"]
    np.testing.assert_allclose(np.load(tmp_path / "Ninv.npy"), inverse_noise / 4)
    np.testing.assert_allclose(np.load(tmp_path / "T_Ninv_T.npy"),
                               operator.conj().T @ (inverse_noise[:, None] * operator) / 4)
