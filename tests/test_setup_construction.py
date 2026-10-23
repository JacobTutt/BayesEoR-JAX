"""Exercise setup's real matrix stages without requiring the GPU solver."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import asdict, FrozenInstanceError
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.configuration import RunConfig


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
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
                     priors=[[-1., 1.]], nside=4)
    return setup, arguments, vis


def test_invalid_prior_count_does_not_save_analysis_metadata(setup_case, monkeypatch):
    setup, arguments, _ = setup_case
    saved = []
    monkeypatch.setattr(setup, "write_analysis_metadata", lambda **kwargs: saved.append(kwargs))
    arguments["priors"] = [[-1., 1.], [-1., 1.]]
    with pytest.raises(ValueError, match="k binning produced 1 non-empty bins"):
        setup.run_setup(RunConfig(**arguments, sigma=.5))
    assert saved == []
    assert not (Path(arguments["array_save_directory"]) / "T_21.npy").exists()


def test_setup_preserves_requested_settings_and_resolves_observation(setup_case):
    setup, arguments, vis = setup_case
    requested = dict(arguments, n_freq=99, f0_mhz=1., df_mhz=.01, n_t=20,
                     jd_center=2450000., fov_fg=None, sigma=.5)
    config = RunConfig(**requested)
    before = asdict(config)
    result = setup.run_setup(config)
    assert asdict(config) == before
    assert result.config is config
    assert result.bm.config is config
    assert result.bm.n_freq == 3
    assert result.bm.n_t == 1
    np.testing.assert_array_equal(result.bm.freq_hz, vis["freqs_hz"])
    with pytest.raises(FrozenInstanceError):
        config.n_freq = 3


@pytest.mark.parametrize("model", ["eor", "fg", "eor+fg"])
@pytest.mark.parametrize("requested,expected", [(None, 64), (32, 32), (128, 128)])
def test_setup_resolves_shared_nside_from_selected_observation(
    setup_case, monkeypatch, model, requested, expected,
):
    setup, arguments, vis = setup_case
    # The w component and highest channel must contribute to the estimate.
    vis["uvws"] = np.array([[[0., 0., 40.], [0., 2., 0.]]])
    vis["freqs_hz"] = np.array([50e6, 100e6, 150e6])
    vis["df_hz"] = 50e6
    geometry = setup.build_image_geometry()
    received, metadata = {}, {}

    def capture_geometry(**kwargs):
        received.update(kwargs)
        return geometry

    monkeypatch.setattr(setup, "build_image_geometry", capture_geometry)
    monkeypatch.setattr(setup, "write_analysis_metadata", lambda **kw: metadata.update(kw["args"]))
    monkeypatch.setattr(setup, "write_log_files", lambda **kw: metadata.update(kw["args"]))
    arguments.update(nside=requested, model=model, f0_mhz=1., df_mhz=.01)
    config = RunConfig(**arguments, sigma=.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setup.run_setup(config)
    resolution_warnings = [str(w.message) for w in caught if "nside=" in str(w.message)]
    assert len(resolution_warnings) == (requested == 32)
    assert received["nside"] == metadata["nside"] == expected
    assert metadata["nside_required"] == 64
    assert config.nside == requested


@pytest.mark.parametrize("nside", [0, -1, 2.5])
def test_setup_rejects_invalid_nside(setup_case, nside):
    setup, arguments, _ = setup_case
    arguments["nside"] = nside
    with pytest.raises(ValueError, match="nside must be a positive integer"):
        setup.run_setup(RunConfig(**arguments, sigma=.5))


def test_setup_refreshes_data_and_noise_without_rebuilding_components(tmp_path, setup_case):
    setup, arguments, vis = setup_case
    data = vis['vis_noisy']
    result = setup.run_setup(RunConfig(**arguments, sigma=.5))
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
    setup.run_setup(RunConfig(**arguments, sigma=.5))
    assert all((tmp_path / name).stat().st_mtime_ns == stamp for name, stamp in stamps.items())
    np.testing.assert_allclose(np.load(tmp_path / "dbar.npy"),
                               2 * operator.conj().T @ (inverse_noise * data))
    np.testing.assert_allclose(np.load(tmp_path / "d_ninv_d.npy"),
                               4 * np.vdot(data, inverse_noise * data).real)

    setup.run_setup(RunConfig(**arguments, sigma=1.))
    assert (tmp_path / "T.npy").stat().st_mtime_ns == stamps["T.npy"]
    np.testing.assert_allclose(np.load(tmp_path / "Ninv.npy"), inverse_noise / 4)
    np.testing.assert_allclose(np.load(tmp_path / "T_Ninv_T.npy"),
                               operator.conj().T @ (inverse_noise[:, None] * operator) / 4)


@pytest.mark.parametrize('rank', [0, 3, 6])
def test_setup_selects_kl_operator_and_unit_prior(tmp_path, setup_case, monkeypatch, rank):
    setup, arguments, vis = setup_case
    original = setup.run_setup(RunConfig(**arguments, sigma=.5, foreground_prior_std_mk=[1., 2., 4.]))
    original_t = np.load(tmp_path / 'T.npy')
    np.testing.assert_allclose(original.phi_inverse_class.foreground_precision,
                               [1., 1., .25, .25, .0625, .0625])
    rng = np.random.default_rng(7)
    reduced = (rng.normal(size=(6, rank)) + 1j * rng.normal(size=(6, rank))).astype(np.complex128)
    reduced_path = tmp_path / 'reduced.npy'
    np.save(reduced_path, reduced)
    saved_bytes = reduced_path.read_bytes()

    def unexpected_build(*args, **kwargs):
        raise AssertionError('A supplied KL operator must not rebuild the physical foreground')

    with monkeypatch.context() as patch:
        patch.setattr(setup.BuildMatrices, 'T_fg_build_save', unexpected_build)
        # Even different physical widths must not be applied a second time.
        result = setup.run_setup(RunConfig(**arguments, sigma=.5, foreground_kl_path=reduced_path,
                                 foreground_prior_std_mk=[100., 200., 400.]))
    operator = np.load(tmp_path / 'T.npy')
    np.testing.assert_array_equal(operator[:, :16], original_t[:, :16])
    np.testing.assert_array_equal(operator[:, 16:], reduced)
    np.testing.assert_array_equal(result.phi_inverse_class.foreground_precision, np.ones(rank))
    assert result.phi_inverse_class.n_par == 16 + rank
    assert reduced_path.read_bytes() == saved_bytes
    inverse_noise = np.load(tmp_path / 'Ninv.npy')
    np.testing.assert_allclose(np.load(tmp_path / 'T_Ninv_T.npy'),
                               operator.conj().T @ (inverse_noise[:, None] * operator), atol=1e-12)
    np.testing.assert_allclose(np.load(tmp_path / 'dbar.npy'),
                               operator.conj().T @ (inverse_noise * vis['vis_noisy']), atol=1e-12)
    np.testing.assert_array_equal(
        result.phi_inverse_class.evaluate(jnp.ones(1))[:16],
        original.phi_inverse_class.evaluate(jnp.ones(1))[:16],
    )

    # Switching back invalidates cached joint products, including equal-sized bases.
    setup.run_setup(RunConfig(**arguments, sigma=.5, foreground_prior_std_mk=[1., 2., 4.]))
    np.testing.assert_array_equal(np.load(tmp_path / 'T.npy'), original_t)


def test_kl_setup_does_not_need_original_foreground_file(tmp_path, setup_case, monkeypatch):
    setup, arguments, _ = setup_case
    path = tmp_path / 'reduced.npy'
    np.save(path, np.ones((6, 2), dtype=np.complex128))
    monkeypatch.setattr(setup.BuildMatrices, 'T_fg_build_save', lambda **kwargs: pytest.fail('Unexpected build'))
    setup.run_setup(RunConfig(**arguments, sigma=.5, foreground_kl_path=path))
    assert not (tmp_path / 'T_fg.npy').exists()


@pytest.mark.parametrize('matrix', [np.ones((5, 2), dtype=np.complex128), np.ones((6, 2)), np.ones(6)])
def test_setup_rejects_incompatible_kl_matrix(tmp_path, setup_case, matrix):
    setup, arguments, _ = setup_case
    path = tmp_path / 'reduced.npy'
    np.save(path, matrix)
    with pytest.raises(ValueError, match='KL foreground operator'):
        setup.run_setup(RunConfig(**arguments, sigma=.5, foreground_kl_path=path))


def test_kl_input_cannot_be_overwritten_by_setup(tmp_path, setup_case):
    setup, arguments, _ = setup_case
    path = tmp_path / 'T_21.npy'
    np.save(path, np.ones((6, 2), dtype=np.complex128))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='setup output'):
        setup.run_setup(RunConfig(**arguments, sigma=.5, foreground_kl_path=path, clobber=True))
    assert path.read_bytes() == before


@pytest.mark.parametrize("model,n_columns", [("eor", 16), ("fg", 6), ("eor+fg", 22)])
def test_setup_constructs_only_selected_components(tmp_path, setup_case, monkeypatch, model, n_columns):
    setup, arguments, _ = setup_case
    if model == "fg":
        arguments["priors"] = None
        monkeypatch.setattr(setup, "_resolve_uv_grid_from_extents", lambda **kw: pytest.fail("Unexpected Fourier grid"))
        monkeypatch.setattr(setup, "generate_k_cube_model_spherical_binning", lambda **kw: pytest.fail("Unexpected k bins"))
        monkeypatch.setattr(setup.BuildMatrices, "T_21_build_save", lambda **kw: pytest.fail("Unexpected T_21"))
    if model == "eor":
        arguments.update(beta_fg=[], foreground_prior_std_mk=None)
        monkeypatch.setattr(setup.BuildMatrices, "T_fg_build_save", lambda **kw: pytest.fail("Unexpected T_fg"))
    result = setup.run_setup(RunConfig(**arguments, model=model, sigma=.5))
    assert result.model == model
    assert np.load(tmp_path / "T.npy").shape == (6, n_columns)
    assert (tmp_path / "T_21.npy").exists() == (model != "fg")
    assert (tmp_path / "T_fg.npy").exists() == (model != "eor")
    assert result.prior_class.bounds_log.shape == ((0, 2) if model == "fg" else (1, 2))
    assert result.phi_inverse_class.foreground_precision.size == (0 if model == "eor" else 6)
    assert (result.bm.fourier_geometry is None) == (model == "fg")


def test_fg_only_setup_accepts_kl_basis_without_cosmological_priors(tmp_path, setup_case, monkeypatch):
    setup, arguments, _ = setup_case
    path = tmp_path / "reduced.npy"
    reduced = np.ones((6, 2), dtype=np.complex128)
    np.save(path, reduced)
    arguments["priors"] = None
    monkeypatch.setattr(setup.BuildMatrices, "T_fg_build_save", lambda **kw: pytest.fail("Unexpected T_fg"))
    result = setup.run_setup(RunConfig(**arguments, model="fg", foreground_kl_path=path, sigma=.5))
    np.testing.assert_array_equal(np.load(tmp_path / "T.npy"), reduced)
    np.testing.assert_array_equal(result.phi_inverse_class.evaluate(jnp.empty(0)), np.ones(2))
    assert not (tmp_path / "T_21.npy").exists()
    assert not (tmp_path / "T_fg.npy").exists()


def test_switching_models_invalidates_joint_and_gram_caches(tmp_path, setup_case):
    setup, arguments, _ = setup_case
    for model, n_columns in (("eor+fg", 22), ("eor", 16), ("fg", 6), ("eor+fg", 22)):
        setup.run_setup(RunConfig(**arguments, model=model, sigma=.5))
        operator = np.load(tmp_path / "T.npy")
        inverse_noise = np.load(tmp_path / "Ninv.npy")
        assert operator.shape == (6, n_columns)
        np.testing.assert_allclose(np.load(tmp_path / "T_Ninv_T.npy"),
                                   operator.conj().T @ (inverse_noise[:, None] * operator))


def test_fg_only_rejects_empty_kl_basis(tmp_path, setup_case):
    setup, arguments, _ = setup_case
    path = tmp_path / "empty.npy"
    np.save(path, np.empty((6, 0), dtype=np.complex128))
    with pytest.raises(ValueError, match="at least one retained foreground mode"):
        setup.run_setup(RunConfig(**arguments, model="fg", foreground_kl_path=path))


@pytest.mark.parametrize("options", [dict(model="unknown"),
                                    dict(model="eor", foreground_kl_path="unused.npy"),
                                    dict(model="fg", use_intrinsic_noise_fitting=True)])
def test_model_selection_rejects_incompatible_options(setup_case, options):
    setup, arguments, _ = setup_case
    with pytest.raises(ValueError):
        setup.run_setup(RunConfig(**arguments, **options))
