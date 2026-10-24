"""Tests for ``bayeseor_jax.setup``."""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import replace

import numpy as np
import pytest

from bayeseor_jax.configuration import RunConfig
from bayeseor_jax.setup import SetupResult, run_setup

# The fixture grid (nside=16) is deliberately coarser than the Nyquist estimate.
pytestmark = pytest.mark.filterwarnings("ignore:nside=16 is below:RuntimeWarning")

N_VIS = 192            # 4 times x 4 channels x 12 signed baselines
N_FG = 66              # 22 pixels x 3 spectral components
N_EOR = 348            # retained EoR Fourier coefficients
THETA = np.array([1.0, 1.0])


@pytest.fixture(scope="module")
def setups(tmp_path_factory, tiny_npy_dir, hera_latlonalt):
    """Run setup once per model on the ``.npy`` fixture and return (result, config, dir)."""

    def config_in(directory, **overrides):
        model = overrides.get("model", "eor+fg")
        base = RunConfig(
            model=model, data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            n_freq=4, f0_mhz=150.0, df_mhz=0.1, n_t=4, d_t=10.0, jd_min=2459999.1,
            telescope_latlonalt=hera_latlonalt, beam_type="airy", diam=14.0,
            fov_eor=20.0, fov_fg=20.0, nside=16, sigma=1.0,
            priors=None if model == "fg" else [[-2.0, 4.0], [-2.0, 4.0]],
            array_save_directory=directory / "matrices", output_dir=directory / "output",
            save_dir=directory / "saved",
        )
        return replace(base, **overrides)

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in ("eor", "fg", "eor+fg"):
            directory = tmp_path_factory.mktemp(model.replace("+", "_"))
            config = config_in(directory, model=model)
            results[model] = (run_setup(config), config, directory)
        directory = tmp_path_factory.mktemp("kl")
        config = config_in(directory, foreground_kl_path=directory / "kl" / "T_fg_reduced.npy")
        results["kl"] = (run_setup(config, build_foreground_kl=True), config, directory)
    return results


@pytest.mark.slow
class TestRunSetupOutputs:
    """Results and saved products of a successful setup."""

    @pytest.mark.parametrize("model", ["eor", "fg", "eor+fg"])
    def test_returns_setup_result(self, setups, model):
        """Setup returns a ``SetupResult`` carrying the requested model and configuration."""
        result, config, _ = setups[model]
        assert isinstance(result, SetupResult)
        assert result.model == model and result.config is config

    @pytest.mark.parametrize(
        ("model", "n_columns"), [("eor", N_EOR), ("fg", N_FG), ("eor+fg", N_EOR + N_FG)],
    )
    def test_joint_operator_columns(self, setups, model, n_columns):
        """The saved joint operator has one row per visibility and one column per coefficient."""
        _, _, directory = setups[model]
        assert np.load(directory / "matrices" / "T.npy", mmap_mode="r").shape == (N_VIS, n_columns)

    def test_joint_operator_is_concatenation(self, setups):
        """``T.npy`` equals the EoR columns followed by the foreground columns."""
        _, _, directory = setups["eor+fg"]
        m = directory / "matrices"
        expected = np.hstack([np.load(m / "T_21.npy"), np.load(m / "T_fg.npy")])
        np.testing.assert_array_equal(np.load(m / "T.npy"), expected)

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("eor", {"T_21.npy"}),
            ("fg", {"T_fg.npy"}),
            ("eor+fg", {"T_21.npy", "T_fg.npy"}),
        ],
    )
    def test_matrix_files(self, setups, model, expected):
        """Only the selected component operators are built, alongside the likelihood products."""
        _, _, directory = setups[model]
        names = {p.name for p in (directory / "matrices").glob("*.npy")}
        common = {"T.npy", "Ninv.npy", "T_Ninv_T.npy", "dbar.npy", "d_ninv_d.npy"}
        assert names == common | expected

    @pytest.mark.parametrize(("model", "has_k_files"), [("eor", True), ("fg", False), ("eor+fg", True)])
    def test_output_metadata(self, setups, model, has_k_files):
        """``args.json`` is always written; k-bin files only when the EoR is modelled."""
        _, _, directory = setups[model]
        out = directory / "output"
        assert (out / "args.json").exists()
        assert (out / "k-vals.txt").exists() is has_k_files

    def test_args_json_records_resolved_values(self, setups):
        """The saved metadata records resolved observation and model settings."""
        _, _, directory = setups["eor+fg"]
        args = json.loads((directory / "output" / "args.json").read_text())
        assert args["model"] == "eor+fg"
        assert args["n_freq"] == 4 and args["n_t"] == 4 and args["n_eta"] == 4
        assert args["nside"] == 16 and args["nside_required"] == 32
        assert args["f0_mhz"] == pytest.approx(150.0) and args["df_mhz"] == pytest.approx(0.1)
        assert args["foreground_kl_built"] is False
        assert args["foreground_prior_std_mk"] == pytest.approx(list(RunConfig().foreground_prior_std_mk))

    def test_k_vals_match_metadata(self, setups):
        """The returned k values match ``k-vals.txt`` and are positive and increasing."""
        result, _, directory = setups["eor"]
        k = np.asarray(result.k_vals)
        np.testing.assert_allclose(np.loadtxt(directory / "output" / "k-vals.txt"), k)
        assert k.size == 2 and np.all(k > 0) and np.all(np.diff(k) > 0)

    def test_fg_only_has_no_power_parameters(self, setups):
        """A foreground-only model samples no power-spectrum parameters."""
        result, _, _ = setups["fg"]
        assert np.asarray(result.k_vals).size == 0
        assert result.prior_class.n_power_params == 0
        assert result.phi_inverse_class.n_eor_voxels == 0

    @pytest.mark.parametrize(("model", "n_fg"), [("eor", 0), ("fg", N_FG), ("eor+fg", N_FG)])
    def test_foreground_prior_precision(self, setups, model, n_fg):
        """Physical foreground precisions are the inverse squared prior widths per pixel."""
        result, config, _ = setups[model]
        precision = np.asarray(result.phi_inverse_class.foreground_precision)
        expected = 1.0 / np.repeat(config.foreground_prior_std_mk, n_fg // 3) ** 2 if n_fg else np.empty(0)
        np.testing.assert_allclose(precision, expected)

    def test_inverse_noise_from_sigma(self, setups):
        """``Ninv.npy`` equals redundancy / sigma² for every visibility."""
        result, config, directory = setups["eor+fg"]
        redundancy = np.asarray(result.vis_dict["redundancy"]).reshape(4, -1)
        expected = np.broadcast_to(redundancy[:, None, :] / config.sigma**2, (4, 4, redundancy.shape[1]))
        np.testing.assert_allclose(np.load(directory / "matrices" / "Ninv.npy"), expected.reshape(-1))

    def test_noise_is_added(self, setups):
        """Without a noise file the loaded data are treated as noiseless and noise is added."""
        result, _, _ = setups["eor"]
        vis = np.asarray(result.vis_dict["vis"])
        np.testing.assert_allclose(np.asarray(result.vis_dict["vis_noisy"]) - vis,
                                   np.asarray(result.vis_dict["noise"]))
        assert not np.allclose(result.vis_dict["noise"], 0)

    @pytest.mark.parametrize(("model", "theta"), [("eor", THETA), ("eor+fg", THETA), ("fg", np.empty(0))])
    def test_likelihood_is_finite(self, setups, model, theta):
        """The marginalised likelihood is finite inside the prior."""
        result, _, _ = setups[model]
        assert np.isfinite(float(result.likelihood_class.log_likelihood(theta)))

    def test_config_is_not_mutated(self, setups, tiny_npy_dir):
        """Setup resolves values locally and leaves the configuration unchanged."""
        result, config, _ = setups["eor+fg"]
        assert result.config.n_eta is None and result.config.jd_center is None
        assert dataclasses.asdict(result.config) == dataclasses.asdict(config)

    def test_box_sizes_positive(self, setups):
        """The comoving box dimensions are finite and positive."""
        result, _, _ = setups["eor"]
        for size in (result.ps_box_size_ra_mpc, result.ps_box_size_dec_mpc, result.ps_box_size_para_mpc):
            assert np.isfinite(size) and size > 0


@pytest.mark.slow
class TestRunSetupKL:
    """Building and loading a foreground KL reduced basis during setup."""

    def test_builds_reduced_basis(self, setups):
        """The reduced operator, its singular values and diagnostics are written."""
        _, _, directory = setups["kl"]
        names = {p.name for p in (directory / "kl").iterdir()}
        assert names == {"T_fg_reduced.npy", "T_fg_reduced.singular_values.npy", "T_fg_reduced.svd.json"}

    def test_reduced_columns_have_unit_prior(self, setups):
        """Joint columns are EoR plus retained KL modes, the latter with unit prior precision."""
        result, _, directory = setups["kl"]
        rank = np.load(directory / "kl" / "T_fg_reduced.npy").shape[1]
        assert 0 < rank <= N_FG
        assert np.load(directory / "matrices" / "T.npy", mmap_mode="r").shape == (N_VIS, N_EOR + rank)
        np.testing.assert_array_equal(np.asarray(result.phi_inverse_class.foreground_precision), np.ones(rank))

    def test_args_record_kl(self, setups):
        """The metadata records that the basis was built and its threshold."""
        _, _, directory = setups["kl"]
        args = json.loads((directory / "output" / "args.json").read_text())
        assert args["foreground_kl_built"] is True and args["foreground_kl_f"] == pytest.approx(0.1)

    def test_reuses_existing_basis(self, setups, tiny_config):
        """A saved basis is loaded unchanged when not rebuilding, with a unit prior."""
        _, _, directory = setups["kl"]
        kl_path = directory / "kl" / "T_fg_reduced.npy"
        before = kl_path.stat().st_mtime_ns
        result = run_setup(tiny_config(foreground_kl_path=kl_path))
        assert kl_path.stat().st_mtime_ns == before
        rank = np.load(kl_path).shape[1]
        np.testing.assert_array_equal(np.asarray(result.phi_inverse_class.foreground_precision), np.ones(rank))

    def test_kl_likelihood_close_to_physical(self, setups):
        """The reduced and physical joint likelihoods agree to well within one unit."""
        physical = float(setups["eor+fg"][0].likelihood_class.log_likelihood(THETA))
        reduced = float(setups["kl"][0].likelihood_class.log_likelihood(THETA))
        assert abs(physical - reduced) < 0.5


class TestRunSetupValidation:
    """Invalid configurations are rejected with informative errors."""

    def test_requires_run_config(self):
        """A plain dictionary is rejected."""
        with pytest.raises(TypeError, match="RunConfig"):
            run_setup({"model": "eor"})

    @pytest.mark.parametrize(
        ("overrides", "kwargs", "match"),
        [
            ({"model": "joint"}, {}, "model must be"),
            ({}, {"build_foreground_kl": True}, "foreground_kl_path"),
            ({"model": "eor"}, {"build_foreground_kl": True}, "KL construction"),
            ({"model": "eor", "foreground_kl_path": "kl.npy"}, {}, "cannot be used with model='eor'"),
            ({"model": "fg", "use_intrinsic_noise_fitting": True}, {}, "fixed noise"),
            ({"telescope_latlonalt": (0.0, 0.0)}, {}, "exactly 3 values"),
            ({"beta_fg": ()}, {}, "beta_fg"),
            ({"foreground_prior_std_mk": (1.0, 2.0)}, {}, "foreground_prior_std_mk"),
        ],
    )
    def test_rejects_before_loading_data(self, tiny_config, overrides, kwargs, match):
        """Inconsistent options are rejected before any data are loaded."""
        with pytest.raises(ValueError, match=match):
            run_setup(tiny_config(**overrides), **kwargs)

    def test_rejects_invalid_kl_threshold(self, tiny_config, tmp_path):
        """A negative KL threshold is rejected before construction."""
        with pytest.raises(ValueError):
            run_setup(tiny_config(foreground_kl_path=tmp_path / "kl.npy"), build_foreground_kl=True, kl_f=-1.0)

    def test_kl_path_must_not_be_setup_output(self, tiny_config, tmp_path):
        """The KL operator path may not coincide with a setup product."""
        config = tiny_config(foreground_kl_path=tmp_path / "matrices" / "T.npy")
        with pytest.raises(ValueError, match="setup output path"):
            run_setup(config)

    def test_kl_path_must_not_overwrite_data(self, tiny_config, tiny_npy_dir):
        """Building a basis may not overwrite the input data file."""
        config = tiny_config(foreground_kl_path=tiny_npy_dir / "vis_vector.npy")
        with pytest.raises(ValueError, match="must not overwrite"):
            run_setup(config, build_foreground_kl=True)

    def test_kl_must_be_complex_matrix(self, tiny_config, tmp_path):
        """A supplied KL operator must be a complex128 matrix."""
        np.save(tmp_path / "kl.npy", np.ones((N_VIS, 3)))
        with pytest.raises(ValueError, match="complex128 matrix"):
            run_setup(tiny_config(foreground_kl_path=tmp_path / "kl.npy"))

    def test_fg_only_needs_kl_modes(self, tiny_config, tmp_path):
        """A foreground-only model with an empty KL basis is rejected."""
        np.save(tmp_path / "kl.npy", np.ones((N_VIS, 0), dtype=complex))
        with pytest.raises(ValueError, match="at least one retained"):
            run_setup(tiny_config(model="fg", foreground_kl_path=tmp_path / "kl.npy"))

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"model": "eor", "priors": None}, "priors must be provided"),
            ({"model": "eor", "priors": [[0.0, 1.0]]}, "non-empty bins"),
            ({"model": "eor", "priors": [0.0, 1.0]}, "shape"),
            ({"model": "eor", "uniform_priors": [1, 0, 1]}, "uniform_priors"),
            ({"nside": 0}, "nside"),
            ({"nside": 16.5}, "nside"),
        ],
    )
    def test_rejects_invalid_model_settings(self, tiny_config, overrides, match):
        """Invalid priors, prior masks or resolutions are rejected."""
        with pytest.raises(ValueError, match=match):
            run_setup(tiny_config(**overrides))

    def test_rejects_wrong_effective_noise_length(self, tiny_config, tmp_path):
        """An effective-noise vector of the wrong length is rejected."""
        np.save(tmp_path / "noise.npy", np.ones(5))
        with pytest.raises(ValueError, match="effective_noise_path length"):
            run_setup(tiny_config(model="eor", effective_noise_path=tmp_path / "noise.npy"))

    @pytest.mark.slow
    def test_kl_rows_must_match_data(self, tiny_config, tmp_path):
        """A KL operator with the wrong number of rows is rejected."""
        np.save(tmp_path / "kl.npy", np.ones((N_VIS - 1, 3), dtype=complex))
        with pytest.raises(ValueError, match="visibility rows"):
            run_setup(tiny_config(foreground_kl_path=tmp_path / "kl.npy"))


@pytest.mark.slow
class TestRunSetupOptions:
    """Optional settings that change the setup products."""

    @pytest.mark.filterwarnings("default:nside=16 is below:RuntimeWarning")
    def test_warns_below_nyquist_nside(self, tiny_config, tmp_path):
        """An explicit nside below the Nyquist estimate warns but is used."""
        with pytest.warns(RuntimeWarning, match="below the baseline-driven Nyquist"):
            run_setup(tiny_config(model="fg"))
        assert json.loads((tmp_path / "output" / "args.json").read_text())["nside"] == 16

    def test_effective_noise_sets_weights(self, tiny_config, tmp_path):
        """Per-visibility effective noise amplitudes set the inverse noise variances."""
        amplitudes = np.linspace(0.5, 2.0, N_VIS)
        np.save(tmp_path / "noise.npy", amplitudes)
        run_setup(tiny_config(model="fg", effective_noise_path=tmp_path / "noise.npy"))
        np.testing.assert_allclose(np.load(tmp_path / "matrices" / "Ninv.npy"), 1.0 / amplitudes**2)

    def test_uniform_priors_mask(self, tiny_config):
        """A per-bin uniform-prior mask is passed to the likelihood."""
        result = run_setup(tiny_config(model="eor", uniform_priors=[1, 0]))
        np.testing.assert_array_equal(np.asarray(result.likelihood_class.uniform_priors), [True, False])
