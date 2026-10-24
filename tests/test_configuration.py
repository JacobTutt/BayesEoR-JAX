"""Tests for ``bayeseor_jax.configuration``."""

from __future__ import annotations

import dataclasses

import pytest

from bayeseor_jax.configuration import BayesEoRParser, ModelConfig, RunConfig

#: Minimal command-line tokens supplying every required option.
REQUIRED_ARGS = [
    "--n-freq", "4",
    "--f0-mhz", "150.0",
    "--df-mhz", "0.1",
    "--telescope-latlonalt", "[-30.7, 21.4, 1073.0]",
]


@pytest.fixture
def yaml_config(tmp_path):
    """Write a YAML configuration file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: eor\n"
        "n_freq: 8\n"
        "f0_mhz: 158.0\n"
        "df_mhz: 0.25\n"
        "telescope_latlonalt: [-30.7, 21.4, 1073.0]\n"
        "beam_type: Airy\n"
        "diam: 14.2\n"
        "pol: XX\n"
        "priors: [[-1, 6], [0, 5]]\n"
        "beta_fg: [2.5]\n"
    )
    return path


class TestBayesEoRParser:
    """Parsing YAML files and command-line flags into a ``RunConfig``."""

    def test_returns_run_config(self):
        """Parsing the required options returns a typed ``RunConfig``."""
        config = BayesEoRParser().parse_args(REQUIRED_ARGS)
        assert isinstance(config, RunConfig)

    def test_required_values_are_typed(self):
        """Required options are converted to their declared types."""
        config = BayesEoRParser().parse_args(REQUIRED_ARGS)
        assert config.n_freq == 4 and isinstance(config.n_freq, int)
        assert config.f0_mhz == pytest.approx(150.0)
        assert config.df_mhz == pytest.approx(0.1)
        assert list(config.telescope_latlonalt) == pytest.approx([-30.7, 21.4, 1073.0])

    @pytest.mark.parametrize("missing", ["--n-freq", "--f0-mhz", "--df-mhz", "--telescope-latlonalt"])
    def test_missing_required_option_exits(self, missing):
        """Omitting any required option is rejected by the parser."""
        index = REQUIRED_ARGS.index(missing)
        args = REQUIRED_ARGS[:index] + REQUIRED_ARGS[index + 2:]
        with pytest.raises(SystemExit):
            BayesEoRParser().parse_args(args)

    def test_reads_yaml(self, yaml_config):
        """Values in a YAML file populate the configuration."""
        config = BayesEoRParser().parse_args(["--config", str(yaml_config)])
        assert config.model == "eor"
        assert config.n_freq == 8
        assert config.diam == pytest.approx(14.2)
        assert [list(p) for p in config.priors] == [[-1.0, 6.0], [0.0, 5.0]]
        assert list(config.beta_fg) == [2.5]

    def test_cli_overrides_yaml(self, yaml_config):
        """A flag given after the YAML file overrides the file's value."""
        config = BayesEoRParser().parse_args(["--config", str(yaml_config), "--n-freq", "3"])
        assert config.n_freq == 3

    def test_lowercases_beam_and_polarisation(self, yaml_config):
        """Beam type and polarisation strings are lower-cased."""
        config = BayesEoRParser().parse_args(["--config", str(yaml_config)])
        assert config.beam_type == "airy"
        assert config.pol == "xx"

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [([], True), (["--form-pi"], True), (["--no-form-pi"], False)],
    )
    def test_form_pi_flags(self, flags, expected):
        """``--form-pi`` defaults to True and ``--no-form-pi`` turns it off."""
        assert BayesEoRParser().parse_args(REQUIRED_ARGS + flags).form_pi is expected

    @pytest.mark.parametrize(
        "flag", ["--quiet", "--clobber", "--run", "--resume", "--redundant-avg",
                 "--uniform-redundancy", "--calc-noise", "--save-vis", "--save-model",
                 "--use-intrinsic-noise-fitting"],
    )
    def test_store_true_flags(self, flag):
        """Boolean switches default to False and become True when given."""
        field = flag[2:].replace("-", "_")
        assert getattr(BayesEoRParser().parse_args(REQUIRED_ARGS), field) is False
        assert getattr(BayesEoRParser().parse_args(REQUIRED_ARGS + [flag]), field) is True

    def test_invalid_model_choice_exits(self):
        """A model outside ``eor``, ``fg`` and ``eor+fg`` is rejected."""
        with pytest.raises(SystemExit):
            BayesEoRParser().parse_args(REQUIRED_ARGS + ["--model", "foregrounds"])

    def test_cli_output_dir_default(self):
        """The CLI keeps its historical ``./BayesEoROutputs/`` output directory."""
        assert BayesEoRParser().parse_args(REQUIRED_ARGS).output_dir == "./BayesEoROutputs/"

    def test_defaults_match_run_config(self):
        """Every other parser default equals the corresponding ``RunConfig`` default."""
        parsed = dataclasses.asdict(BayesEoRParser().parse_args(REQUIRED_ARGS))
        defaults = dataclasses.asdict(RunConfig())
        skip = {"n_freq", "f0_mhz", "df_mhz", "telescope_latlonalt", "output_dir"}
        for name, value in defaults.items():
            if name in skip:
                continue
            got = parsed[name]
            if isinstance(value, tuple):
                assert list(got) == pytest.approx(list(value)), name
            else:
                assert got == value, name

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_parse_as_dict(self):
        """``parse_as_dict=True`` returns a plain dictionary of the same settings."""
        result = BayesEoRParser(parse_as_dict=True).parse_args(REQUIRED_ARGS)
        assert isinstance(result, dict)
        assert set(result) == {f.name for f in dataclasses.fields(RunConfig)}
        assert result["n_freq"] == 4

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_config_key_is_removed(self, yaml_config):
        """The YAML path itself is not stored in the resulting configuration."""
        result = BayesEoRParser(parse_as_dict=True).parse_args(["--config", str(yaml_config)])
        assert "config" not in result


class TestModelConfig:
    """Defaults and immutability of ``ModelConfig``."""

    def test_defaults(self):
        """The physical-model defaults are the documented values."""
        config = ModelConfig()
        assert config.model == "eor+fg"
        assert config.beta_fg == (2.63, 2.82)
        assert config.beam_type == "uniform"
        assert config.peak_amp == 1.0
        assert config.pol == "xx"
        assert config.freq_interp_kind == "cubic"
        assert config.array_save_directory == "./matrices/"

    def test_is_frozen(self):
        """Fields cannot be reassigned after construction."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            ModelConfig().model = "eor"

    def test_requires_keywords(self):
        """Positional construction is rejected."""
        with pytest.raises(TypeError):
            ModelConfig("eor")


class TestRunConfig:
    """Defaults, inheritance and immutability of ``RunConfig``."""

    def test_extends_model_config(self):
        """``RunConfig`` contains every ``ModelConfig`` field."""
        model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        run_fields = {f.name for f in dataclasses.fields(RunConfig)}
        assert issubclass(RunConfig, ModelConfig)
        assert model_fields <= run_fields

    def test_defaults(self):
        """Selected run defaults have their documented values."""
        config = RunConfig()
        assert config.n_t == 1
        assert config.fov_eor == 10.0
        assert config.antenna_selection == "cross"
        assert config.form_pi is True
        assert config.pi_norm == 1.0
        assert config.noise_seed == 742123
        assert config.n_live_scale_factor == 25.0
        assert config.random_key == 0
        assert len(config.foreground_prior_std_mk) == 1 + len(config.beta_fg)

    def test_optional_settings_default_to_none(self):
        """Settings resolved from the observation default to ``None``."""
        config = RunConfig()
        for name in ("data_path", "inst_model", "n_u_eor", "n_v_eor", "du_eor", "dv_eor",
                     "n_freq", "f0_mhz", "df_mhz", "n_eta", "d_t", "jd_center", "nside",
                     "footprint_nside", "fov_fg", "sigma", "priors", "foreground_kl_path"):
            assert getattr(config, name) is None, name

    def test_is_frozen(self):
        """Fields cannot be reassigned after construction."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            RunConfig().sigma = 1.0

    def test_replace_returns_new_object(self):
        """``dataclasses.replace`` derives a new configuration, leaving the original unchanged."""
        original = RunConfig(sigma=1.0)
        derived = dataclasses.replace(original, sigma=2.0)
        assert original.sigma == 1.0 and derived.sigma == 2.0
