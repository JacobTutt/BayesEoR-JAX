"""Configuration is typed, serializable and separate from resolved inputs."""

from dataclasses import asdict, replace
import json

import pytest

from bayeseor_jax import BayesEoRParser, ModelConfig, RunConfig
from bayeseor_jax.io import write_log_files


def test_parser_returns_typed_settings_and_applies_overrides():
    config = BayesEoRParser().parse_args([
        "--config", "test_examples/eor/config.yaml",
        "--model", "fg", "--fov-fg", "180", "--nside", "64",
        "--no-form-pi", "--run", "--random-key", "42",
    ])
    assert isinstance(config, RunConfig)
    assert isinstance(config, ModelConfig)
    assert config.model == "fg"
    assert config.fov_fg == 180
    assert config.nside == 64
    assert not config.form_pi
    assert config.run
    assert config.random_key == 42
    changed = replace(config, nside=128)
    assert config.nside == 64 and changed.nside == 128


def test_explicit_dictionary_parser_remains_available():
    values = BayesEoRParser(parse_as_dict=True).parse_args([
        "--config", "test_examples/eor/config.yaml",
    ])
    assert isinstance(values, dict)
    assert "config" not in values
    assert asdict(RunConfig(**values)) == values


def test_typo_cannot_be_silently_ignored_by_config():
    with pytest.raises(TypeError, match="unexpected keyword"):
        RunConfig(nsied=64)


def test_parser_allows_automatic_image_and_fourier_resolution():
    config = BayesEoRParser().parse_args([
        "--n-freq", "3", "--f0-mhz", "150", "--df-mhz", "1",
        "--telescope-latlonalt", "[0, 0, 0]",
    ])
    assert config.nside is None
    assert config.n_u_eor is None
    assert config.n_v_eor is None
    assert RunConfig().nside is None


def test_config_metadata_serializes_without_parser_namespace(tmp_path):
    parser = BayesEoRParser()
    config = parser.parse_args(["--config", "test_examples/eor/config.yaml"])
    write_log_files(config, out_dir=tmp_path)
    saved = json.loads((tmp_path / "args.json").read_text())
    assert saved["nside"] == config.nside
    assert saved["model"] == config.model
    assert saved["beta_fg"] == list(config.beta_fg)
