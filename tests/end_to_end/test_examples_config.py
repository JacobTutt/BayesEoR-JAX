"""The shipped example configurations parse and point to existing inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from bayeseor_jax import BayesEoRParser
from bayeseor_jax.configuration import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {"eor": "eor", "eor_fgs": "eor+fg"}


@pytest.fixture(params=sorted(EXAMPLES), ids=sorted(EXAMPLES))
def example(request, monkeypatch):
    """Parse one example's ``config.yaml`` from the repository root."""
    monkeypatch.chdir(REPO_ROOT)
    path = REPO_ROOT / "examples" / request.param / "config.yaml"
    return request.param, BayesEoRParser().parse_args(["--config", str(path)])


class TestExampleConfigs:
    """Each example configuration is complete and self-consistent."""

    def test_parses_to_run_config(self, example):
        """The YAML parses into a typed ``RunConfig``."""
        _, config = example
        assert isinstance(config, RunConfig)

    def test_model(self, example):
        """Each example fits the components its name describes."""
        name, config = example
        assert config.model == EXAMPLES[name]

    def test_data_exists(self, example):
        """The data vector and instrument model are present in the repository."""
        _, config = example
        assert (REPO_ROOT / config.data_path).is_file()
        for name in ("uvw_model.npy", "redundancy_model.npy"):
            assert (REPO_ROOT / config.inst_model / name).is_file()

    def test_priors_shape(self, example):
        """There is one ``[min, max]`` prior pair per bin, with ``min < max``."""
        _, config = example
        assert all(len(pair) == 2 and pair[0] < pair[1] for pair in config.priors)

    def test_outputs_outside_examples(self, example):
        """Matrices and sampler outputs are written outside the committed example folder."""
        _, config = example
        for path in (config.array_save_directory, config.output_dir):
            assert not Path(path).parts[0] == "examples"
