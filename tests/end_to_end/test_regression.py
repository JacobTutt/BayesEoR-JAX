"""Regression checks of small end-to-end runs against stored golden values.

The expected values live in ``tests/fixtures/expected/end_to_end.json``. After an
intentional change to the numerical results, regenerate them with::

    BAYESEOR_REGEN_EXPECTED=1 python -m pytest tests/end_to_end/test_regression.py

and review the diff before committing. Log likelihoods are small differences of
large terms, so they are compared to ``1e-8`` of the cancelling scale,
``d^H N⁻¹ d + N_d``, rather than to their own value.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

EXPECTED_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "end_to_end.json"
REGENERATE = os.environ.get("BAYESEOR_REGEN_EXPECTED") == "1"
MODELS = ("eor", "fg", "eor+fg")


def _summarise(p, dense) -> dict:
    """Collect the regression quantities of one completed setup."""
    return {
        "n_data": int(p.d.size),
        "n_coefficients": int(p.T.shape[1]),
        "n_k_bins": int(np.asarray(p.setup.k_vals).size),
        "k_vals": np.asarray(p.setup.k_vals, dtype=float).tolist(),
        "scale": dense.scale(p),
        "thetas": [np.asarray(t).tolist() for t in p.thetas()],
        "log_likelihoods": [float(p.setup.likelihood_class.log_likelihood(t)) for t in p.thetas()],
    }


@pytest.fixture(scope="module")
def actual(problem, kl_problem, dense) -> dict:
    """Regression quantities of the current code, for every small configuration."""
    values = {model: _summarise(problem(model), dense) for model in MODELS}
    kl = kl_problem("eor+fg", 0.1)
    values["eor+fg_kl0.1"] = _summarise(kl, dense)
    values["eor+fg_kl0.1"]["kl_rank"] = int(kl.T.shape[1] - problem("eor").T.shape[1])
    return values


@pytest.fixture(scope="module")
def expected(actual) -> dict:
    """Stored golden values, rewritten first when ``BAYESEOR_REGEN_EXPECTED=1``."""
    if REGENERATE:
        EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED_PATH.write_text(json.dumps(actual, indent=2) + "\n")
    return json.loads(EXPECTED_PATH.read_text())


CASES = (*MODELS, "eor+fg_kl0.1")


@pytest.mark.parametrize("case", CASES)
class TestGoldenValues:
    """Each small configuration reproduces its stored results."""

    def test_dimensions(self, actual, expected, case):
        """The data, coefficient and power-spectrum dimensions are unchanged."""
        for key in ("n_data", "n_coefficients", "n_k_bins"):
            assert actual[case][key] == expected[case][key], key

    def test_k_vals(self, actual, expected, case):
        """The representative wavenumber of each bin is unchanged."""
        np.testing.assert_allclose(actual[case]["k_vals"], expected[case]["k_vals"], rtol=1e-10)

    def test_same_parameters(self, actual, expected, case):
        """The stored values were evaluated at the same parameter vectors."""
        assert actual[case]["thetas"] == expected[case]["thetas"]

    def test_log_likelihoods(self, actual, expected, case):
        """The log likelihood at each stored parameter vector is unchanged."""
        np.testing.assert_allclose(
            actual[case]["log_likelihoods"], expected[case]["log_likelihoods"],
            rtol=0, atol=1e-8 * expected[case]["scale"],
        )


def test_kl_rank(actual, expected):
    """The number of KL modes retained at ``f = 0.1`` is unchanged."""
    assert actual["eor+fg_kl0.1"]["kl_rank"] == expected["eor+fg_kl0.1"]["kl_rank"]
