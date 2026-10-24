"""Tests for ``bayeseor_jax.inference.prior``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.inference.prior import (
    Prior,
    linear_uniform_log_jacobian,
    parse_uprior_inds,
    theta_power_to_linear,
)

BOUNDS = np.array([[-1.0, 3.0], [0.5, 2.5], [2.0, 6.0]])


@pytest.fixture
def prior() -> Prior:
    """Three-parameter box prior with unequal widths."""
    return Prior(BOUNDS)


class TestPrior:
    def test_stores_bounds(self, prior):
        """The lower and upper bounds are the columns of ``bounds_log``."""
        np.testing.assert_array_equal(prior.lower, BOUNDS[:, 0])
        np.testing.assert_array_equal(prior.upper, BOUNDS[:, 1])
        np.testing.assert_array_equal(prior.lower_numpy, BOUNDS[:, 0])
        np.testing.assert_array_equal(prior.upper_numpy, BOUNDS[:, 1])

    def test_n_power_params_defaults_to_bound_count(self, prior):
        """Omitting ``n_power_params`` infers it from the number of bounds."""
        assert prior.n_power_params == 3
        assert prior.power_param_start == 0

    @pytest.mark.parametrize("bounds", [np.zeros(3), np.zeros((3, 3)), np.zeros((2, 2, 2))])
    def test_rejects_bad_bound_shape(self, bounds):
        """Bounds must have shape ``(n, 2)``."""
        with pytest.raises(ValueError, match="shape"):
            Prior(bounds)

    def test_rejects_mismatched_parameter_count(self):
        """An explicit ``n_power_params`` must match the number of bounds."""
        with pytest.raises(ValueError, match="must match"):
            Prior(BOUNDS, n_power_params=2)

    def test_logprior_inside_is_normalised_box(self, prior):
        """Inside the box the log density is ``-sum(log(width))``."""
        theta = BOUNDS.mean(axis=1)
        expected = -np.sum(np.log(BOUNDS[:, 1] - BOUNDS[:, 0]))
        np.testing.assert_allclose(float(prior.logprior(jnp.asarray(theta))), expected)

    @pytest.mark.parametrize("index,value", [(0, -1.5), (1, 2.6), (2, 7.0)])
    def test_logprior_outside_is_minus_infinity(self, prior, index, value):
        """Any coordinate outside its bounds gives zero prior density."""
        theta = BOUNDS.mean(axis=1)
        theta[index] = value
        assert float(prior.logprior(jnp.asarray(theta))) == -np.inf

    @pytest.mark.parametrize("cube", [np.zeros(3), np.ones(3), np.array([0.25, 0.5, 0.75])])
    def test_transform_unit_cube_is_affine(self, prior, cube):
        """The unit cube maps linearly onto the prior box."""
        expected = BOUNDS[:, 0] + cube * (BOUNDS[:, 1] - BOUNDS[:, 0])
        np.testing.assert_allclose(prior.transform_unit_cube(cube), expected)

    def test_sample_from_priors_shape_and_support(self, prior):
        """Samples have shape ``(n_samples, n_dim)`` and lie inside the box."""
        samples = np.asarray(prior.sample_from_priors(jax.random.PRNGKey(0), 500))
        assert samples.shape == (500, 3)
        assert np.all(samples >= BOUNDS[:, 0]) and np.all(samples <= BOUNDS[:, 1])

    def test_sample_from_priors_is_uniform(self, prior):
        """Sample means approach the box centres for a uniform prior."""
        samples = np.asarray(prior.sample_from_priors(jax.random.PRNGKey(1), 20000))
        widths = BOUNDS[:, 1] - BOUNDS[:, 0]
        np.testing.assert_allclose(samples.mean(axis=0), BOUNDS.mean(axis=1), atol=0.03 * widths.max())


class TestThetaPowerToLinear:
    def test_exponentiates_base_ten(self):
        """Sampler coordinates are ``log10`` of the linear power."""
        theta = jnp.array([-1.0, 0.0, 2.5])
        np.testing.assert_allclose(theta_power_to_linear(theta), 10.0 ** np.array([-1.0, 0.0, 2.5]))


class TestLinearUniformLogJacobian:
    def test_sums_selected_bins(self):
        """The Jacobian is ``ln(10) * sum(theta)`` over the selected bins."""
        theta = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([True, False, True])
        np.testing.assert_allclose(linear_uniform_log_jacobian(theta, mask), np.log(10.0) * 4.0)

    def test_equals_log_of_linear_power(self):
        """The correction equals ``sum(log(power_linear))`` over selected bins."""
        theta = jnp.array([0.3, -1.2])
        mask = jnp.array([True, True])
        np.testing.assert_allclose(
            linear_uniform_log_jacobian(theta, mask), np.sum(np.log(10.0 ** np.asarray(theta)))
        )

    def test_no_selection_gives_zero(self):
        """With no bins selected the correction vanishes."""
        assert float(linear_uniform_log_jacobian(jnp.array([5.0, 6.0]), jnp.zeros(2, bool))) == 0.0


class TestParseUpriorInds:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("all", [True] * 5),
            ("ALL", [True] * 5),
            ("", [False] * 5),
            ("1:4", [False, True, True, True, False]),
            ("1,4", [False, True, False, False, True]),
            ("3", [False, False, False, True, False]),
            ("-1", [False, False, False, False, True]),
        ],
    )
    def test_selection_forms(self, text, expected):
        """Each supported selection string gives the expected boolean mask."""
        np.testing.assert_array_equal(parse_uprior_inds(text, 5), expected)
