"""Tests for ``bayeseor_jax.model.instrument.noise``: the likelihood noise covariance."""

from __future__ import annotations

import numpy as np
import pytest

from bayeseor_jax.model.instrument.noise import build_noise_variance


class TestBuildNoiseVariance:
    """Diagonal complex-visibility noise variances."""

    def test_sigma_over_redundancy_broadcast_over_frequency(self):
        """Each row is sigma^2 / R_tb, repeated across frequencies in t/f/b order."""
        red = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 2.0]])
        out = np.asarray(build_noise_variance(2, 3, 3, sigma=2.0, redundancy=red))
        expected = np.repeat((4.0 / red)[:, None, :], 3, axis=1).reshape(-1)
        np.testing.assert_allclose(out, expected)

    def test_accepts_trailing_singleton_redundancy(self):
        """Redundancy shaped (n_t, n_b, 1) gives the same variances."""
        red = np.ones((2, 3))
        a = build_noise_variance(2, 1, 3, sigma=1.0, redundancy=red)
        b = build_noise_variance(2, 1, 3, sigma=1.0, redundancy=red[..., None])
        np.testing.assert_array_equal(a, b)

    def test_effective_noise_overrides_sigma(self, rng):
        """Per-visibility amplitudes replace the sigma model with |a|^2."""
        amp = rng.normal(size=12) + 1j * rng.normal(size=12)
        out = np.asarray(build_noise_variance(2, 2, 3, sigma=99.0, effective_noise=amp))
        np.testing.assert_allclose(out, np.abs(amp) ** 2)

    def test_float64_output(self):
        """Variances are returned as float64."""
        out = build_noise_variance(1, 1, 2, sigma=1.0, redundancy=np.ones((1, 2)))
        assert out.dtype == np.float64

    @pytest.mark.parametrize("dims", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_non_positive_dimensions_raise(self, dims):
        """Any non-positive observation dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimensions"):
            build_noise_variance(*dims, sigma=1.0, redundancy=np.ones((1, 1)))

    @pytest.mark.parametrize("sigma", [None, 0.0, -1.0, np.inf, np.nan])
    def test_invalid_sigma_raises(self, sigma):
        """Without effective noise, sigma must be finite and positive."""
        with pytest.raises(ValueError, match="sigma"):
            build_noise_variance(1, 1, 2, sigma=sigma, redundancy=np.ones((1, 2)))

    def test_wrong_redundancy_shape_raises(self):
        """Redundancy not shaped (n_t, n_b) raises ValueError."""
        with pytest.raises(ValueError, match="redundancy must have shape"):
            build_noise_variance(2, 1, 3, sigma=1.0, redundancy=np.ones((3, 2)))

    @pytest.mark.parametrize("value", [0.0, -1.0, np.inf])
    def test_non_positive_redundancy_raises(self, value):
        """Redundancies must be finite and positive."""
        red = np.ones((1, 2))
        red[0, 1] = value
        with pytest.raises(ValueError, match="finite and positive"):
            build_noise_variance(1, 1, 2, sigma=1.0, redundancy=red)

    def test_wrong_effective_noise_length_raises(self):
        """Effective noise not of length n_t*n_f*n_b raises ValueError."""
        with pytest.raises(ValueError, match="effective_noise must have shape"):
            build_noise_variance(1, 2, 2, effective_noise=np.ones(5))

    def test_zero_effective_noise_raises(self):
        """A zero effective amplitude gives a non-positive variance and raises."""
        with pytest.raises(ValueError, match="finite and positive"):
            build_noise_variance(1, 1, 2, effective_noise=np.array([1.0, 0.0]))
