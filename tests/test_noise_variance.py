import numpy as np
import pytest

from bayeseor_jax.data.noise import build_noise_variance


def test_variance_matches_time_frequency_baseline_order():
    redundancy = np.array([[1., 2.], [4., 8.]])
    expected = np.array([4., 2., 4., 2., 1., .5, 1., .5])
    for red in (redundancy, redundancy[..., None]):
        np.testing.assert_allclose(
            build_noise_variance(2, 2, 2, sigma=2., redundancy=red), expected,
        )


def test_effective_noise_overrides_sigma_and_redundancy():
    np.testing.assert_allclose(
        build_noise_variance(1, 2, 1, effective_noise=[2., 3j]), [4., 9.],
    )


@pytest.mark.parametrize("kwargs", [
    {"effective_noise": [1.]}, {"effective_noise": [0., 1.]},
    {"effective_noise": [np.nan, 1.]},
    {"sigma": -1., "redundancy": [[1.]]},
    {"sigma": 1., "redundancy": [[0.]]},
    {"sigma": 1., "redundancy": [[np.inf]]},
    {"sigma": 1., "redundancy": [1.]},
])
def test_invalid_noise_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_noise_variance(1, 2, 1, **kwargs)
