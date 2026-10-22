"""Verify that prior/noise weighting preserves the intended predictive covariance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax import T_fg, T_fg_weighted, weight_foreground_operator


@pytest.mark.parametrize("dense_prior", [False, True])
@pytest.mark.parametrize("dense_noise", [False, True])
def test_diagonal_factors_match_explicit_matrix_product(dense_prior, dense_noise):
    operator = np.array([[1 + 2j, 3], [-2j, 4], [5, 1 - 1j]])
    prior_std = np.array([2., 5.])
    inverse_noise_std = np.array([.5, 2., .25])
    result = jax.jit(weight_foreground_operator)(
        operator,
        np.diag(prior_std) if dense_prior else prior_std,
        np.diag(inverse_noise_std) if dense_noise else inverse_noise_std,
    )
    expected = np.diag(inverse_noise_std) @ operator @ np.diag(prior_std)
    np.testing.assert_allclose(result, expected, atol=1e-13)


def test_correlated_factors_give_whitened_prior_predictive_covariance():
    operator = np.array([[1 + 2j, 3], [-2j, 4], [5, 1 - 1j]])
    prior_root = np.array([[2., 0.], [1 + 1j, 3.]])
    noise_root = np.array([[2., 0., 0.], [1j, 3., 0.], [1., -1j, 4.]])
    noise_whitener = np.linalg.inv(noise_root)
    prior = prior_root @ prior_root.conj().T
    noise = noise_root @ noise_root.conj().T
    np.testing.assert_allclose(noise_whitener @ noise @ noise_whitener.conj().T,
                               np.eye(3), atol=1e-13)
    weighted = np.asarray(jax.jit(weight_foreground_operator)(
        operator, prior_root, noise_whitener,
    ))
    expected = noise_whitener @ operator @ prior @ operator.conj().T @ noise_whitener.conj().T
    np.testing.assert_allclose(weighted @ weighted.conj().T, expected, atol=1e-13)
    # Both decomposition routes must recover the same nonzero variance spectrum.
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    eigenvalues = np.linalg.eigvalsh(expected)[::-1]
    np.testing.assert_allclose(eigenvalues[:2], singular_values**2, atol=1e-13)
    assert abs(eigenvalues[-1]) < 1e-13


def test_wrapper_matches_cached_operator_and_selected_visibility_block():
    lmn = np.array([[[0., .6], [0., 0.], [1., .8]]])
    uvw = np.array([[[1.], [0.], [0.]], [[2.], [0.], [0.]]])
    beam = np.array([[[1., .5], [.8, .2]]])
    spectral = np.array([[1., 1.], [1., .4]])
    prior_std = np.array([2., 3., 4., 5.])
    inverse_noise_std = np.array([.5, 2.])
    physical = T_fg(lmn, uvw, beam, .1, spectral)
    compiled = jax.jit(T_fg_weighted)
    weighted = compiled(lmn, uvw, beam, .1, spectral,
                        prior_sqrt=prior_std, noise_whitener=inverse_noise_std)
    np.testing.assert_allclose(weighted, physical * prior_std[None, :] * inverse_noise_std[:, None])
    block = compiled(lmn, uvw[1:], beam[:, 1:], .1, spectral[1:],
                     prior_sqrt=prior_std, noise_whitener=inverse_noise_std[1:])
    np.testing.assert_allclose(block, weighted[1:], atol=1e-13)

    # Weighting is part of the JAX graph, including dynamic prior parameters.
    def total_variance(scale):
        matrix = compiled(lmn, uvw, beam, .1, spectral,
                          prior_sqrt=scale * prior_std, noise_whitener=inverse_noise_std)
        return jnp.real(jnp.vdot(matrix, matrix))

    np.testing.assert_allclose(jax.jit(jax.grad(total_variance))(1.), 2 * total_variance(1.))


@pytest.mark.parametrize("operator, prior, noise, message", [
    (np.ones(3), np.ones(2), np.ones(3), "two-dimensional"),
    (np.ones((3, 2)), np.ones(3), np.ones(3), "coefficient axis"),
    (np.ones((3, 2)), np.ones((2, 1)), np.ones(3), "coefficient axis"),
    (np.ones((3, 2)), np.ones(2), np.ones(2), "visibility axis"),
    (np.ones((3, 2)), np.ones(2), np.ones((3, 1)), "visibility axis"),
])
def test_mismatched_factors_are_rejected_under_jit(operator, prior, noise, message):
    with pytest.raises(ValueError, match=message):
        jax.jit(weight_foreground_operator)(operator, prior, noise)
