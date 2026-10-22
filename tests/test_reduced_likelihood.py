"""Verify basis-independent marginalisation, substituting only the native solver."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from scipy.stats import multivariate_normal

from bayeseor_jax.inference.phi import PhiInverse
from bayeseor_jax.inference.prior import Prior
from bayeseor_jax.model.construction.likelihood_products import (
    build_gram_matrix, build_data_projection, build_data_power,
)


@pytest.fixture
def likelihood_class(monkeypatch):
    def potrs(matrix, rhs, **kwargs):
        return jnp.linalg.solve(matrix, rhs), jnp.linalg.slogdet(matrix)[1]

    monkeypatch.setitem(sys.modules, 'jaxmg', SimpleNamespace(potrs=potrs))
    spec = importlib.util.spec_from_file_location(
        'bayeseor_jax.inference._likelihood_under_test',
        Path(__file__).parents[1] / 'bayeseor_jax/inference/likelihood_marg.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MarginalisedLikelihood


def test_original_and_kl_likelihoods_match_data_space_covariance(likelihood_class):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('x',))
    rng = np.random.default_rng(31)
    foreground = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    cosmological = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
    data = rng.normal(size=5) + 1j * rng.normal(size=5)
    prior_std = np.linspace(.4, 2., 7)
    inverse_noise = np.linspace(.5, 2., 5)
    weighted = np.sqrt(inverse_noise[:, None]) * foreground * prior_std[None, :]
    u, singular_values, _ = np.linalg.svd(weighted, full_matrices=False)
    reduced = u * singular_values / np.sqrt(inverse_noise[:, None])
    models = []

    # Include a zero-retained-mode model, which has only cosmological columns.
    for fg, widths in ((foreground, prior_std), (reduced, np.ones(5)), (reduced[:, :0], np.ones(0))):
        operator = jnp.asarray(np.concatenate([cosmological, fg], axis=1))
        phi = PhiInverse(
            bin_id_per_k_mode=jnp.array([0, 1]), valid_mask=jnp.ones(2, dtype=bool),
            k_vals=jnp.array([1., 2.]), ps_box_size_ra_mpc=1., ps_box_size_dec_mpc=1.,
            ps_box_size_para_mpc=1., inst_to_cosmo_vol=1., n_par=operator.shape[1],
            foreground_prior_std=widths,
        )
        likelihood = likelihood_class(
            t_ninv_t=build_gram_matrix(operator, jnp.asarray(inverse_noise)),
            dbar=build_data_projection(operator, jnp.asarray(inverse_noise), jnp.asarray(data)),
            d_ninv_d=build_data_power(jnp.asarray(inverse_noise), jnp.asarray(data)),
            n_data=5, phi_inv_class=phi,
            prior_class=Prior(bounds_log=jnp.array([[-2., 2.], [-2., 2.]]),
                              power_param_start=0, n_power_params=2),
            n_power_params=2, mesh=mesh,
        )
        values = []
        for rho in (jnp.array([-.3, .2]), jnp.array([.5, -.8])):
            precision = np.asarray(phi.evaluate(10.**rho))
            response = np.asarray(operator)
            covariance = np.diag(1. / inverse_noise) + (response / precision) @ response.conj().T
            # The existing likelihood omits the fixed data/noise normalisation.
            expected = .5 * (
                -np.linalg.slogdet(covariance)[1] - np.log(inverse_noise).sum()
                + np.vdot(data, inverse_noise * data).real
                - np.vdot(data, np.linalg.solve(covariance, data)).real
            )
            value = float(likelihood.log_likelihood(rho))
            np.testing.assert_allclose(value, expected, atol=1e-10)
            values.append(value)
        models.append(values)
    np.testing.assert_allclose(models[0], models[1], atol=1e-10)


@pytest.mark.parametrize("representation", ["original", "kl", "truncated_kl", "no_fg", "fg_only"])
@pytest.mark.parametrize("fit_noise", [False, True])
def test_augmented_likelihood_matches_independent_real_gaussian(
    likelihood_class, representation, fit_noise,
):
    """Check the half factors against a density on independent Re/Im data.

    Foreground amplitudes are real. Cosmological coefficients are conjugate
    Fourier pairs, obtained by a unitary change from real cosine/sine amplitudes.
    Neither conjugate data rows nor Fourier partners add independent samples.
    """
    rng = np.random.default_rng(94)
    n_independent = 5
    visibility = rng.normal(size=5) + 1j * rng.normal(size=5)
    real_data = np.r_[visibility.real, visibility.imag]
    data = np.r_[visibility, visibility.conj()]
    variance = np.linspace(.6, 1.4, n_independent)
    inverse_noise = jnp.asarray(np.tile(1. / variance, 2))
    foreground = rng.normal(size=(5, 12)) + 1j * rng.normal(size=(5, 12))
    widths = np.linspace(.4, 1.3, 12)
    real_fg = np.vstack([foreground.real, foreground.imag])
    augmented_fg = np.vstack([foreground, foreground.conj()])

    if representation in ("kl", "truncated_kl"):
        rank = 10 if representation == "kl" else 3
        weighted = np.sqrt(inverse_noise[:, None]) * augmented_fg * widths
        u, singular_values, _ = np.linalg.svd(weighted, full_matrices=False)
        augmented_fg = u[:, :rank] * singular_values[:rank] / np.sqrt(inverse_noise[:, None])
        # Independently reduce the real-data model; this also checks truncation.
        real_whitening = np.sqrt(2. * np.asarray(inverse_noise))
        u_real, s_real, _ = np.linalg.svd(real_whitening[:, None] * real_fg * widths,
                                         full_matrices=False)
        real_fg = u_real[:, :rank] * s_real[:rank] / real_whitening[:, None]
        widths = np.ones(augmented_fg.shape[1])
    elif representation == "no_fg":
        augmented_fg, real_fg, widths = augmented_fg[:, :0], real_fg[:, :0], widths[:0]

    # Each pair (a_k, a_-k) = ((x+i*y), (x-i*y))/sqrt(2) has equal prior variance.
    fourier_change = np.kron(np.eye(2), np.array([[1., 1j], [1., -1j]]) / np.sqrt(2.))
    cosmological = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    real_21 = np.vstack([cosmological.real, cosmological.imag])
    augmented_21 = np.vstack([cosmological, cosmological.conj()]) @ fourier_change.conj().T
    bins = np.array([0, 0, 1, 1])
    n_power = 2
    if representation == "fg_only":
        augmented_21, real_21, bins = augmented_21[:, :0], real_21[:, :0], bins[:0]
        n_power = 0
    operator = jnp.asarray(np.column_stack([augmented_21, augmented_fg]))
    real_operator = np.column_stack([real_21, real_fg])
    phi = PhiInverse(
        bin_id_per_k_mode=bins, valid_mask=np.ones(bins.size, dtype=bool),
        k_vals=jnp.array([1., 2.])[:n_power], ps_box_size_ra_mpc=1.,
        ps_box_size_dec_mpc=1., ps_box_size_para_mpc=1., inst_to_cosmo_vol=1.,
        n_par=operator.shape[1], foreground_prior_std=widths,
    )
    data_power = build_data_power(inverse_noise, jnp.asarray(data))
    likelihood = likelihood_class(
        t_ninv_t=build_gram_matrix(operator, inverse_noise),
        dbar=build_data_projection(operator, inverse_noise, jnp.asarray(data)),
        d_ninv_d=data_power, n_data=data.size, phi_inv_class=phi,
        prior_class=Prior(bounds_log=jnp.tile(jnp.array([[-2., 2.]]), (n_power, 1)),
                          power_param_start=int(fit_noise), n_power_params=n_power),
        power_param_start=int(fit_noise), n_power_params=n_power,
        intrinsic_noise_fitting=fit_noise,
        mesh=Mesh(np.asarray(jax.devices()[:1]), ('x',)),
    )
    for rho, alpha in ((np.array([-.3, .2])[:n_power], .7),
                       (np.array([.5, -.8])[:n_power], 1.4)):
        if not fit_noise:
            alpha = 1.
        precision = np.asarray(phi.evaluate(jnp.asarray(10.**rho)))
        real_covariance = (np.diag(np.tile(variance * alpha**2 / 2., 2))
                           + (real_operator / precision) @ real_operator.T)
        expected = multivariate_normal.logpdf(real_data, cov=real_covariance)
        theta = np.r_[alpha, rho] if fit_noise else rho
        value = float(likelihood.log_likelihood(theta))
        # Restore only terms deliberately omitted by the sampling target.
        constant = .5 * (np.log(inverse_noise).sum() - data.size * np.log(np.pi))
        if not fit_noise:
            constant -= .5 * float(data_power)
        np.testing.assert_allclose(value + constant, expected, atol=1e-10)
