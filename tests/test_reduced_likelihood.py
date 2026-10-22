"""Verify basis-independent marginalisation, substituting only the native solver."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from bayeseor_jax.inference.phi import PhiInverse
from bayeseor_jax.inference.prior import Prior
from bayeseor_jax.model.construction.likelihood_products import (
    build_gram_matrix, build_data_projection, build_data_power,
)


def test_original_and_kl_likelihoods_match_data_space_covariance(monkeypatch):
    def potrs(matrix, rhs, **kwargs):
        return jnp.linalg.solve(matrix, rhs), jnp.linalg.slogdet(matrix)[1]

    monkeypatch.setitem(sys.modules, 'jaxmg', SimpleNamespace(potrs=potrs))
    spec = importlib.util.spec_from_file_location(
        'bayeseor_jax.inference._likelihood_under_test',
        Path(__file__).parents[1] / 'bayeseor_jax/inference/likelihood_marg.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
        likelihood = module.MarginalisedLikelihood(
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
