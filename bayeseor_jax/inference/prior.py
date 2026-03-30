"""JAX-native prior utilities for BayesEoR-JAX posterior evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import distrax
from functools import partial

class Prior:
    """
    Sampler-space prior configuration.

    Parameters
    ----------
    bounds_log : jax.Array
        Power-spectrum parameter bounds with shape ``(n_power_params, 2)`` in
        log10 space.
    power_param_start : int, optional
        Start index of power-spectrum parameters within ``theta``.
    n_power_params : int
        Number of power-spectrum parameters in ``theta``.
    """

    def __init__(
        self,
        bounds_log: jax.Array,
        power_param_start: int = 0,
        n_power_params: int = 0,
    ) -> None:
        # 1) Validate and standardize the incoming log10 prior bounds.
        #    Expected shape is (n_power_params, 2) -> [low, high] per bin.
        bounds_log = jnp.asarray(bounds_log, dtype=jnp.float64)
        if bounds_log.ndim != 2 or bounds_log.shape[1] != 2:
            raise ValueError("bounds_log must have shape (n_power_params, 2)")
        if n_power_params <= 0:
            n_power_params = int(bounds_log.shape[0])
        if int(bounds_log.shape[0]) != int(n_power_params):
            raise ValueError(
                "bounds_log first dimension must match n_power_params: "
                f"{bounds_log.shape[0]} != {n_power_params}"
            )

        # The sampler always works in log10(power) coordinates. Any optional
        # Jacobian correction for linear-uniform physical priors is handled in
        # the likelihood classes, not here.
        self.bounds_log = bounds_log
        self.lower = bounds_log[:, 0]
        self.upper = bounds_log[:, 1]
        # Cache one distrax prior object for repeated log_prob/sample calls.
        self.uniform_prior = distrax.Uniform(self.lower, self.upper)
        self.power_param_start = power_param_start
        self.n_power_params = n_power_params

    @partial(jax.jit, static_argnums=0)
    def logprior(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate sampler-space log prior density.
        """
        log_prior = jnp.sum(self.uniform_prior.log_prob(theta))
        return log_prior

    def sample_from_priors(self, rng_key: jax.Array, n_samples: int) -> jax.Array:
        """
        Draw independent samples from the sampler-space box priors.

        Parameters
        ----------
        rng_key : jax.Array
            JAX PRNG key.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        jax.Array
            Array with shape ``(n_samples, n_dim)``.
        """
        return self.uniform_prior.sample(
            seed=rng_key,
            sample_shape=(int(n_samples),),
        )


def theta_power_to_linear(
    theta_power: jax.Array,
) -> jax.Array:
    """
    Convert sampled power parameters to linear scale.

    The sampler always works in log10(power) coordinates, so all sampled
    power parameters are exponentiated here.
    """
    return jnp.power(10.0, theta_power)


def linear_uniform_log_jacobian(
    theta_power: jax.Array,
    uniform_priors: jax.Array,
) -> jax.Array:
    """
    Jacobian correction for bins with linear-uniform physical priors.

    The sampler coordinates are always ``log10(power)``. For bins whose
    intended physical prior is uniform in linear power, the old BayesEoR
    target adds ``sum(log(power_linear))`` to the log density, which is
    equivalent to ``log(10) * sum(theta_power)`` over those bins.
    """
    return jnp.log(10.0) * jnp.sum(jnp.where(uniform_priors, theta_power, 0.0))
