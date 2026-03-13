"""JAX-native marginalized Gaussian likelihood for BayesEoR-JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .prior import PhiInverse, Prior, build_phi_inverse, theta_power_to_linear


class Likelihood:
    """
    Configuration for marginalized log-likelihood evaluation.

    Parameters
    ----------
    t_ninv_t : jax.Array
        Matrix ``T^H N^{-1} T``.
    dbar : jax.Array
        Vector ``T^H N^{-1} d``.
    d_ninv_d : float
        Scalar ``d^H N^{-1} d``.
    n_data : int
        Number of data samples contributing to the noise normalization term.
    phi_cfg : PhiInverse
        Prior-precision construction settings.
    prior_cfg : Prior
        Sampler-space prior configuration used to evaluate the full log density.
    uniform_priors : jax.Array, optional
        Boolean mask over sampled power parameters where True entries are
        sampled in linear space and False entries are sampled in log10 space.
    intrinsic_noise_fitting : bool, optional
        Whether the first sampled parameter is intrinsic noise amplitude
        ``alpha_prime``.
    power_param_start : int, optional
        Start index of power-spectrum parameters within ``theta``.
    n_power_params : int
        Number of power-spectrum parameters within ``theta``.
    """

    def __init__(
        self,
        t_ninv_t: jax.Array,
        dbar: jax.Array,
        d_ninv_d: float,
        n_data: int,
        phi_cfg: PhiInverse,
        prior_cfg: Prior,
        uniform_priors: jax.Array | None = None,
        intrinsic_noise_fitting: bool = False,
        power_param_start: int = 0,
        n_power_params: int = 0,
    ) -> None:
        # Ensure dense JAX arrays for the core likelihood linear algebra path.
        if hasattr(t_ninv_t, "todense"):
            t_ninv_t = t_ninv_t.todense()
        if hasattr(dbar, "todense"):
            dbar = dbar.todense()

        self.t_ninv_t = jnp.asarray(t_ninv_t)
        self.dbar = jnp.asarray(dbar)
        self.d_ninv_d = jnp.asarray(d_ninv_d)
        self.n_data = n_data
        self.phi_cfg = phi_cfg
        self.prior_cfg = prior_cfg
        if uniform_priors is None:
            self.uniform_priors = jnp.zeros((n_power_params,), dtype=bool)
        else:
            mask = jnp.asarray(uniform_priors, dtype=bool)
            if mask.shape != (n_power_params,):
                raise ValueError(
                    "uniform_priors must have shape "
                    f"({n_power_params},), got {mask.shape}"
                )
            self.uniform_priors = mask
        self.intrinsic_noise_fitting = intrinsic_noise_fitting
        self.power_param_start = power_param_start
        self.n_power_params = n_power_params

    def log_likelihood(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate marginalized Gaussian log-likelihood.

        This integrates out the latent linear coefficient vector analytically and
        returns the old BayesEoR-style objective:

        ``-0.5*log|Sigma| - 0.5*log|Phi^{-1}| + 0.5*dbar^H*Sigma^{-1}*dbar``

        plus optional intrinsic-noise terms.
        """
        # Extract sampled power-spectrum parameters from the full parameter vector.
        theta_power = theta[
            self.power_param_start : self.power_param_start + self.n_power_params
        ]

        # Optional intrinsic-noise amplitude parameter alpha' (default 1 if disabled).
        alpha = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda t: t[0],
            lambda t: jnp.array(1.0, dtype=t.dtype),
            theta,
        )
        alpha2 = alpha * alpha
        inv_alpha2 = 1.0 / alpha2

        # Convert sampled power parameters into linear-space amplitudes.
        power_linear = theta_power_to_linear(theta_power, self.uniform_priors)
        # Build diagonal prior precision Phi^{-1} for latent linear coefficients.
        phi_i = build_phi_inverse(power_linear, self.phi_cfg)

        # Apply intrinsic-noise scaling to compressed normal-equation terms.
        t_ninv_t_scaled = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda m: m * inv_alpha2,
            lambda m: m,
            self.t_ninv_t,
        )
        dbar_scaled = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda v: v * inv_alpha2,
            lambda v: v,
            self.dbar,
        )

        # Sigma = T^H N^{-1} T + Phi^{-1}.
        sigma = t_ninv_t_scaled + jnp.diag(phi_i)
        # Cholesky factorization for stable solves and log-determinant evaluation.
        chol = jnp.linalg.cholesky(sigma)

        # Solve Sigma^{-1} dbar via two triangular solves.
        y = jsp.linalg.solve_triangular(chol, dbar_scaled, lower=True)
        sigma_inv_dbar = jsp.linalg.solve_triangular(chol.conj().T, y, lower=False)

        # log|Sigma| from Cholesky diagonal (|Sigma| = prod(diag(L))^2).
        log_sigma_det = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))
        # log|Phi| computed as -log|Phi^{-1}| since Phi^{-1} is diagonal.
        log_phi_det = -jnp.sum(jnp.log(phi_i))
        # Quadratic compressed-data term dbar^H Sigma^{-1} dbar.
        quadratic = jnp.real(jnp.vdot(dbar_scaled, sigma_inv_dbar))

        # Core marginalized Gaussian objective.
        base = -0.5 * log_sigma_det - 0.5 * log_phi_det + 0.5 * quadratic

        # Optional intrinsic-noise normalization terms.
        noise_term = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda _: -0.5
            * (self.d_ninv_d * inv_alpha2 + self.n_data * jnp.log(alpha2)),
            lambda _: 0.0,
            operand=None,
        )
        return base + noise_term

    def log_posterior(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate full BayesEoR-style log density for sampling.

        This folds the sampler-space prior into the marginalized Gaussian
        objective so the returned value matches the old single-call target.
        """
        return self.log_likelihood(theta) + self.prior_cfg.logprior(theta)
