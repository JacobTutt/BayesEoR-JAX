"""JAX-native full-space Gaussian likelihood for BayesEoR-JAX."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .phi import PhiInverse
from .prior import Prior, linear_uniform_log_jacobian, theta_power_to_linear


class FullLikelihood:
    """
    Full joint log density over power-spectrum and latent linear coefficients.

    This likelihood keeps the latent linear coefficients in the sampled state
    rather than analytically marginalizing them out. It therefore evaluates the
    joint density

    ``Pr(theta_power, a, q | d) ∝ Pr(d | a, q) Pr(a, q | theta_power) Pr(theta_power)``

    using the current JAX pipeline objects. In the present pipeline the forward
    operator ``T`` already maps from the full latent linear state into the data
    space, so the latent coefficients are represented by one vector whose
    leading block corresponds to EoR coefficients ``a`` and whose trailing
    block corresponds to the non-EoR ``q``-like coefficients already present in
    the model basis.

    Parameters
    ----------
    t_matrix : jax.Array
        Forward operator ``T`` mapping latent coefficients into the data space.
    vis_noisy : jax.Array
        Complex visibility vector.
    ninv : jax.Array
        Noise precision, either diagonal weights with shape ``(n_data,)`` or a
        dense precision matrix with shape ``(n_data, n_data)``.
    n_data : int
        Number of data samples contributing to the noise normalization term.
    phi_inv_class : PhiInverse
        Prior-precision construction settings.
    prior_class : Prior
        Sampler-space prior configuration for the power-spectrum parameters.
    uniform_priors : jax.Array, optional
        Boolean mask over power-spectrum parameters whose intended physical
        prior is uniform in linear power. The sampler still works in
        ``log10(power)`` coordinates; True entries receive the Jacobian
        correction in the likelihood.
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
        t_matrix: jax.Array,
        vis_noisy: jax.Array,
        ninv: jax.Array,
        n_data: int,
        phi_inv_class: PhiInverse,
        prior_class: Prior,
        uniform_priors: jax.Array | None = None,
        intrinsic_noise_fitting: bool = False,
        power_param_start: int = 0,
        n_power_params: int = 0,
    ) -> None:
        self.t_matrix = jnp.asarray(t_matrix)
        self.vis_noisy = jnp.asarray(vis_noisy, dtype=jnp.complex128)
        self.ninv = jnp.asarray(ninv)
        if self.ninv.ndim != 1:
            raise ValueError(
                "FullLikelihood currently expects diagonal Ninv stored as a 1D "
                "weight vector."
            )
        self.n_data = int(n_data)
        self.phi_inv_class = phi_inv_class
        self.prior_class = prior_class
        self.intrinsic_noise_fitting = bool(intrinsic_noise_fitting)
        self.power_param_start = int(power_param_start)
        self.n_power_params = int(n_power_params)
        self.n_linear_params = int(self.t_matrix.shape[1])
        self.n_eor_linear_params = int(self.phi_inv_class.n_eor_voxels)
        self.n_q_linear_params = int(self.n_linear_params - self.n_eor_linear_params)

        if self.n_linear_params != int(self.phi_inv_class.n_par):
            raise ValueError(
                "t_matrix column count must match phi_inv_class.n_par: "
                f"{self.n_linear_params} != {self.phi_inv_class.n_par}"
            )

        if uniform_priors is None:
            self.uniform_priors = jnp.zeros((self.n_power_params,), dtype=bool)
        else:
            mask = jnp.asarray(uniform_priors, dtype=bool)
            if mask.shape != (self.n_power_params,):
                raise ValueError(
                    "uniform_priors must have shape "
                    f"({self.n_power_params},), got {mask.shape}"
                )
            self.uniform_priors = mask

        self.linear_param_start = self.power_param_start + self.n_power_params
        self.theta_dim = self.linear_param_start + self.n_linear_params
        self._log_probability_impl = jax.jit(self.log_probability_terms)
        self._value_and_grad_impl = jax.jit(
            jax.value_and_grad(self.log_probability_terms, argnums=3)
        )

    def split_theta(self, theta: jax.Array) -> dict[str, jax.Array]:
        """
        Split the sampler vector into power, EoR, and non-EoR coefficient blocks.

        The expected sampler layout is:

        1. optional intrinsic-noise amplitude ``alpha_prime``
        2. ``n_power_params`` power-spectrum parameters
        3. ``n_linear_params`` latent coefficients
        """
        theta = jnp.asarray(theta)
        if theta.shape != (self.theta_dim,):
            raise ValueError(
                f"theta must have shape ({self.theta_dim},), got {theta.shape}"
            )

        alpha = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda t: t[0],
            lambda t: jnp.array(1.0, dtype=t.dtype),
            theta,
        )
        theta_power = theta[
            self.power_param_start : self.power_param_start + self.n_power_params
        ]
        linear_coeffs = theta[self.linear_param_start :]

        return {
            "alpha": alpha,
            "theta_power": theta_power,
            "a": linear_coeffs[: self.n_eor_linear_params],
            "q": linear_coeffs[self.n_eor_linear_params :],
            "linear_coeffs": linear_coeffs,
        }

    def _log_power_sampler_prior(self, theta_power: jax.Array) -> jax.Array:
        """
        Evaluate the box prior on sampled power-spectrum parameters locally.

        The `Prior` object is still retained for nested-sampling workflows and
        prior draws, but the full-state HMC likelihood can evaluate the same
        uniform sampler-space prior directly from the cached bounds without
        going through the distrax object.
        """
        theta_power = jnp.asarray(theta_power, dtype=self.prior_class.lower.dtype)
        lower = self.prior_class.lower
        upper = self.prior_class.upper
        widths = upper - lower
        inside = jnp.all((theta_power >= lower) & (theta_power <= upper))
        log_norm = -jnp.sum(jnp.log(widths))
        return jnp.where(inside, log_norm, -jnp.inf)

    def log_probability(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate the full sampled-state log posterior density.

        This keeps the linear coefficients explicit and therefore replaces the
        marginalized ``Sigma`` solve with a direct forward-model residual term,
        the diagonal latent Gaussian prior encoded by ``PhiInverse``, and the
        sampler-space box prior on the sampled power-spectrum parameters.
        """
        return self._log_probability_impl(
            self.t_matrix,
            self.vis_noisy,
            self.ninv,
            theta,
        )

    def value_and_grad(self, theta: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Evaluate the full sampled-state log posterior and its gradient.

        This shares the same explicit-argument compiled kernel as
        ``log_probability`` so the large saved arrays are treated as runtime
        inputs rather than captured constants.
        """
        return self._value_and_grad_impl(
            self.t_matrix,
            self.vis_noisy,
            self.ninv,
            theta,
        )

    def log_probability_terms(
        self,
        t_matrix: jax.Array,
        vis_noisy: jax.Array,
        ninv: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """
        Evaluate the full sampled-state log posterior density for explicit data terms.

        This variant keeps the large arrays as normal function arguments so JAX
        can trace them as runtime inputs instead of embedding them as closed-over
        constants inside a compiled bound method.
        """
        split = self.split_theta(theta)
        alpha = split["alpha"]
        theta_power = split["theta_power"]
        linear_coeffs = split["linear_coeffs"]

        alpha2 = alpha * alpha
        inv_alpha2 = 1.0 / alpha2

        power_linear = theta_power_to_linear(theta_power)
        phi_i = self.phi_inv_class.evaluate(power_linear)
        jacobian_term = linear_uniform_log_jacobian(
            theta_power,
            self.uniform_priors,
        )

        t_matrix = jnp.asarray(t_matrix)
        vis_noisy = jnp.asarray(vis_noisy, dtype=jnp.complex128)
        ninv = jnp.asarray(ninv)

        model_vis = t_matrix @ linear_coeffs
        residual = vis_noisy - model_vis
        weighted_residual = ninv * residual
        data_quadratic = jnp.real(jnp.vdot(residual, weighted_residual))
        prior_quadratic = jnp.sum(phi_i * jnp.square(linear_coeffs))

        latent_prior_norm = 0.5 * jnp.sum(jnp.log(phi_i))
        noise_term = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda _: -0.5
            * (data_quadratic * inv_alpha2 + self.n_data * jnp.log(alpha2)),
            lambda _: -0.5 * data_quadratic,
            operand=None,
        )
        latent_term = latent_prior_norm - 0.5 * prior_quadratic
        power_prior = self._log_power_sampler_prior(theta_power)
        return noise_term + latent_term + power_prior + jacobian_term
