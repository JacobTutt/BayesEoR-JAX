"""JAX-native prior utilities for BayesEoR-JAX posterior evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import distrax
from functools import partial


class PhiInverseConfig:
    """
    Configuration required to build the diagonal prior precision ``Phi^{-1}``.

    Parameters
    ----------
    bin_id_per_k_mode : jax.Array
        Native spherical-bin assignment from ``k_cube.py`` with shape
        ``(n_eor_voxels,)`` and invalid entries encoded as ``-1``.
    valid_mask : jax.Array
        Native validity mask from ``k_cube.py`` with shape
        ``(n_eor_voxels,)``.
    k_vals : jax.Array
        Mean k value per spherical bin.
    ps_box_size_ra_mpc : float
        Comoving box size along RA-like transverse axis, in Mpc.
    ps_box_size_dec_mpc : float
        Comoving box size along Dec-like transverse axis, in Mpc.
    ps_box_size_para_mpc : float
        Comoving box size along LOS axis, in Mpc.
    inst_to_cosmo_vol : float or jax.Array
        Conversion factor from instrumental voxel units to comoving units.
    n_par : int
        Total number of latent linear coefficients (EoR + FG/LSSM).
    inverse_lw_power : float, optional
        Prior precision assigned to FG/LSSM amplitudes.
    """

    def __init__(
        self,
        bin_id_per_k_mode: jax.Array,
        valid_mask: jax.Array,
        k_vals: jax.Array,
        ps_box_size_ra_mpc: float,
        ps_box_size_dec_mpc: float,
        ps_box_size_para_mpc: float,
        inst_to_cosmo_vol: float | jax.Array,
        n_par: int,
        inverse_lw_power: float = 1e-16,
    ) -> None:
        self.kbin_index_per_voxel = build_kbin_index_per_voxel(
            bin_id_per_k_mode=bin_id_per_k_mode,
            valid_mask=valid_mask,
        )
        self.power_norm_per_bin = compute_power_norm_per_bin(
            k_vals=jnp.asarray(k_vals, dtype=jnp.float64),
            ps_box_size_ra_mpc=ps_box_size_ra_mpc,
            ps_box_size_dec_mpc=ps_box_size_dec_mpc,
            ps_box_size_para_mpc=ps_box_size_para_mpc,
            inst_to_cosmo_vol=jnp.asarray(inst_to_cosmo_vol, dtype=jnp.float64),
        )
        self.n_par = n_par
        self.n_eor_voxels = int(self.kbin_index_per_voxel.shape[0])
        self.inverse_lw_power = inverse_lw_power


class Prior:
    """
    Sampler-space prior configuration.

    Parameters
    ----------
    bounds : jax.Array
        Parameter bounds with shape ``(n_dim, 2)`` in sampler coordinates.
    log_priors : bool
        If True, the power-spectrum parameters are interpreted as log10 values.
    uprior_inds : jax.Array, optional
        Boolean mask over power-spectrum parameters identifying bins that should
        behave as uniform-in-linear-space under log10 sampling.
    power_param_start : int, optional
        Start index of power-spectrum parameters within ``theta``.
    n_power_params : int
        Number of power-spectrum parameters in ``theta``.
    """

    def __init__(
        self,
        bounds: jax.Array,
        log_priors: bool = False,
        uprior_inds: jax.Array | None = None,
        power_param_start: int = 0,
        n_power_params: int = 0,
    ) -> None:
        self.bounds = bounds
        self.log_priors = log_priors
        if uprior_inds is None:
            self.uprior_inds = jnp.zeros((n_power_params,), dtype=bool)
        else:
            self.uprior_inds = uprior_inds
        self.power_param_start = power_param_start
        self.n_power_params = n_power_params

    @partial(jax.jit, static_argnums=0)
    def logprior(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate sampler-space log prior density.

        This uses a box prior in sampler coordinates and applies the BayesEoR
        uniform-in-linear correction on any bins flagged by ``uprior_inds``.
        """
        bounds = jnp.asarray(self.bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        in_support = jnp.all((theta >= lower) & (theta <= upper))
        base = jnp.sum(distrax.Uniform(lower, upper).log_prob(theta))

        theta_power = theta[
            self.power_param_start : self.power_param_start + self.n_power_params
        ]
        theta_power_linear = theta_power_to_linear(theta_power, self.log_priors)
        safe_theta_linear = jnp.maximum(
            theta_power_linear,
            jnp.finfo(theta_power_linear.dtype).tiny,
        )
        jacobian_correction = jnp.sum(
            jnp.where(
                jnp.asarray(self.uprior_inds, dtype=bool),
                jnp.log(safe_theta_linear),
                0.0,
            )
        )

        total = base + jacobian_correction
        return jnp.where(in_support, total, -jnp.inf)

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
        bounds = jnp.asarray(self.bounds)
        low = bounds[:, 0]
        high = bounds[:, 1]
        u = jax.random.uniform(
            rng_key,
            shape=(int(n_samples), int(bounds.shape[0])),
            dtype=low.dtype,
        )
        return low + (high - low) * u


def build_kbin_index_per_voxel(
    bin_id_per_k_mode: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """
    Build an index map from native ``k_cube.py`` assignments.

    Valid entries keep their bin ID and invalid entries are set to ``-1``.
    """
    bid = jnp.asarray(bin_id_per_k_mode, dtype=jnp.int32)
    valid = jnp.asarray(valid_mask, dtype=bool)
    return jnp.where(valid, bid, -1).astype(jnp.int32)


def compute_power_norm_per_bin(
    k_vals: jax.Array,
    ps_box_size_ra_mpc: float,
    ps_box_size_dec_mpc: float,
    ps_box_size_para_mpc: float,
    inst_to_cosmo_vol: jax.Array,
) -> jax.Array:
    """
    Compute old-BayesEoR dimensionless power normalization per k-bin.

    This matches the normalization used in old `posterior.py`:

    ``norm_i = (k_i^3 / (2*pi^2)) / V_cosmo * (inst_to_cosmo_vol)^2``
    """
    cosmo_volume = (
        jnp.asarray(ps_box_size_ra_mpc, dtype=jnp.float64)
        * jnp.asarray(ps_box_size_dec_mpc, dtype=jnp.float64)
        * jnp.asarray(ps_box_size_para_mpc, dtype=jnp.float64)
    )
    return (k_vals**3 / (2.0 * jnp.pi**2)) / cosmo_volume * (inst_to_cosmo_vol**2)


def theta_power_to_linear(theta_power: jax.Array, log_priors: bool) -> jax.Array:
    """Convert sampled power parameters to linear scale when needed."""
    return jax.lax.cond(
        log_priors,
        lambda t: jnp.power(10.0, t),
        lambda t: t,
        theta_power,
    )


def build_phi_inverse(theta_power_linear: jax.Array, cfg: PhiInverseConfig) -> jax.Array:
    """
    Construct diagonal prior precision ``Phi^{-1}`` for latent coefficients.

    The first ``n_eor_voxels`` entries follow the binned EoR power model and
    the remaining entries (FG/LSSM) are assigned ``inverse_lw_power``.
    """
    safe_theta = jnp.maximum(theta_power_linear, jnp.finfo(theta_power_linear.dtype).tiny)
    poweri_per_bin = cfg.power_norm_per_bin / safe_theta

    idx = cfg.kbin_index_per_voxel
    valid = idx >= 0
    idx_clip = jnp.clip(idx, 0, poweri_per_bin.shape[0] - 1)
    phi_eor = jnp.where(valid, poweri_per_bin[idx_clip], 0.0)

    n_fg = cfg.n_par - cfg.n_eor_voxels
    phi_fg = jnp.full((n_fg,), cfg.inverse_lw_power, dtype=phi_eor.dtype)
    return jnp.concatenate([phi_eor, phi_fg], axis=0)

