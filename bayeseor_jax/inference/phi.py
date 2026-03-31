"""Latent prior-precision utilities for BayesEoR-JAX."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class PhiInverse:
    """
    Prior-precision configuration for the latent linear coefficients.

    In the hierarchical model, the latent coefficient vector ``a`` is given a
    zero-mean Gaussian prior, ``a ~ N(0, Phi)``, where ``Phi`` is diagonal in
    this implementation. The object stores the static ingredients needed to
    construct ``Phi^{-1}`` (the prior precision) from sampled power-spectrum
    amplitudes during each likelihood evaluation.

    Statistically, ``Phi^{-1}`` controls shrinkage/regularization of latent
    modes:

    - larger precision => stronger pull of coefficients toward zero
    - smaller precision => weaker prior constraint (broader prior)

    The resulting precision enters the marginalized Gaussian system as
    ``Sigma = T^H N^{-1} T + Phi^{-1}``, so it directly affects MAP estimates,
    posterior curvature, and evidence values.

    To keep repeated log-likelihood calls cheap, this class precomputes and
    stores geometry/cosmology-dependent terms (voxel->bin mapping and per-bin
    normalization factors) that do not change with ``theta``.

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

    def evaluate(self, theta_power_linear: jax.Array) -> jax.Array:
        """
        Build diagonal prior precision ``Phi^{-1}`` for current power amplitudes.

        Parameters
        ----------
        theta_power_linear : jax.Array
            Linear-space power-spectrum amplitudes for each non-empty k-bin.

        Returns
        -------
        jax.Array
            Diagonal precision vector with EoR entries followed by FG/LSSM
            entries.
        """
        safe_theta = jnp.maximum(
            theta_power_linear, jnp.finfo(theta_power_linear.dtype).tiny
        )
        poweri_per_bin = self.power_norm_per_bin / safe_theta

        idx = self.kbin_index_per_voxel
        valid = idx >= 0
        idx_clip = jnp.clip(idx, 0, poweri_per_bin.shape[0] - 1)
        phi_eor = jnp.where(valid, poweri_per_bin[idx_clip], 0.0)

        n_fg = self.n_par - self.n_eor_voxels
        phi_fg = jnp.full((n_fg,), self.inverse_lw_power, dtype=phi_eor.dtype)
        return jnp.concatenate([phi_eor, phi_fg], axis=0)


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
    Compute dimensionless power normalization per k-bin.

    ``norm_i = (k_i^3 / (2*pi^2)) / V_cosmo * (inst_to_cosmo_vol)^2``
    """
    cosmo_volume = (
        jnp.asarray(ps_box_size_ra_mpc, dtype=jnp.float64)
        * jnp.asarray(ps_box_size_dec_mpc, dtype=jnp.float64)
        * jnp.asarray(ps_box_size_para_mpc, dtype=jnp.float64)
    )
    return (k_vals**3 / (2.0 * jnp.pi**2)) / cosmo_volume * (inst_to_cosmo_vol**2)
