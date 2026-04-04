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
    fg_model_domain : {"uv", "image"}, optional
        Spatial basis used for the non-monopole foreground block. When set to
        ``"image"``, the leading non-monopole image-domain FG slice uses the
        pushed-forward precision implied by the current UV prior and FG
        ``UV -> LM`` embedding.
    n_nonmp_fg : int, optional
        Number of leading FG coefficients belonging to the non-monopole
        foreground block. These coefficients immediately follow the EoR block
        in the latent vector. Only used when ``fg_model_domain="image"``.
    n_nonmp_uv : int, optional
        Number of non-monopole FG UV modes used in the pushed-forward image
        prior calculation.
    du_fg, dv_fg : float, optional
        FG UV pixel spacings used in the raw quadrature ``UV -> LM`` embedding.
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
        fg_model_domain: str = "uv",
        n_nonmp_fg: int = 0,
        n_nonmp_uv: int = 0,
        du_fg: float | None = None,
        dv_fg: float | None = None,
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
        self.fg_model_domain = str(fg_model_domain)
        self.n_nonmp_fg = int(n_nonmp_fg)
        self.n_nonmp_uv = int(n_nonmp_uv)
        self.du_fg = None if du_fg is None else float(du_fg)
        self.dv_fg = None if dv_fg is None else float(dv_fg)
        self.fg_nonmp_image_inverse_prior = None
        if self.fg_model_domain == "image":
            if self.n_nonmp_uv <= 0:
                raise ValueError("image-domain FG prior requires n_nonmp_uv > 0")
            if self.du_fg is None or self.dv_fg is None:
                raise ValueError("image-domain FG prior requires du_fg and dv_fg")
            self.fg_nonmp_image_inverse_prior = float(self.inverse_lw_power) / (
                float(self.n_nonmp_uv) * (self.du_fg * self.dv_fg) ** 2
            )
        self.use_fg_nonmp_image_override = (
            self.fg_model_domain == "image"
            and self.n_nonmp_fg > 0
        )
        self.n_nonmp_fg_override = min(
            self.n_nonmp_fg,
            self.n_par - self.n_eor_voxels,
        )

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
        if self.use_fg_nonmp_image_override:
            phi_fg = phi_fg.at[: self.n_nonmp_fg_override].set(
                self.fg_nonmp_image_inverse_prior
            )
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
