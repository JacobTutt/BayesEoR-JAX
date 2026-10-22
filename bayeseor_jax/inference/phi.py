"""Latent prior-precision utilities for BayesEoR-JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class PhiInverse:
    """Construct the diagonal prior precision for the joint coefficients.

    The cosmological block is set by the sampled spherical power-spectrum
    amplitudes. The foreground block accepts one standard deviation per operator
    column, independent of its representation: physical widths for the original
    coefficients or unit widths for prior-standardised KL coefficients.

    Parameters
    ----------
    bin_id_per_k_mode, valid_mask : array-like
        Spherical bin assignments and validity flags for cosmological columns.
    k_vals : array-like
        Representative physical wavenumbers of the sampled power-spectrum bins.
    ps_box_size_ra_mpc, ps_box_size_dec_mpc, ps_box_size_para_mpc : float
        Comoving dimensions used in the Fourier power normalisation.
    inst_to_cosmo_vol : float
        Instrumental-to-cosmological volume conversion.
    n_par : int
        Total number of cosmological and foreground coefficient columns.
    foreground_prior_std : array-like
        Finite, positive real standard deviations, one per foreground column
        in operator order. Pass ones for a reduced KL operator; its physical
        prior has already been absorbed. An empty vector supports zero retained
        foreground modes.
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
        foreground_prior_std: jax.Array,
    ) -> None:
        self.kbin_index_per_voxel = build_kbin_index_per_voxel(
            bin_id_per_k_mode=bin_id_per_k_mode,
            valid_mask=valid_mask,
        )
        k_vals = jnp.asarray(k_vals, dtype=jnp.float64)
        self.power_norm_per_bin = (
            compute_power_norm_per_bin(
                k_vals=k_vals,
                ps_box_size_ra_mpc=ps_box_size_ra_mpc,
                ps_box_size_dec_mpc=ps_box_size_dec_mpc,
                ps_box_size_para_mpc=ps_box_size_para_mpc,
                inst_to_cosmo_vol=jnp.asarray(
                    inst_to_cosmo_vol,
                    dtype=jnp.float64,
                ),
            )
            if k_vals.size
            else jnp.empty((0,), dtype=jnp.float64)
        )
        self.n_par = int(n_par)
        self.n_eor_voxels = int(self.kbin_index_per_voxel.shape[0])
        foreground_std = np.asarray(foreground_prior_std)
        n_foreground = self.n_par - self.n_eor_voxels
        if foreground_std.shape != (n_foreground,):
            raise ValueError(
                "foreground prior length mismatch: the joint model contains "
                f"{n_foreground} foreground coefficients; supply one width per column"
            )
        if (not np.isrealobj(foreground_std) or not np.all(np.isfinite(foreground_std))
                or np.any(foreground_std <= 0)):
            raise ValueError("foreground prior widths must be finite, real and positive")
        self.foreground_precision = 1.0 / jnp.square(jnp.asarray(foreground_std, dtype=jnp.float64))

    def evaluate(self, theta_power_linear: jax.Array) -> jax.Array:
        """Return the joint EoR and foreground prior-precision diagonal."""
        if self.n_eor_voxels:
            safe_theta = jnp.maximum(
                theta_power_linear,
                jnp.finfo(theta_power_linear.dtype).tiny,
            )
            poweri_per_bin = self.power_norm_per_bin / safe_theta
            indices = self.kbin_index_per_voxel
            valid = indices >= 0
            clipped = jnp.clip(indices, 0, poweri_per_bin.shape[0] - 1)
            phi_eor = jnp.where(valid, poweri_per_bin[clipped], 0.0)
        else:
            phi_eor = jnp.empty((0,), dtype=theta_power_linear.dtype)
        return jnp.concatenate(
            [
                phi_eor,
                self.foreground_precision.astype(theta_power_linear.dtype),
            ]
        )



def build_kbin_index_per_voxel(
    bin_id_per_k_mode: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """
    Build an index map from ``model/cosmology/power_spectrum.py`` assignments.

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
