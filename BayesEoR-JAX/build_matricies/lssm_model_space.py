"""
Large Spectral Scale Model (LSSM) basis construction.

This module defines JAX-compatible utilities for building low-rank, smooth
frequency-space basis vectors used to model foregrounds and large-scale
bandpass structure in 21-cm interferometric analyses.
"""
import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Union
import jax.numpy as jnp

def build_lssm_basis(
    n_freq: int,
    n_modes: int,
    f0_mhz: float,
    df_mhz: float,
    n_powerlaws: int = 0,
    beta: Optional[Union[float, Sequence[float], jnp.ndarray]] = None,
) -> jnp.ndarray:
    """
    Build Large Spectral Scale Model (LSSM) basis vectors for smooth spectral structure.

    The LSSM is a low-rank, smooth-in-frequency model typically used to capture
    foregrounds and/or large-scale bandpass structure.

    This function constructs a basis matrix B with shape (n_freq, n_modes),
    where each column is a spectral basis vector.

    Basis layout
    ------------
    Modes are filled by:
        - first (n_modes - n_powerlaws) modes as polynomials: x^p
        - remaining n_powerlaws modes as power laws: (nu / f0)^(-beta_i)

    Parameters
    ----------
    n_freq
        Number of frequency channels.
    n_modes
        Total number of basis vectors (columns) returned.
    f0_mhz
        Minimum frequency in MHz.
    df_mhz
        Channel width in MHz.
    n_powerlaws
        Number of power-law modes to include (placed after polynomial modes).
    beta
        Spectral indices for the power-law modes. Must have length n_powerlaws.
        If n_powerlaws==1, beta may be a scalar.

    Returns
    -------
    lssm_basis
        Basis matrix of shape (n_freq, n_modes), complex128.
    """
    # Input validation
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_powerlaws < 0:
        raise ValueError("n_powerlaws must be >= 0")
    if n_powerlaws > n_modes:
        raise ValueError(
            f"n_powerlaws={n_powerlaws} exceeds available modes ({n_modes})."
        )

    # Validate beta parameter if power-law modes are requested.
    if n_powerlaws > 0:
        if beta is None:
            raise ValueError("beta must be provided when n_powerlaws > 0")
        # Ensure we are working with float64 to maintain parity with legacy code
        beta_arr = jnp.atleast_1d(jnp.asarray(beta, dtype=jnp.float64))
        if beta_arr.size != n_powerlaws:
            raise ValueError(f"len(beta) must equal n_powerlaws; got {beta_arr.size} vs {n_powerlaws}")
    else:
        beta_arr = jnp.asarray([], dtype=jnp.float64)

    # Frequency axis (MHz) - Explicit float64 for high precision
    nu = f0_mhz + df_mhz * jnp.arange(n_freq, dtype=jnp.float64)

    # Polynomial coordinate (centered index)
    x = jnp.arange(n_freq, dtype=jnp.float64)
    x = x - (n_freq / 2.0)

    # Allocate complex basis (complex128 for 64-bit precision per component)
    lssm_basis = jnp.zeros((n_freq, n_modes), dtype=jnp.complex128)

    # 1. Fill Polynomials first
    n_poly = n_modes - n_powerlaws
    col = 0
    if n_poly > 0:
        powers = jnp.arange(1, n_poly + 1, dtype=jnp.float64)
        poly = x[:, None] ** powers[None, :]
        lssm_basis = lssm_basis.at[:, col:n_poly].set(poly.astype(jnp.complex128))
        col = n_poly

    # 2. Fill Remaining Power Laws 
    if n_powerlaws > 0:
        pl = (nu[:, None] / f0_mhz) ** (-beta_arr[None, :])
        lssm_basis = lssm_basis.at[:, col:].set(pl.astype(jnp.complex128))

    return lssm_basis