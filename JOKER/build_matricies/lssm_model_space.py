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
        - first n_powerlaws modes as power laws: (nu / f0)^(-beta_i)
        - remaining modes as polynomials in a centered index coordinate: x^p

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
        Number of power-law modes to include (replacing the first polynomial modes).
        These modes are placed immediately after the constant (if present).
    beta
        Spectral indices for the power-law modes. Must have length n_powerlaws.
        If n_powerlaws==1, beta may be a scalar.
    Returns
    -------
    lssm_basis
        Basis matrix of shape (n_freq, n_modes), complex.
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
        beta_arr = jnp.atleast_1d(jnp.asarray(beta))
        if beta_arr.size != n_powerlaws:
            raise ValueError(f"len(beta) must equal n_powerlaws; got {beta_arr.size} vs {n_powerlaws}")
    else:
        beta_arr = jnp.asarray([], dtype=jnp.float64)

    # Frequency axis (MHz) for polynomial and power-law evaluation.
    nu = f0_mhz + df_mhz * jnp.arange(n_freq, dtype=jnp.float64)

    # Polynomial coordinate
    x = jnp.arange(n_freq, dtype=jnp.float64)
    x = x - (n_freq / 2.0)

    # Allocate complex basis.
    lssm_basis = jnp.zeros((n_freq, n_modes), dtype=jnp.complex128)

    col = 0
    if n_powerlaws > 0:
        pl = (nu[:, None] / f0_mhz) ** (-beta_arr[None, :])
        lssm_basis = lssm_basis.at[:, col : col + n_powerlaws].set(pl.astype(jnp.complex128))
        col += n_powerlaws

    n_poly = n_modes - col
    if n_poly > 0:
        powers = jnp.arange(1, n_poly + 1, dtype=jnp.float64)
        poly = x[:, None] ** powers[None, :]
        lssm_basis = lssm_basis.at[:, col:].set(poly.astype(jnp.complex128))

    return lssm_basis
