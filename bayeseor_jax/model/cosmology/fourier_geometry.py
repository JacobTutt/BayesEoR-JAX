"""Define the (u, v, eta) coordinates of cosmological brightness coefficients.

These model coordinates supply both the Fourier-to-image operators and the
conversion to cosmological k bins; they are distinct from measured baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FourierGeometry:
    r"""Retained Fourier coordinates and their integration spacings.

    Parameters
    ----------
    uv : jax.Array
        Retained angular Fourier coordinates, shape (2, n_uv_mode), excluding
        the transverse monopole.
    eta : jax.Array
        Retained line-of-sight Fourier coordinates in inverse Hz, shape
        (n_eta_mode,), excluding eta=0.
    du : float
        Transverse u spacing in inverse radians; must be positive.
    dv : float
        Transverse v spacing in inverse radians; must be positive.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).

    Notes
    -----
    ``uv`` has shape ``(2, n_uv_mode)`` and contains angular spatial
    frequencies; ``eta`` has shape ``(n_eta_mode,)`` in inverse Hz.
    ``du``, ``dv``, and ``d_eta`` supply the corresponding integration weights.
    A coefficient belongs to each (uv, eta) pair, with eta varying fastest
    when coefficients are flattened into a vector.
    """

    uv: jax.Array
    eta: jax.Array
    du: float
    dv: float
    d_eta: float


def _build_uv_modes(
    n_u: int,
    n_v: int,
    du: float,
    dv: float,
) -> jax.Array:
    r"""Place transverse brightness modes on a centred rectangular uv grid.

    Parameters
    ----------
    n_u : int
        Number of transverse u modes before removing the angular monopole.
    n_v : int
        Number of transverse v modes before removing the angular monopole.
    du : float
        Transverse u spacing in inverse radians; must be positive.
    dv : float
        Transverse v spacing in inverse radians; must be positive.

    Returns
    -------
    uv : jax.Array
        Centred coordinates, shape (2, n_u*n_v - 1), ordered with u varying
        fastest.

    Notes
    -----
    Axis coordinates are ``(arange(n) - n//2) * spacing``. Return shape
    ``(2, n_u*n_v - 1)`` with u varying fastest within each v row and remove
    the angular monopole ``(u,v)=(0,0)``.
    """
    # Centre both angular Fourier axes on the zero mode.
    u = (jnp.arange(n_u, dtype=jnp.float64) - n_u // 2) * du
    v = (jnp.arange(n_v, dtype=jnp.float64) - n_v // 2) * dv
    # Row-major flattening makes u the fast axis within each v row.
    u_grid, v_grid = jnp.meshgrid(u, v, indexing="xy")
    uv = jnp.stack([u_grid.reshape(-1), v_grid.reshape(-1)])
    # Remove only the transverse monopole, retaining the other uv pairs.
    monopole = (n_v // 2) * n_u + (n_u // 2)
    return jnp.concatenate(
        [uv[:, :monopole], uv[:, monopole + 1 :]],
        axis=1,
    )


def _build_eta_modes(n_eta: int, d_eta: float) -> jax.Array:
    r"""Place line-of-sight brightness modes on a centred eta axis.

    Parameters
    ----------
    n_eta : int
        Number of line-of-sight Fourier modes before excluding eta=0.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).

    Returns
    -------
    eta : jax.Array
        Centred nonzero eta modes in inverse Hz, shape (n_eta - 1,).

    Notes
    -----
    Return ``(arange(n_eta) - n_eta//2) * d_eta`` in inverse Hz with the
    frequency-independent ``eta=0`` mode removed, giving ``n_eta - 1`` modes.
    """
    # Use the same centred indexing as the transverse grid.
    eta = (jnp.arange(n_eta, dtype=jnp.float64) - n_eta // 2) * d_eta
    # Exclude the frequency-independent mode from the cosmological component.
    eta0 = n_eta // 2
    return jnp.concatenate([eta[:eta0], eta[eta0 + 1 :]])


def build_fourier_geometry(
    *,
    n_u: int,
    n_v: int,
    n_eta: int,
    du: float,
    dv: float,
    d_eta: float,
) -> FourierGeometry:
    r"""Define the Fourier grid on which cosmological coefficients are placed.

    Parameters
    ----------
    n_u : int
        Number of transverse u modes before removing the angular monopole.
    n_v : int
        Number of transverse v modes before removing the angular monopole.
    n_eta : int
        Number of line-of-sight Fourier modes before excluding eta=0.
    du : float
        Transverse u spacing in inverse radians; must be positive.
    dv : float
        Transverse v spacing in inverse radians; must be positive.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).

    Returns
    -------
    geometry : FourierGeometry
        Retained coordinates and spacings for (n_u*n_v - 1)*(n_eta - 1)
        coefficients.

    Notes
    -----
    ``n_u``, ``n_v``, and ``n_eta`` count modes before zero-mode exclusions;
    ``du`` and ``dv`` are transverse spacings and ``d_eta`` is in inverse Hz.
    Return a ``FourierGeometry`` representing
    ``(n_u*n_v - 1) * (n_eta - 1)`` coefficients. The transverse monopole
    and eta zero mode are excluded consistently for operators and k binning.
    No image coordinates, beam response, or transform matrices are built here.
    """
    # Reject invalid dimensions and integration weights before building axes.
    if min(n_u, n_v, n_eta) < 1:
        raise ValueError("Fourier-grid dimensions must all be positive")
    if min(du, dv, d_eta) <= 0:
        raise ValueError("Fourier-grid spacings must all be positive")

    # Share these retained coordinates between operator construction and k binning.
    return FourierGeometry(
        uv=_build_uv_modes(n_u, n_v, du, dv),
        eta=_build_eta_modes(n_eta, d_eta),
        du=float(du),
        dv=float(dv),
        d_eta=float(d_eta),
    )
