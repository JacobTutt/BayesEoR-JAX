"""Three-dimensional Fourier geometry of the cosmological 21-cm model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FourierGeometry:
    """Retained coordinates of the cosmological Fourier model."""

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
    """Return the flattened transverse grid without its monopole."""
    u = (jnp.arange(n_u, dtype=jnp.float64) - n_u // 2) * du
    v = (jnp.arange(n_v, dtype=jnp.float64) - n_v // 2) * dv
    u_grid, v_grid = jnp.meshgrid(u, v, indexing="xy")
    uv = jnp.stack([u_grid.reshape(-1), v_grid.reshape(-1)])
    monopole = (n_v // 2) * n_u + (n_u // 2)
    return jnp.concatenate(
        [uv[:, :monopole], uv[:, monopole + 1 :]],
        axis=1,
    )


def _build_eta_modes(n_eta: int, d_eta: float) -> jax.Array:
    """Return the physical line-of-sight Fourier modes without ``eta=0``."""
    eta = (jnp.arange(n_eta, dtype=jnp.float64) - n_eta // 2) * d_eta
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
    """Construct the retained ``(u, v, eta)`` coordinates."""
    if min(n_u, n_v, n_eta) < 1:
        raise ValueError("Fourier-grid dimensions must all be positive")
    if min(du, dv, d_eta) <= 0:
        raise ValueError("Fourier-grid spacings must all be positive")

    return FourierGeometry(
        uv=_build_uv_modes(n_u, n_v, du, dv),
        eta=_build_eta_modes(n_eta, d_eta),
        du=float(du),
        dv=float(dv),
        d_eta=float(d_eta),
    )
