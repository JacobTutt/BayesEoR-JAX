"""Three-dimensional Fourier geometry of the cosmological 21-cm model."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .dft_funcs import build_nudft_matrix
from .image_geometry import ImageGeometry

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FourierGeometry:
    """Retained ``(u,v,eta)`` modes and their image/frequency bases."""

    uv: jax.Array
    eta: jax.Array
    transverse_basis: jax.Array
    spectral_basis: jax.Array


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
    image_geometry: ImageGeometry,
    n_u: int,
    n_v: int,
    n_eta: int,
    du: float,
    dv: float,
    d_eta: float,
    n_frequency: int,
) -> FourierGeometry:
    """Construct the retained 21-cm modes and separable Fourier bases."""
    if min(n_u, n_v, n_eta, n_frequency) < 1:
        raise ValueError("Fourier-grid dimensions must all be positive")
    if min(du, dv, d_eta) <= 0:
        raise ValueError("Fourier-grid spacings must all be positive")

    uv = _build_uv_modes(n_u, n_v, du, dv)
    eta = _build_eta_modes(n_eta, d_eta)

    lm_eor = image_geometry.lmn_fg[
        image_geometry.time_ind_center,
        :2,
    ]
    lm_eor = lm_eor[:, image_geometry.eor_fg_shared_axis]
    transverse_basis = build_nudft_matrix(
        uv,
        lm_eor,
        sign=-1,
    ) * (du * dv)

    frequency_indices = (
        jnp.arange(n_frequency, dtype=jnp.float64) - n_frequency // 2
    )
    frequency_offsets_hz = frequency_indices / (n_frequency * d_eta)
    spectral_basis = (
        jnp.exp(-2j * jnp.pi * frequency_offsets_hz[:, None] * eta[None, :])
        * d_eta
    ).astype(jnp.complex128)

    return FourierGeometry(
        uv=uv,
        eta=eta,
        transverse_basis=transverse_basis,
        spectral_basis=spectral_basis,
    )
