"""Rectilinear transverse Fourier grid for the cosmological model."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


class UV_ModelSpace:
    """
    Build the rectilinear ``(u, v)`` grid used by the 21-cm model.

    Attributes
    ----------
    uv_eor : jnp.ndarray
        Flattened EoR (u, v) coordinates with shape (2, N_uv_eor).
    """

    def __init__(
        self,
        n_u_eor: int,
        n_v_eor: int,
        du_eor: float = 1.0,
        dv_eor: float = 1.0,
    ) -> None:
        self.uv_eor = self.create_uv_regular_grid(
            n_u_eor, n_v_eor, du=du_eor, dv=dv_eor, exclude_monopole=True,
        )

    def create_uv_regular_grid(self,
        n_u: int,
        n_v: int,
        du: float = 1.0,
        dv: float = 1.0,
        exclude_monopole: bool = True,
    ) -> jnp.ndarray:
        """
        Create 2D rectilinear ``(u, v)`` coordinates centred on zero.

        Parameters
        ----------
        n_u : int
            Number of pixels along the u-axis.
        n_v : int
            Number of pixels along the v-axis.
        du : float, optional
            Spacing between adjacent u samples (Delta u). Defaults to 1.0.
        dv : float, optional
            Spacing between adjacent v samples (Delta v). Defaults to 1.0.
        exclude_monopole : bool, optional
            If True, remove the (u, v) = (0, 0) mode. Defaults to True.

        Returns
        -------
        uv_vec : jnp.ndarray
            Flattened (u, v) coordinates with shape (2, N_uv).
        """
        # Centered integer grid indices for u and v 
        u = jnp.arange(n_u) - n_u // 2
        v = jnp.arange(n_v) - n_v // 2
        # Build the 2D mesh and scale by spacing
        us, vs = jnp.meshgrid(u, v, indexing="xy")
        # Flatten the grids to 1D vectors 
        u_vec = (du * us).ravel()
        v_vec = (dv * vs).ravel()

        # Flat index of monopole (u=0, v=0) for indexing="xy" + C-order ravel:
        # i0 = v_idx * n_u + u_idx, with u_idx=n_u//2 and v_idx=n_v//2.
        i0 = (n_v // 2) * n_u + (n_u // 2)

        # The transverse monopole is not part of the cosmological model.
        if exclude_monopole:
            u_vec = jnp.concatenate([u_vec[:i0], u_vec[i0 + 1 :]], axis=0)
            v_vec = jnp.concatenate([v_vec[:i0], v_vec[i0 + 1 :]], axis=0)
        else:
            u_mp = u_vec[i0 : i0 + 1]
            v_mp = v_vec[i0 : i0 + 1]
            u_vec = jnp.concatenate([u_vec[:i0], u_vec[i0 + 1 :], u_mp], axis=0)
            v_vec = jnp.concatenate([v_vec[:i0], v_vec[i0 + 1 :], v_mp], axis=0)

        return jnp.stack([u_vec, v_vec], axis=0) # Shape (2, N_uv)
