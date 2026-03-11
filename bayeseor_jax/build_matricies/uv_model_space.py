"""
Model uses three spatial Fourier grids:
- EoR grid: rectilinear (u,v) modes within the EoR FoV.
    - Captures the 21cm signal fluctuations on relevant angular scales.
- FG grid: rectilinear (u,v) grid for smooth foreground emission.
    - Captures large spectral scale foreground modes.
"""
import jax
jax.config.update("jax_enable_x64", True)

from typing import Tuple
import logging
import jax.numpy as jnp


class UV_ModelSpace:
    """
    Build the (U, V) model grids used by the Bayesian sky model.

    This class constructs:
    - An EoR rectilinear grid for 21cm signal modes.
    - A foreground (FG) rectilinear grid for smooth spatial and spectral emission.

    Attributes
    ----------
    uv_eor : jnp.ndarray
        Flattened EoR (u, v) coordinates with shape (2, N_uv_eor).
    uv_fg : jnp.ndarray
        Flattened foreground (u, v) coordinates with shape (2, N_uv_fg).
    """

    def __init__(
        self,
        n_u_eor: int,
        n_v_eor: int,
        n_u_fg: int,
        n_v_fg: int,
        du_eor: float = 1.0,
        dv_eor: float = 1.0,
        du_fg: float = 1.0,
        dv_fg: float = 1.0,
        exclude_monopole: bool = True,
    ) -> None:
        
        # Create the rectilinear EoR and foreground grids.
        self.uv_eor = self.create_uv_regular_grid(
            n_u_eor, n_v_eor, du=du_eor, dv=dv_eor, exclude_monopole=True,
        )
        self.uv_fg = self.create_uv_regular_grid(
            n_u_fg, n_v_fg, du=du_fg, dv=dv_fg, exclude_monopole=exclude_monopole,
        )

    def create_uv_regular_grid(self,
        n_u: int,
        n_v: int,
        du: float = 1.0,
        dv: float = 1.0,
        exclude_monopole: bool = True,
    ) -> jnp.ndarray:
        """
        Create 2D rectilinear (u, v) grid coordinates centered at zero for 
        both 21cm EoR angular modes and smooth spectral foreground modes. 

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

        # Identify the monopole (0, 0) entry if present.
        is_monopole = (u_vec == 0) & (v_vec == 0)
        has_monopole = bool(jnp.any(is_monopole))

        # Optionally exclude the monopole mode (not present in interferometric data).
        if exclude_monopole:
            u_vec = u_vec[~is_monopole]
            v_vec = v_vec[~is_monopole]
            if not has_monopole:
                logging.info("No (u, v) = (0, 0) monopole found to remove as n_u and n_v are even")
        else:
            # If keeping the monopole, move it to the rightmost column so it is adjacent 
            # to LSSM modes downstream in the UV_ModelSpace ordering.
            if has_monopole:
                u_mp = u_vec[is_monopole]
                v_mp = v_vec[is_monopole]
                u_vec = jnp.concatenate([u_vec[~is_monopole], u_mp], axis=0)
                v_vec = jnp.concatenate([v_vec[~is_monopole], v_mp], axis=0)

        return jnp.stack([u_vec, v_vec], axis=0) # Shape (2, N_uv)
