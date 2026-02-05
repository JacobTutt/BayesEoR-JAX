"""
Model uses three spatial Fourier grids:
- EoR grid: rectilinear (u,v) modes within the EoR FoV.
    - Captures the 21cm signal fluctuations on relevant angular scales.
- FG grid: rectilinear (u,v) grid for smooth foreground emission.
    - Captures large spectral scale foreground modes.
- SHG: extra low |u|,|v| modes to capture scales larger than the main grids.
    - Captures large angular scale foreground modes.
"""
from typing import Tuple
import logging
import jax.numpy as jnp


class UVEta_ModelSpace:
    """
    Build the (U, V, eta) model grids used by the Bayesian sky model.

    This class constructs:
    - An EoR rectilinear grid for 21cm signal modes.
    - A foreground (FG) rectilinear grid for smooth spectral emission.
    - A subharmonic grid (SHG) for large angular-scale foreground structure.

    Attributes
    ----------
    uv_eor : jnp.ndarray
        Flattened EoR (u, v) coordinates with shape (N_uv_eor, 2).
    uv_fg : jnp.ndarray
        Flattened foreground (u, v) coordinates with shape (N_uv_fg, 2).
    uv_sh : jnp.ndarray
        Flattened subharmonic (u, v) coordinates with shape (N_uv_sh, 2).
    """

    def __init__(
        self,
        n_u_eor: int,
        n_v_eor: int,
        n_u_fg: int,
        n_v_fg: int,
        n_u_sh: int,
        n_v_sh: int,
        du_eor: float = 1.0,
        dv_eor: float = 1.0,
        du_fg: float = 1.0,
        dv_fg: float = 1.0,
        du_sh: float = 1.0,
        dv_sh: float = 1.0,
        sh_spacing: str = "linear",
        min_u_sh: float = 0.5,
        min_v_sh: float = 0.5,
        max_u_sh: float | None = None,
        max_v_sh: float | None = None,
        exclude_monopole: bool = True,
    ) -> None:
        
        # Create the rectilinear EoR and foreground grids.
        self.uv_eor = self.create_uv_regular_grid(
            n_u_eor, n_v_eor, du=du_eor, dv=dv_eor, exclude_monopole=exclude_monopole,
        )
        self.uv_fg = self.create_uv_regular_grid(
            n_u_fg, n_v_fg, du=du_fg, dv=dv_fg, exclude_monopole=exclude_monopole,
        )

        # Create the subharmonic grid (linear or log spacing).
        self.uv_sh = self.create_uv_subharmonic_grid(
            n_u_sh, n_v_sh, spacing=sh_spacing, du_sh=du_sh, dv_sh=dv_sh,
            min_u_sh=min_u_sh, min_v_sh=min_v_sh, max_u_sh=max_u_sh, max_v_sh=max_v_sh,
            exclude_monopole=exclude_monopole,
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
            Flattened (u, v) coordinates with shape (N_uv, 2).
        """
        # Centered integer grid indices for u and v 
        u = jnp.arange(n_u) - n_u // 2
        v = jnp.arange(n_v) - n_v // 2
        # Build the 2D mesh and scale by spacing
        us, vs = jnp.meshgrid(u, v, indexing="xy")
        # Flatten the grids to 1D vectors 
        u_vec = (du * us).ravel()
        v_vec = (dv * vs).ravel()

        # Optionally exclude the monopole mode as U,V = 0 is not present in interferometric data
        if exclude_monopole:
            # Drop the (0, 0) mode if it exists in the flattened arrays
            is_monopole = (u_vec == 0) & (v_vec == 0)
            non_monopole = ~is_monopole
            u_vec = u_vec[non_monopole]
            v_vec = v_vec[non_monopole]
            # Warn if no monopole was found to remove (simple if n_u and n_v are even)
            if not bool(jnp.any(is_monopole)):
                logging.info("No (u, v) = (0, 0) monopole found to remove as n_u and n_v are even")


        return jnp.stack([u_vec, v_vec], axis=-1)

    def create_uv_subharmonic_grid(self,
        n_u: int,
        n_v: int,
        spacing: str = "linear",
        du_sh: float = 1.0,
        dv_sh: float = 1.0,
        min_u_sh: float = 0.5,
        min_v_sh: float = 0.5,
        max_u_sh: float | None = None,
        max_v_sh: float | None = None,
        exclude_monopole: bool = True,
    ) -> jnp.ndarray:
        """
        Create 2D subharmonic (u, v) grid coordinates defined on a larger angular scale
        than the main model grids (ie. smaller |u|, |v| values) to capture large galactic
        foreground modes. Supports both linear and logarithmic spacing of subharmonic modes.

        Parameters
        ----------
        n_u : int
            Number of pixels along the u-axis in the subharmonic grid.
        n_v : int
            Number of pixels along the v-axis in the subharmonic grid.
        spacing : {"linear", "log"}, optional
            Spacing rule for subharmonic modes. Defaults to "linear".
        du_sh : float, optional
            Linear spacing between adjacent subharmonic u samples.
        dv_sh : float, optional
            Linear spacing between adjacent subharmonic v samples.
        min_u_sh : float, optional
            Minimum |u| for log-spaced subharmonic modes.
        min_v_sh : float, optional
            Minimum |v| for log-spaced subharmonic modes.
        max_u_sh : float, optional
            Maximum |u| for log-spaced subharmonic modes.
        max_v_sh : float, optional
            Maximum |v| for log-spaced subharmonic modes.
        exclude_monopole : bool, optional
            If True, remove the (u, v) = (0, 0) mode. Defaults to True.

        Returns
        -------
        uv_vec : jnp.ndarray
            Flattened (u, v) coordinates with shape (N_uv, 2).
        """
        # Type check for spacing parameter 
        if spacing not in ("linear", "log"):
            raise ValueError("spacing must be 'linear' or 'log'")
        
        # Linear subharmonic grid centered at zero.
        if spacing == "linear":
            # Centered integer grid indices for u and v
            u = jnp.arange(n_u) - n_u // 2
            v = jnp.arange(n_v) - n_v // 2
            # Build the 2D mesh and scale by spacing
            us, vs = jnp.meshgrid(u, v, indexing="xy")
            # Flatten the grids to 1D vectors
            u_vec = (du_sh * us).ravel()
            v_vec = (dv_sh * vs).ravel()
        
        # Log-spaced modes with symmetric positive/negative branches.
        else:
            # Set the positive values for u and v
            u_pos = jnp.logspace(jnp.log10(min_u_sh), jnp.log10(max_u_sh), n_u // 2, endpoint=False)
            v_pos = jnp.logspace(jnp.log10(min_v_sh), jnp.log10(max_v_sh), n_v // 2, endpoint=False)
            # Mirror positive values to negative side and include zero mode
            u = jnp.concatenate((-u_pos[::-1], jnp.zeros(1), u_pos))
            v = jnp.concatenate((-v_pos[::-1], jnp.zeros(1), v_pos))
            # Build the 2D mesh
            us, vs = jnp.meshgrid(u, v, indexing="xy")
            # Flatten the grids to 1D vectors
            u_vec = us.ravel()
            v_vec = vs.ravel()


        # Optionally exclude the monopole mode as U,V = 0 is not present in interferometric data
        if exclude_monopole:
            # Drop the (0, 0) mode if it exists in the flattened arrays
            is_monopole = (u_vec == 0) & (v_vec == 0)
            non_monopole = ~is_monopole
            u_vec = u_vec[non_monopole]
            v_vec = v_vec[non_monopole]
            # Warn if no monopole was found to remove (simple if n_u and n_v are even)
            if not bool(jnp.any(is_monopole)):
                logging.info("No (u, v) = (0, 0) monopole found to remove as n_u and n_v are even")

        return jnp.stack([u_vec, v_vec], axis=-1)
