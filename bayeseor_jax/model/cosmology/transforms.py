"""
Convert observational coordinates to cosmological distances and volume factors.

The module provides conversion factors used in 21-cm analyses:
- `f <-> z` via `1 + z = f_21 / f`
- `dchi/df` along line of sight
- `D_M(z)` per radian in the transverse direction
- `dV / (dOmega df) = D_M^2 * dchi/df`

Conventions
-----------
- Frequency `f` is in `Hz`.
- Angles are in `radians` (`sr` for solid angle).
- Comoving distances are in `Mpc`.
- `inst_to_cosmo_vol` returns `Mpc^3 / (sr Hz)` in real-space coordinates.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck18

jax.config.update("jax_enable_x64", True)

@dataclass(frozen=True)
class Cosmology:
    """
    FRW cosmology helper for 21-cm coordinate and volume conversion transforms.

    Parameters
    ----------
    H0_km_s_Mpc : float
        Hubble constant `H0` in `(km / s) / Mpc`.
    Om0 : float
        Present-day matter density parameter.
    Ode0 : float
        Present-day dark-energy density parameter.
    Ok0 : float
        Present-day curvature density parameter.
    Or0 : float
        Present-day radiation density parameter, including the Planck18
        photon and neutrino contributions.
    f_21_hz : float
        Rest frequency of the 21-cm line in `Hz`.
    c_ms : float
        Speed of light in `m/s`.

    Notes
    -----
    Defaults are loaded from ``astropy.cosmology.Planck18``. Expansion rates
    and transverse distances are evaluated with that fixed Astropy cosmology
    on the host and returned as JAX arrays; those methods are not JIT-compatible.
    Changing the stored density parameters does not change these Astropy
    evaluations. Frequency/redshift and line-of-sight conversions use the
    stored rest frequency, speed of light, and Hubble constant where applicable.
    """

    H0_km_s_Mpc: float = float(Planck18.H0.value)
    Om0: float = float(Planck18.Om0)
    Ode0: float = float(Planck18.Ode0)
    Ok0: float = float(Planck18.Ok0)
    Or0: float = float(Planck18.Ogamma0 + Planck18.Onu0)
    f_21_hz: float = 1.42040575177e9
    c_ms: float = float(c.to("m/s").value)

    def f2z(self, f_hz: jnp.ndarray) -> jnp.ndarray:
        """
        Convert observing frequency to redshift via the 21-cm rest frequency.

        Parameters
        ----------
        f_hz : jnp.ndarray
            Observing frequency in `Hz`.

        Returns
        -------
        z : jnp.ndarray
            Redshift with broadcasted shape of `f_hz`.
        """
        # Apply `1 + z = f_21 / f`.
        return (self.f_21_hz / f_hz) - 1.0

    def z2f(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Convert redshift to observing frequency.

        Parameters
        ----------
        z : jnp.ndarray
            Redshift.

        Returns
        -------
        f_hz : jnp.ndarray
            Observed frequency in `Hz`, same broadcasted shape as `z`.
        """
        # Apply `f = f_21 / (1 + z)`.
        return self.f_21_hz / (1.0 + z)

    def e_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate dimensionless expansion rate `E(z) = H(z)/H0`.

        Parameters
        ----------
        z : jnp.ndarray
            Redshift.

        Returns
        -------
        e_z : jnp.ndarray
            Dimensionless expansion-rate factor, same shape as input.

        Notes
        -----
        Uses ``Planck18.efunc`` on the host, independently of the density
        parameters stored on this instance.
        """
        # Convert to NumPy for Astropy Planck18 evaluation to match legacy behavior exactly.
        z_np = np.asarray(z, dtype=np.float64)
        e_np = Planck18.efunc(z_np)
        # Convert back to JAX array for downstream JAX pipeline compatibility.
        return jnp.asarray(e_np, dtype=jnp.float64)

    def dL_df(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute LOS comoving conversion factor magnitude per observing frequency.

        Parameters
        ----------
        z : jnp.ndarray
            Redshift.

        Returns
        -------
        dL_df : jnp.ndarray
            Positive conversion factor in `Mpc / Hz`, same shape as `z`.

        Notes
        -----
        Uses `|dchi/df| = D_H * (1+z)^2 / (f_21 * E(z))` with
        `D_H = c/H0`. The signed derivative `dchi/df` is negative.
        """
        # Construct: Hubble distance `D_H = c / H0` in `Mpc`.
        d_h_mpc = (self.c_ms / 1e3) / self.H0_km_s_Mpc
        # Compute: apply `|dchi/df| = D_H * (1+z)^2 / (f_21 * E(z))`.
        return d_h_mpc * (1.0 + z) ** 2 / (self.f_21_hz * self.e_z(z))

    def dL_dth(self, z: jnp.ndarray, n_z_samples: int = 2048) -> jnp.ndarray:
        """
        Compute transverse comoving distance `D_M(z)` per radian.

        Parameters
        ----------
        z : jnp.ndarray
            Scalar or array of redshifts; the output preserves its shape.
        n_z_samples : int, optional
            Unused compatibility argument, default 2048. Astropy controls
            the distance calculation rather than an explicit sampling grid.

        Returns
        -------
        dL_dth : jnp.ndarray
            `D_M` in `Mpc/rad` (numerically `Mpc` since rad is dimensionless).
            Returns scalar for scalar `z`; otherwise shape `(N,)`.

        Notes
        -----
        Uses ``Planck18.comoving_transverse_distance`` on the host. This
        distance converts transverse angular Fourier modes to cosmological
        wavenumbers through ``k_perp = 2*pi*sqrt(u**2 + v**2)/D_M``.
        """
        # Convert to NumPy and evaluate Astropy transverse comoving distance for exact parity.
        # `n_z_samples` is retained for API compatibility with earlier integration-based version.
        z_np = np.asarray(z, dtype=np.float64)
        d_m_np = Planck18.comoving_transverse_distance(z_np).value
        d_m = jnp.asarray(d_m_np, dtype=jnp.float64)
        # Return: preserve scalar-vs-vector API.
        return d_m

    def inst_to_cosmo_vol(self, z: jnp.ndarray, n_z_samples: int = 2048) -> jnp.ndarray:
        """
        Convert `(dOmega, df)` elements to a comoving volume conversion factor.

        Parameters
        ----------
        z : jnp.ndarray
            Redshift array; output shape follows input/broadcast.
        n_z_samples : int, optional
            Unused compatibility argument forwarded to ``dL_dth``.

        Returns
        -------
        inst_to_cosmo_vol : jnp.ndarray
            `dV/(dOmega df)` in `Mpc^3 / (sr Hz)`.
        """
        # Construct: transverse and LOS conversion factors.
        d_transverse = self.dL_dth(z, n_z_samples=n_z_samples)
        d_los = self.dL_df(z)
        # Compute: `dV/(dOmega df) = D_M(z)^2 * |dchi/df|`.
        return d_transverse**2 * d_los
