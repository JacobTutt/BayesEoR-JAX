"""JAXMg-backed conditional MAP solves for BayesEoR-JAX."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxmg
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..model.construction.parallel_layout import build_solver_mesh
from .phi import PhiInverse
from .prior import linear_uniform_log_jacobian, theta_power_to_linear
from ..utils import load_sharded_npy


class MaximumAPosteriori:
    """
    Conditional MAP solver for the latent linear coefficient vector.

    For fixed sampled parameters ``theta``, this class constructs the dense
    Gaussian system

    ``Sigma = T^H N^{-1} T + Phi^{-1}``

    and solves for the conditional MAP coefficients

    ``b_hat = Sigma^{-1} dbar``

    with ``dbar = T^H N^{-1} d``.

    The implementation uses the same compressed likelihood terms as the
    marginalized likelihood: ``T_Ninv_T``, ``dbar``, and ``d_ninv_d``. The
    stored normal matrix is row-sharded over a 1D solver mesh, while the
    right-hand side and scalar normalization term are replicated. Each call
    builds the diagonal prior precision implied by ``theta`` and uses
    ``jaxmg.potrs`` to solve the system and return the MAP coefficient vector.

    Parameters
    ----------
    matrix_dir : pathlib.Path or str or None, optional
        Directory containing saved compressed likelihood terms. When provided,
        the constructor loads ``T_Ninv_T.npy``, ``dbar.npy``, and
        ``d_ninv_d.npy`` from disk.
    t_ninv_t, dbar, d_ninv_d : optional
        Explicit compressed likelihood terms used when ``matrix_dir`` is not
        supplied.
    n_data : int
        Number of data samples contributing to the likelihood normalization.
    phi_inv_class : PhiInverse
        Object that maps power amplitudes to the diagonal prior precision
        ``Phi^{-1}``.
    k_vals : jax.Array or None, optional
        Mean k value per non-empty spherical bin. Required only for the
        ``ps`` -> ``dmps`` convenience conversion helpers.
    uniform_priors : jax.Array or None, optional
        Boolean mask over power bins that receive the old BayesEoR Jacobian
        correction in the marginalized objective.
    intrinsic_noise_fitting : bool, optional
        Whether the leading sampled parameter is the intrinsic noise amplitude
        ``alpha``.
    power_param_start : int, optional
        Start index of the power-spectrum block inside ``theta``.
    n_power_params : int
        Number of power-spectrum parameters in ``theta``.
    jaxmg_tile_size : int, optional
        Tile size forwarded to ``jaxmg.potrs``.
    mesh : jax.sharding.Mesh or None, optional
        Optional pre-built 1D solver mesh. When omitted, all visible devices
        are used via ``build_solver_mesh()``.
    """

    def __init__(
        self,
        *,
        matrix_dir: Path | str | None = None,
        t_ninv_t: jax.Array | None = None,
        dbar: jax.Array | None = None,
        d_ninv_d: float | jax.Array | None = None,
        n_data: int,
        phi_inv_class: PhiInverse,
        k_vals: jax.Array | None = None,
        uniform_priors: jax.Array | None = None,
        intrinsic_noise_fitting: bool = False,
        power_param_start: int = 0,
        n_power_params: int = 0,
        jaxmg_tile_size: int = 256,
        mesh: Mesh | None = None,
    ) -> None:
        self.n_data = int(n_data)
        self.phi_inv_class = phi_inv_class
        self.k_vals = None if k_vals is None else np.asarray(k_vals, dtype=float)
        self.intrinsic_noise_fitting = bool(intrinsic_noise_fitting)
        self.power_param_start = int(power_param_start)
        self.n_power_params = int(n_power_params)
        self.jaxmg_tile_size = int(jaxmg_tile_size)
        self.mesh = build_solver_mesh() if mesh is None else mesh
        self.matrix_dir = None if matrix_dir is None else Path(matrix_dir)

        if uniform_priors is None:
            self.uniform_priors = jnp.zeros((self.n_power_params,), dtype=bool)
        else:
            mask = jnp.asarray(uniform_priors, dtype=bool)
            if mask.shape != (self.n_power_params,):
                raise ValueError(
                    "uniform_priors must have shape "
                    f"({self.n_power_params},), got {mask.shape}"
                )
            self.uniform_priors = mask

        if self.matrix_dir is not None:
            if any(arg is not None for arg in (t_ninv_t, dbar, d_ninv_d)):
                raise ValueError(
                    "Pass either explicit compressed terms or matrix_dir, not both."
                )
            t_ninv_t_path = self.matrix_dir / "T_Ninv_T.npy"
            dbar_path = self.matrix_dir / "dbar.npy"
            d_ninv_d_path = self.matrix_dir / "d_ninv_d.npy"
            for path in (t_ninv_t_path, dbar_path, d_ninv_d_path):
                if not path.exists():
                    raise FileNotFoundError(f"Required MAP term is missing: {path}")

            self.t_ninv_t = load_sharded_npy(
                t_ninv_t_path,
                NamedSharding(self.mesh, P("x", None)),
            )
            self.dbar = jax.device_put(
                jnp.asarray(np.load(dbar_path), dtype=jnp.complex128),
                NamedSharding(self.mesh, P(None)),
            )
            self.d_ninv_d = jax.device_put(
                jnp.asarray(np.load(d_ninv_d_path)),
                NamedSharding(self.mesh, P()),
            )
        else:
            if any(arg is None for arg in (t_ninv_t, dbar, d_ninv_d)):
                raise ValueError(
                    "t_ninv_t, dbar, and d_ninv_d must all be provided when "
                    "matrix_dir is not used."
                )
            self.t_ninv_t = jax.device_put(
                jnp.asarray(t_ninv_t),
                NamedSharding(self.mesh, P("x", None)),
            )
            self.dbar = jax.device_put(
                jnp.asarray(dbar),
                NamedSharding(self.mesh, P(None)),
            )
            self.d_ninv_d = jax.device_put(
                jnp.asarray(d_ninv_d),
                NamedSharding(self.mesh, P()),
            )

        self._map_impl = self._build_jaxmg_map_impl()
        
    def _build_jaxmg_map_impl(self):
        """
        Build the compiled JAXMg kernel for the conditional MAP solve.

        The kernel consumes the compressed likelihood terms together with a
        sampled parameter vector ``theta``. For each call it:

        - extracts the power-spectrum block from ``theta``
        - builds the diagonal prior precision ``Phi^{-1}``
        - forms the sharded work matrix ``Sigma``
        - solves ``Sigma b_hat = dbar``
        - returns ``b_hat`` together with the scalar quantities needed to
          evaluate the associated conditional objective
        """
        mesh = self.mesh
        tile_size = int(self.jaxmg_tile_size)

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P("x", None), P(None), P()),
            out_specs=P("x", None),
            check_vma=False,
        )
        def _build_sigma_work_matrix(
            base_matrix: jax.Array,
            phi_diag: jax.Array,
            inv_alpha2: jax.Array,
        ) -> jax.Array:
            local_rows = base_matrix.shape[0]
            axis_index = jax.lax.axis_index("x")
            row_offset = axis_index * local_rows
            local_indices = jnp.arange(local_rows, dtype=jnp.int32)
            diag_cols = row_offset + local_indices
            local_phi = jax.lax.dynamic_slice_in_dim(phi_diag, row_offset, local_rows)

            sigma_local = jax.lax.cond(
                self.intrinsic_noise_fitting,
                lambda m: m * inv_alpha2,
                lambda m: jnp.astype(m, m.dtype, copy=True),
                base_matrix,
            )
            return sigma_local.at[local_indices, diag_cols].add(local_phi)

        @jax.jit
        def _jaxmg_map(
            t_ninv_t: jax.Array,
            dbar: jax.Array,
            theta: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            theta_power = theta[
                self.power_param_start : self.power_param_start + self.n_power_params
            ]
            alpha = jax.lax.cond(
                self.intrinsic_noise_fitting,
                lambda t: t[0],
                lambda t: jnp.array(1.0, dtype=t.dtype),
                theta,
            )
            alpha2 = alpha * alpha
            inv_alpha2 = 1.0 / alpha2

            power_linear = theta_power_to_linear(theta_power)
            phi_i = self.phi_inv_class.evaluate(power_linear)

            dbar_scaled = jax.lax.cond(
                self.intrinsic_noise_fitting,
                lambda v: v * inv_alpha2,
                lambda v: v,
                dbar,
            )

            sigma = _build_sigma_work_matrix(t_ninv_t, phi_i, inv_alpha2)
            sigma_inv_dbar, log_sigma_det = jaxmg.potrs(
                sigma,
                dbar_scaled,
                T_A=tile_size,
                mesh=mesh,
                in_specs=(P("x", None),),
                return_logdet=True,
            )
            sigma_inv_dbar = jnp.ravel(sigma_inv_dbar)
            quadratic = jnp.real(jnp.vdot(dbar_scaled, sigma_inv_dbar))
            return sigma_inv_dbar, quadratic, phi_i, log_sigma_det, alpha2

        return _jaxmg_map

    def _theta_from_dmps(self, dmps: float | np.ndarray) -> np.ndarray:
        """Convert linear ``dmps`` values into sampler-space ``theta``."""
        if self.k_vals is None:
            raise ValueError("k_vals are required for ps/dmps convenience helpers.")
        if hasattr(dmps, "__iter__"):
            dmps_arr = np.asarray(dmps, dtype=float)
            if dmps_arr.size != self.k_vals.size:
                raise ValueError(
                    f"'dmps' has size {dmps_arr.size}, expected {self.k_vals.size}"
                )
        else:
            dmps_arr = float(dmps) * np.ones_like(self.k_vals)
        if np.any(dmps_arr <= 0):
            raise ValueError("dmps must be strictly positive to map into log10-space.")
        return np.log10(dmps_arr)

    def calculate_dmps(
        self,
        *,
        ps: float | np.ndarray | None = None,
        dmps: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert ``ps`` or ``dmps`` inputs into linear ``dmps`` values."""
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None.")
        if self.k_vals is None:
            raise ValueError("k_vals are required for ps/dmps convenience helpers.")

        if ps is not None:
            if hasattr(ps, "__iter__"):
                ps_arr = np.asarray(ps, dtype=float)
                if ps_arr.size != self.k_vals.size:
                    raise ValueError(
                        f"'ps' has size {ps_arr.size}, expected {self.k_vals.size}"
                    )
            else:
                ps_arr = float(ps) * np.ones_like(self.k_vals)
            return self.k_vals**3 / (2.0 * np.pi**2) * ps_arr

        if hasattr(dmps, "__iter__"):
            dmps_arr = np.asarray(dmps, dtype=float)
            if dmps_arr.size != self.k_vals.size:
                raise ValueError(
                    f"'dmps' has size {dmps_arr.size}, expected {self.k_vals.size}"
                )
        else:
            dmps_arr = float(dmps) * np.ones_like(self.k_vals)
        return dmps_arr

    def map_estimate(
        self,
        theta: jax.Array | None = None,
        *,
        ps: float | np.ndarray | None = None,
        dmps: float | np.ndarray | None = None,
        return_prior_precision: bool = False,
    ):
        r"""
        Calculate the conditional MAP coefficients for a fixed ``theta``.

        Parameters
        ----------
        theta : jax.Array or None, optional
            Full sampled parameter vector in the same coordinates consumed by
            the marginalized likelihood. If omitted, one of `ps` or `dmps`
            must be provided and is interpreted as the power block only.
        ps, dmps : optional
            Convenience power-spectrum inputs used to construct the power block
            of ``theta``. These follow the old MAP helper interface.
        return_prior_precision : bool, optional
            When ``True``, append the diagonal prior precision ``Phi^{-1}``.

        Returns
        -------
        tuple
            By default returns ``(map_coeffs, log_post)``. ``phi_i`` is
            appended when ``return_prior_precision=True``.
        """
        if theta is None:
            if self.intrinsic_noise_fitting:
                raise ValueError(
                    "theta must be provided explicitly when intrinsic_noise_fitting=True."
                )
            dmps_arr = self.calculate_dmps(ps=ps, dmps=dmps)
            theta = self._theta_from_dmps(dmps_arr)
        theta = jnp.asarray(theta, dtype=jnp.float64)

        with self.mesh:
            map_coeffs, quadratic, phi_i, log_sigma_det, alpha2 = self._map_impl(
                self.t_ninv_t,
                self.dbar,
                theta,
            )

        log_phi_det = -jnp.sum(jnp.log(phi_i))
        noise_term = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda _: -0.5 * (self.d_ninv_d / alpha2 + self.n_data * jnp.log(alpha2)),
            lambda _: 0.0,
            operand=None,
        )

        theta_power = theta[
            self.power_param_start : self.power_param_start + self.n_power_params
        ]
        log_post = (
            -0.5 * log_sigma_det
            - 0.5 * log_phi_det
            + 0.5 * quadratic
            + noise_term
            + linear_uniform_log_jacobian(theta_power, self.uniform_priors)
        )

        outputs: list[object] = [np.asarray(map_coeffs)]
        outputs.append(float(np.asarray(log_post)))
        if return_prior_precision:
            outputs.append(np.asarray(phi_i))
        return tuple(outputs)

    def log_posterior_at_map(self, theta: jax.Array) -> float:
        """Evaluate the old BayesEoR-style marginalized objective at fixed ``theta``."""
        return self.map_estimate(theta)[1]
