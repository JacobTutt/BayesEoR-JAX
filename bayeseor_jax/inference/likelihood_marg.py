"""JAX-native marginalized Gaussian likelihood for BayesEoR-JAX."""

from __future__ import annotations

import importlib
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..build_matricies.jax_grid import build_likelihood_mesh, build_solver_mesh
from ..utils import load_sharded_npy
from .phi import PhiInverse
from .prior import Prior, linear_uniform_log_jacobian, theta_power_to_linear


class MarginalisedLikelihood:
    """
    Configuration for marginalised log-likelihood evaluation.

    Parameters
    ----------
    t_ninv_t : jax.Array
        Matrix ``T^H N^{-1} T``.
    dbar : jax.Array
        Vector ``T^H N^{-1} d``.
    d_ninv_d : float
        Scalar ``d^H N^{-1} d``.
    n_data : int
        Number of data samples contributing to the noise normalization term.
    phi_inv_class : PhiInverse
        Prior-precision construction settings.
    prior_class : Prior
        Sampler-space prior configuration used to evaluate the full log density.
    uniform_priors : jax.Array, optional
        Boolean mask over power-spectrum parameters whose intended physical
        prior is uniform in linear power. The sampler still works in
        ``log10(power)`` coordinates; True entries receive the Jacobian
        correction in the likelihood.
    intrinsic_noise_fitting : bool, optional
        Whether the first sampled parameter is intrinsic noise amplitude
        ``alpha_prime``.
    power_param_start : int, optional
        Start index of power-spectrum parameters within ``theta``.
    n_power_params : int
        Number of power-spectrum parameters within ``theta``.
    matrix_dir : pathlib.Path or str, optional
        Directory containing saved compressed likelihood terms. When provided,
        the constructor loads:
        - ``T_Ninv_T.npy``
        - ``dbar.npy``
        - ``d_ninv_d.npy``
        and ignores the explicit array arguments.
    distributed : bool, optional
        If True, load or redistribute the stored terms onto a 2D likelihood
        mesh for distributed evaluation.
    backend : {"jax", "jaxmg"}, optional
        Dense solver backend. ``"jax"`` uses the existing 2D-sharded
        Cholesky path. ``"jaxmg"`` uses a 1D row-sharded solver path and
        calls ``jaxmg.potrs(..., return_logdet=True)``.
    jaxmg_tile_size : int, optional
        Tile size forwarded to ``jaxmg.potrs`` when ``backend="jaxmg"``.
    mesh : jax.sharding.Mesh or None, optional
        Optional pre-built mesh used when ``distributed=True``.
    """

    def __init__(
        self,
        *,
        matrix_dir: Path | str | None = None,
        distributed: bool = False,
        backend: str = "jax",
        jaxmg_tile_size: int = 256,
        mesh: Mesh | None = None,
        t_ninv_t: jax.Array | None = None,
        dbar: jax.Array | None = None,
        d_ninv_d: float | jax.Array | None = None,
        n_data: int,
        phi_inv_class: PhiInverse,
        prior_class: Prior,
        uniform_priors: jax.Array | None = None,
        intrinsic_noise_fitting: bool = False,
        power_param_start: int = 0,
        n_power_params: int = 0,
    ) -> None:
        self.n_data = n_data
        self.phi_inv_class = phi_inv_class
        self.prior_class = prior_class
        if uniform_priors is None:
            self.uniform_priors = jnp.zeros((n_power_params,), dtype=bool)
        else:
            mask = jnp.asarray(uniform_priors, dtype=bool)
            if mask.shape != (n_power_params,):
                raise ValueError(
                    "uniform_priors must have shape "
                    f"({n_power_params},), got {mask.shape}"
                )
            self.uniform_priors = mask
        self.intrinsic_noise_fitting = intrinsic_noise_fitting
        self.power_param_start = power_param_start
        self.n_power_params = n_power_params
        if backend not in {"jax", "jaxmg"}:
            raise ValueError(f"Unsupported marginal likelihood backend: {backend}")
        self.backend = backend
        self.jaxmg_tile_size = int(jaxmg_tile_size)
        self.mesh: Mesh | None = None
        self._distributed_log_likelihood_impl = None
        self._jaxmg_module = None

        if matrix_dir is not None:
            if any(arg is not None for arg in (t_ninv_t, dbar, d_ninv_d)):
                raise ValueError(
                    "Pass either explicit likelihood arrays or matrix_dir, not both."
                )

            matrix_dir = Path(matrix_dir)
            t_ninv_t_path = matrix_dir / "T_Ninv_T.npy"
            dbar_path = matrix_dir / "dbar.npy"
            d_ninv_d_path = matrix_dir / "d_ninv_d.npy"

            for path in (t_ninv_t_path, dbar_path, d_ninv_d_path):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Required likelihood term is missing: {path}"
                    )

            if distributed:
                if self.backend == "jaxmg":
                    self.mesh = build_solver_mesh() if mesh is None else mesh
                    matrix_sharding = NamedSharding(self.mesh, P("x", None))
                else:
                    self.mesh = build_likelihood_mesh() if mesh is None else mesh
                    matrix_sharding = NamedSharding(self.mesh, P("row", "col"))
                vector_sharding = NamedSharding(self.mesh, P(None))
                scalar_sharding = NamedSharding(self.mesh, P())

                self.t_ninv_t = load_sharded_npy(t_ninv_t_path, matrix_sharding)
                self.dbar = jax.device_put(
                    jnp.asarray(np.load(dbar_path), dtype=jnp.complex128),
                    vector_sharding,
                )
                self.d_ninv_d = jax.device_put(
                    jnp.asarray(np.load(d_ninv_d_path)),
                    scalar_sharding,
                )
                if self.backend == "jaxmg":
                    self._distributed_log_likelihood_impl = self._build_jaxmg_impl()
                else:
                    self._distributed_log_likelihood_impl = self._build_distributed_impl()
            else:
                self.t_ninv_t = jnp.asarray(np.load(t_ninv_t_path))
                self.dbar = jnp.asarray(np.load(dbar_path), dtype=jnp.complex128)
                self.d_ninv_d = jnp.asarray(np.load(d_ninv_d_path))
        else:
            if any(arg is None for arg in (t_ninv_t, dbar, d_ninv_d)):
                raise ValueError(
                    "t_ninv_t, dbar, and d_ninv_d must all be provided when "
                    "matrix_dir is not used."
                )

            self.t_ninv_t = jnp.asarray(t_ninv_t)
            self.dbar = jnp.asarray(dbar)
            self.d_ninv_d = jnp.asarray(d_ninv_d)

            if distributed:
                self.enable_distributed(mesh=mesh, backend=backend)

    def _log_likelihood_exact(self, theta: jax.Array) -> jax.Array:
        """Evaluate the exact dense marginalized likelihood using stored terms."""
        return self._log_likelihood_terms(self.t_ninv_t, self.dbar, self.d_ninv_d, theta)

    def _log_likelihood_terms(
        self,
        t_ninv_t: jax.Array,
        dbar: jax.Array,
        d_ninv_d: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Evaluate the exact dense likelihood for explicit compressed terms."""
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
        jacobian_term = linear_uniform_log_jacobian(
            theta_power,
            self.uniform_priors,
        )

        t_ninv_t_scaled = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda m: m * inv_alpha2,
            lambda m: m,
            t_ninv_t,
        )
        dbar_scaled = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda v: v * inv_alpha2,
            lambda v: v,
            dbar,
        )

        diag_index = jnp.arange(t_ninv_t_scaled.shape[0])
        sigma = t_ninv_t_scaled.at[diag_index, diag_index].add(phi_i)
        chol = jnp.linalg.cholesky(sigma)

        y = jsp.linalg.solve_triangular(chol, dbar_scaled, lower=True)
        sigma_inv_dbar = jsp.linalg.solve_triangular(chol.conj().T, y, lower=False)

        log_sigma_det = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))
        log_phi_det = -jnp.sum(jnp.log(phi_i))
        quadratic = jnp.real(jnp.vdot(dbar_scaled, sigma_inv_dbar))

        base = -0.5 * log_sigma_det - 0.5 * log_phi_det + 0.5 * quadratic
        noise_term = jax.lax.cond(
            self.intrinsic_noise_fitting,
            lambda _: -0.5
            * (d_ninv_d * inv_alpha2 + self.n_data * jnp.log(alpha2)),
            lambda _: 0.0,
            operand=None,
        )
        return base + noise_term + jacobian_term

    def _build_distributed_impl(self):
        """Create the distributed exact likelihood kernel for current config."""
        def _distributed_log_likelihood(
            t_ninv_t: jax.Array,
            dbar: jax.Array,
            d_ninv_d: jax.Array,
            theta: jax.Array,
        ) -> jax.Array:
            return self._log_likelihood_terms(t_ninv_t, dbar, d_ninv_d, theta)

        return pjit(
            _distributed_log_likelihood,
            in_shardings=(P("row", "col"), P(None), P(), P(None)),
            out_shardings=P(),
        )

    def _get_jaxmg(self):
        """Import jaxmg lazily so non-jaxmg environments still import cleanly."""
        if self._jaxmg_module is None:
            try:
                self._jaxmg_module = importlib.import_module("jaxmg")
            except ImportError as exc:
                raise ImportError(
                    "MarginalisedLikelihood backend='jaxmg' requires the jaxmg "
                    "package to be importable in this environment."
                ) from exc
        return self._jaxmg_module

    def _build_jaxmg_impl(self):
        """Create the 1D row-sharded likelihood kernel using ``jaxmg``."""
        jaxmg = self._get_jaxmg()
        mesh = self.mesh
        tile_size = int(self.jaxmg_tile_size)

        if mesh is None:
            raise ValueError("jaxmg likelihood backend requires a device mesh.")

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
            """Build the donated solver matrix from the immutable stored base."""
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
        def _jaxmg_log_likelihood(
            t_ninv_t: jax.Array,
            dbar: jax.Array,
            d_ninv_d: jax.Array,
            theta: jax.Array,
        ) -> jax.Array:
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
            jacobian_term = linear_uniform_log_jacobian(
                theta_power,
                self.uniform_priors,
            )
            
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

            log_phi_det = -jnp.sum(jnp.log(phi_i))
            quadratic = jnp.real(jnp.vdot(dbar_scaled, sigma_inv_dbar))

            base = -0.5 * log_sigma_det - 0.5 * log_phi_det + 0.5 * quadratic
            noise_term = jax.lax.cond(
                self.intrinsic_noise_fitting,
                lambda _: -0.5
                * (d_ninv_d * inv_alpha2 + self.n_data * jnp.log(alpha2)),
                lambda _: 0.0,
                operand=None,
            )
            return base + noise_term + jacobian_term

        return _jaxmg_log_likelihood

    def enable_distributed(
        self,
        *,
        mesh: Mesh | None = None,
        backend: str | None = None,
    ) -> None:
        """
        Redistribute the stored compressed terms onto a solver mesh.

        This is the only distribution entrypoint. Callers do not pass in a
        separate function or sharding rule; the class applies the fixed layout
        required by the selected backend:
        - ``backend="jax"``: ``T_Ninv_T`` is 2D block-sharded over ``("row", "col")``
        - ``backend="jaxmg"``: ``T_Ninv_T`` is row-sharded over ``("x", None)``
        - ``dbar`` is replicated
        - ``d_ninv_d`` is replicated
        """
        if backend is not None:
            if backend not in {"jax", "jaxmg"}:
                raise ValueError(f"Unsupported marginal likelihood backend: {backend}")
            self.backend = backend

        if self.backend == "jaxmg":
            self.mesh = build_solver_mesh() if mesh is None else mesh
            matrix_sharding = NamedSharding(self.mesh, P("x", None))
        else:
            self.mesh = build_likelihood_mesh() if mesh is None else mesh
            matrix_sharding = NamedSharding(self.mesh, P("row", "col"))
        vector_sharding = NamedSharding(self.mesh, P(None))
        scalar_sharding = NamedSharding(self.mesh, P())

        self.t_ninv_t = jax.device_put(jnp.asarray(self.t_ninv_t), matrix_sharding)
        self.dbar = jax.device_put(jnp.asarray(self.dbar), vector_sharding)
        self.d_ninv_d = jax.device_put(jnp.asarray(self.d_ninv_d), scalar_sharding)

        if self.backend == "jaxmg":
            self._distributed_log_likelihood_impl = self._build_jaxmg_impl()
        else:
            self._distributed_log_likelihood_impl = self._build_distributed_impl()

    def log_likelihood(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate marginalized Gaussian log-likelihood.

        This integrates out the latent linear coefficient vector analytically and
        returns the old BayesEoR-style objective:

        ``-0.5*log|Sigma| - 0.5*log|Phi^{-1}| + 0.5*dbar^H*Sigma^{-1}*dbar``

        plus optional intrinsic-noise terms and, for bins marked in
        ``uniform_priors``, the old BayesEoR Jacobian correction
        ``sum(log(power_linear))``.
        """
        theta = jnp.asarray(theta)
        if self.mesh is not None and self._distributed_log_likelihood_impl is not None:
            with self.mesh:
                return self._distributed_log_likelihood_impl(
                    self.t_ninv_t,
                    self.dbar,
                    self.d_ninv_d,
                    theta,
                )
        return self._log_likelihood_exact(theta)

    def log_posterior(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate full BayesEoR-style log density for sampling.

        This folds the sampler-space prior into the marginalized Gaussian
        objective so the returned value matches the old single-call target.
        """
        theta = jnp.asarray(theta)
        theta_power = theta[
            self.power_param_start : self.power_param_start + self.n_power_params
        ]
        return self.log_likelihood(theta) + self.prior_class.logprior(theta_power)
