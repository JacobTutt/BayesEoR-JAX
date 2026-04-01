"""JAXMg-backed marginalized Gaussian likelihood for BayesEoR-JAX."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxmg
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..build_matricies.jax_grid import build_solver_mesh
from ..utils import load_sharded_npy
from .phi import PhiInverse
from .prior import Prior, linear_uniform_log_jacobian, theta_power_to_linear


class MarginalisedLikelihood:
    """
    Configuration for marginalized log-likelihood evaluation.

    This class evaluates the analytically marginalized BayesEoR target from
    the compressed matrix stack

    ``T^H N^{-1} T``, ``T^H N^{-1} d``, and ``d^H N^{-1} d``.

    The implementation is intentionally centered on the JAXMg solve path. The
    large dense matrix is kept row-sharded over a 1D device mesh, while the
    right-hand side vector and scalar normalization term are replicated. Each
    likelihood call builds the diagonal prior precision ``Phi^{-1}``, forms

    ``Sigma = T^H N^{-1} T + Phi^{-1}``

    in a sharded work matrix, and then calls ``jaxmg.potrs(...,
    return_logdet=True)`` to obtain both ``Sigma^{-1} dbar`` and
    ``log|Sigma|``.

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
    jaxmg_tile_size : int, optional
        Tile size forwarded to ``jaxmg.potrs``.
    mesh : jax.sharding.Mesh or None, optional
        Optional pre-built 1D solver mesh. When omitted, the likelihood uses
        all visible devices via ``build_solver_mesh()``. On a single GPU this
        still constructs a mesh of size one, so the same JAXMg solver path is
        used on both 1-GPU and multi-GPU runs.
    """

    def __init__(
        self,
        *,
        matrix_dir: Path | str | None = None,
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
        # Store the static metadata needed on every likelihood call.
        self.n_data = n_data
        self.phi_inv_class = phi_inv_class
        self.prior_class = prior_class

        # ``uniform_priors`` does not change sampler coordinates here; it only
        # controls which power bins receive the old BayesEoR Jacobian
        # correction inside the likelihood evaluation.
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
        self.jaxmg_tile_size = int(jaxmg_tile_size)

        # The marginalized likelihood always runs on the row-sharded JAXMg
        # solver layout, even on a single GPU. A 1-GPU run simply uses a mesh
        # of size one, which keeps the execution path identical across scales.
        self.mesh = build_solver_mesh() if mesh is None else mesh
        self._log_likelihood_impl = None

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

            # Load the saved compressed likelihood terms directly onto the
            # solver layout expected by the JAXMg kernel:
            # - ``T_Ninv_T`` row-sharded over mesh axis ``x``
            # - ``dbar`` replicated
            # - ``d_ninv_d`` replicated
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

            # Explicit in-memory inputs are pushed into the same row-sharded /
            # replicated layout used for the on-disk initialization path so
            # the compiled kernel sees one consistent runtime contract.
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

        # Build the compiled JAXMg kernel once after the static layout and
        # stored terms are in place.
        self._log_likelihood_impl = self._build_jaxmg_impl()

    def _build_jaxmg_impl(self):
        """
        Build the compiled JAXMg likelihood kernel for the current mesh.

        The returned function expects:
        - ``t_ninv_t`` row-sharded over ``("x", None)``
        - ``dbar`` replicated
        - ``d_ninv_d`` replicated
        - ``theta`` replicated

        Internally it uses a small ``shard_map`` helper to construct the local
        rows of the solver work matrix ``Sigma`` and then calls
        ``jaxmg.potrs`` to compute both the solve and the log-determinant in
        one factorization pass.
        """
        mesh = self.mesh
        tile_size = int(self.jaxmg_tile_size)

        if mesh is None:
            raise ValueError("MarginalisedLikelihood requires a solver mesh.")

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
            """
            Build the local rows of the solver work matrix ``Sigma``.

            ``base_matrix`` already holds the row block of ``T^H N^{-1} T``
            owned by the current device. ``phi_diag`` is replicated, so each
            shard slices out the segment that corresponds to its local row
            range, identifies the matching global diagonal columns, and adds
            the prior precision locally. This avoids materializing a separate
            full dense ``Sigma`` on any single device.

            When intrinsic-noise fitting is enabled, the stored
            ``T^H N^{-1} T`` block is also rescaled by ``alpha^{-2}`` before
            the diagonal prior term is added.
            """
            # Each device owns a contiguous row block. Convert that ownership
            # into a global row offset so we can place the local ``Phi^{-1}``
            # slice onto the correct diagonal columns of the full matrix.
            local_rows = base_matrix.shape[0]
            axis_index = jax.lax.axis_index("x")
            row_offset = axis_index * local_rows
            local_indices = jnp.arange(local_rows, dtype=jnp.int32)
            diag_cols = row_offset + local_indices
            local_phi = jax.lax.dynamic_slice_in_dim(phi_diag, row_offset, local_rows)

            # ``sigma_local`` is a writable work buffer derived from the stored
            # immutable base matrix. JAXMg is allowed to treat this as the
            # factorization input for the current likelihood call.
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
            """
            Evaluate the marginalized log-likelihood through the JAXMg solver.

            The algebra matches the old BayesEoR marginalized target:

            ``-0.5 log|Sigma| - 0.5 log|Phi^{-1}| + 0.5 dbar^H Sigma^{-1} dbar``

            plus the optional intrinsic-noise normalization term and the
            sampler-to-physical Jacobian correction for bins flagged by
            ``uniform_priors``.
            """
            # Extract just the power-spectrum block from the sampled parameter
            # vector, then convert it into the linear amplitudes expected by
            # ``PhiInverse``.
            theta_power = theta[
                self.power_param_start : self.power_param_start + self.n_power_params
            ]

            # If intrinsic noise is being fitted, the leading sampled
            # parameter is the noise amplitude ``alpha``. The compressed terms
            # involving ``N^{-1}`` then pick up factors of ``alpha^{-2}``.
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

            # ``dbar`` is scaled in the same way as ``T^H N^{-1} T`` under the
            # intrinsic-noise reparameterization.
            dbar_scaled = jax.lax.cond(
                self.intrinsic_noise_fitting,
                lambda v: v * inv_alpha2,
                lambda v: v,
                dbar,
            )

            # Build the sharded solver matrix and ask JAXMg for both the solve
            # and ``log|Sigma|`` from the same factorization.
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

            # ``log_phi_det`` is trivial because ``Phi^{-1}`` is diagonal.
            # The quadratic term is the data-dependent piece of the
            # marginalized Gaussian objective.
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

    def log_likelihood(self, theta: jax.Array) -> jax.Array:
        """
        Evaluate marginalized Gaussian log-likelihood.

        This integrates out the latent linear coefficient vector analytically,
        evaluates the resulting dense system with ``jaxmg.potrs`` on the stored
        row-sharded solver mesh, and returns the old BayesEoR-style objective
        with optional intrinsic-noise and Jacobian corrections.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        with self.mesh:
            return self._log_likelihood_impl(
                self.t_ninv_t,
                self.dbar,
                self.d_ninv_d,
                theta,
            )

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
