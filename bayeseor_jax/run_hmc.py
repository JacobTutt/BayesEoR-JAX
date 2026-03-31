"""BlackJAX NUTS runner for the full-state BayesEoR-JAX likelihood."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import blackjax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from .inference.likelihood_full import FullLikelihood
from .inference.likelihood_marg import MarginalisedLikelihood
from .inference.prior import theta_power_to_linear
from .setup import SetupResult


def _build_full_likelihood(setup_result: SetupResult) -> FullLikelihood:
    """Instantiate the explicit full-state likelihood from setup outputs."""
    matrix_dir = Path(setup_result.bm.array_save_directory)
    t_matrix = jnp.asarray(np.load(matrix_dir / "T.npy", mmap_mode="r"), dtype=jnp.complex128)
    ninv = jnp.asarray(np.load(matrix_dir / "Ninv.npy")).reshape(-1)
    vis_noisy = jnp.asarray(setup_result.vis_dict["vis_noisy"], dtype=jnp.complex128)

    prior_class = setup_result.prior_class
    likelihood_class = setup_result.likelihood_class

    return FullLikelihood(
        t_matrix=t_matrix,
        vis_noisy=vis_noisy,
        ninv=ninv,
        n_data=int(vis_noisy.size),
        phi_inv_class=setup_result.phi_inverse_class,
        prior_class=prior_class,
        uniform_priors=likelihood_class.uniform_priors,
        intrinsic_noise_fitting=bool(likelihood_class.intrinsic_noise_fitting),
        power_param_start=int(likelihood_class.power_param_start),
        n_power_params=int(likelihood_class.n_power_params),
    )


def _build_marginalised_likelihood(
    setup_result: SetupResult,
) -> MarginalisedLikelihood:
    """Instantiate the JAXMg marginalized likelihood from saved terms."""
    matrix_dir = Path(setup_result.bm.array_save_directory)
    prior_class = setup_result.prior_class
    likelihood_class = setup_result.likelihood_class

    return MarginalisedLikelihood(
        matrix_dir=str(matrix_dir),
        n_data=int(setup_result.vis_dict["vis_noisy"].size),
        phi_inv_class=setup_result.phi_inverse_class,
        prior_class=prior_class,
        uniform_priors=likelihood_class.uniform_priors,
        intrinsic_noise_fitting=bool(likelihood_class.intrinsic_noise_fitting),
        power_param_start=int(likelihood_class.power_param_start),
        n_power_params=int(likelihood_class.n_power_params),
    )


def _build_unconstrained_logdensity(
    full_likelihood: FullLikelihood,
    initial_linear_position: jax.Array | None = None,
):
    """
    Build an unconstrained HMC/NUTS target with simple Stan-style transforms.

    Power-spectrum parameters are bounded box variables in the sampler-space
    coordinates already used by ``Prior``. We map unconstrained reals into
    those intervals with a logistic transform and include the log-Jacobian.

    If intrinsic noise fitting is enabled, the leading parameter is mapped to
    ``alpha = exp(alpha_u)`` so the sampled noise amplitude is positive.
    Linear coefficients are left unconstrained.
    """
    lower = jnp.asarray(full_likelihood.prior_class.lower, dtype=jnp.float64)
    upper = jnp.asarray(full_likelihood.prior_class.upper, dtype=jnp.float64)
    widths = upper - lower

    n_alpha = 1 if full_likelihood.intrinsic_noise_fitting else 0
    n_power = int(full_likelihood.n_power_params)
    n_linear = int(full_likelihood.n_linear_params)
    unconstrained_dim = n_alpha + n_power + n_linear

    def transform(position_u: jax.Array) -> tuple[jax.Array, jax.Array]:
        position_u = jnp.asarray(position_u, dtype=jnp.float64)
        if position_u.shape != (unconstrained_dim,):
            raise ValueError(
                f"unconstrained position must have shape ({unconstrained_dim},), "
                f"got {position_u.shape}"
            )

        offset = 0
        blocks = []
        log_jac = jnp.array(0.0, dtype=jnp.float64)

        if n_alpha:
            alpha_u = position_u[0]
            alpha = jnp.exp(alpha_u)
            blocks.append(alpha[None])
            log_jac = log_jac + alpha_u
            offset = 1

        power_u = position_u[offset : offset + n_power]
        sigma = jnn.sigmoid(power_u)
        theta_power = lower + widths * sigma
        log_jac = log_jac + jnp.sum(
            jnp.log(widths) + jnn.log_sigmoid(power_u) + jnn.log_sigmoid(-power_u)
        )
        blocks.append(theta_power)

        linear_u = position_u[offset + n_power :]
        blocks.append(linear_u)

        theta = jnp.concatenate(blocks, axis=0)
        return theta, log_jac

    def logdensity_fn(position_u: jax.Array) -> jax.Array:
        theta, log_jac = transform(position_u)
        return full_likelihood.log_probability(theta) + log_jac

    initial_position = jnp.zeros((unconstrained_dim,), dtype=jnp.float64)
    if initial_linear_position is not None:
        initial_linear_position = jnp.asarray(initial_linear_position, dtype=jnp.float64)
        if initial_linear_position.shape != (n_linear,):
            raise ValueError(
                f"initial_linear_position must have shape ({n_linear},), "
                f"got {initial_linear_position.shape}"
            )
        initial_position = initial_position.at[n_alpha + n_power :].set(
            initial_linear_position
        )
    return logdensity_fn, transform, initial_position


def _build_unconstrained_logdensity_marginalised(
    marginal_likelihood: MarginalisedLikelihood,
):
    """
    Build an unconstrained HMC/NUTS target for the marginalized likelihood.

    The sampled state only contains:
    - optional positive intrinsic noise amplitude ``alpha``
    - bounded power-spectrum parameters in the sampler-space coordinates used
      by ``Prior``

    The power box is mapped with a logistic transform and the Jacobian is
    included in the unconstrained target.
    """
    lower = jnp.asarray(marginal_likelihood.prior_class.lower, dtype=jnp.float64)
    upper = jnp.asarray(marginal_likelihood.prior_class.upper, dtype=jnp.float64)
    widths = upper - lower

    n_alpha = 1 if marginal_likelihood.intrinsic_noise_fitting else 0
    n_power = int(marginal_likelihood.n_power_params)
    unconstrained_dim = n_alpha + n_power

    def transform(position_u: jax.Array) -> tuple[jax.Array, jax.Array]:
        position_u = jnp.asarray(position_u, dtype=jnp.float64)
        if position_u.shape != (unconstrained_dim,):
            raise ValueError(
                f"unconstrained position must have shape ({unconstrained_dim},), "
                f"got {position_u.shape}"
            )

        offset = 0
        blocks = []
        log_jac = jnp.array(0.0, dtype=jnp.float64)

        if n_alpha:
            alpha_u = position_u[0]
            alpha = jnp.exp(alpha_u)
            blocks.append(alpha[None])
            log_jac = log_jac + alpha_u
            offset = 1

        power_u = position_u[offset : offset + n_power]
        sigma = jnn.sigmoid(power_u)
        theta_power = lower + widths * sigma
        log_jac = log_jac + jnp.sum(
            jnp.log(widths) + jnn.log_sigmoid(power_u) + jnn.log_sigmoid(-power_u)
        )
        blocks.append(theta_power)

        theta = jnp.concatenate(blocks, axis=0)
        return theta, log_jac

    def logdensity_terms(
        t_ninv_t: jax.Array,
        dbar: jax.Array,
        d_ninv_d: jax.Array,
        position_u: jax.Array,
    ) -> jax.Array:
        theta, log_jac = transform(position_u)
        theta_power = theta[
            marginal_likelihood.power_param_start : marginal_likelihood.power_param_start
            + marginal_likelihood.n_power_params
        ]
        power_prior = marginal_likelihood.prior_class.logprior(theta_power)
        with marginal_likelihood.mesh:
            log_like = marginal_likelihood._log_likelihood_impl(
                t_ninv_t,
                dbar,
                d_ninv_d,
                theta,
            )
        return log_like + power_prior + log_jac

    logdensity_fn = jax.jit(
        lambda position_u: logdensity_terms(
            marginal_likelihood.t_ninv_t,
            marginal_likelihood.dbar,
            marginal_likelihood.d_ninv_d,
            position_u,
        )
    )
    initial_position = jnp.zeros((unconstrained_dim,), dtype=jnp.float64)
    return logdensity_fn, transform, initial_position


def _build_conditional_map_linear_init(
    setup_result: SetupResult,
    full_likelihood: FullLikelihood,
) -> np.ndarray:
    """
    Build a conditional MAP start for the latent linear block.

    The current full-state parameterization samples a real latent vector, so
    the stationary point of the quadratic conditional target is obtained from

    ``(Re(T^H N^{-1} T) + Phi^{-1}) x_map = Re(T^H N^{-1} d)``.

    This uses the saved compressed terms from setup and the midpoint power
    parameters implied by the box prior bounds.
    """
    matrix_dir = Path(setup_result.bm.array_save_directory)
    t_ninv_t_path = matrix_dir / "T_Ninv_T.npy"
    dbar_path = matrix_dir / "dbar.npy"

    if not t_ninv_t_path.exists() or not dbar_path.exists():
        return np.zeros((full_likelihood.n_linear_params,), dtype=np.float64)

    theta_power_mid = 0.5 * (
        np.asarray(full_likelihood.prior_class.lower)
        + np.asarray(full_likelihood.prior_class.upper)
    )
    power_linear = np.asarray(
        theta_power_to_linear(
            jnp.asarray(theta_power_mid, dtype=jnp.float64),
        )
    )
    phi_i = np.asarray(
        full_likelihood.phi_inv_class.evaluate(jnp.asarray(power_linear))
    )

    t_ninv_t = np.load(t_ninv_t_path, mmap_mode="r")
    dbar = np.load(dbar_path, mmap_mode="r")
    system = np.array(np.real(t_ninv_t), dtype=np.float64, copy=True)
    rhs = np.asarray(np.real(dbar), dtype=np.float64)
    diag_index = np.diag_indices_from(system)
    system[diag_index] += phi_i
    return np.linalg.solve(system, rhs)


def _inference_loop(
    rng_key: jax.Array,
    kernel,
    initial_state,
    num_samples: int,
):
    """Run a simple jit-scanned inference loop."""

    @jax.jit
    def one_step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, int(num_samples))
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def run_full_nuts(
    setup_result: SetupResult,
    out_dir: Path | str,
    *,
    random_key: int = 0,
    num_warmup: int = 100,
    num_samples: int = 50,
    initial_step_size: float = 1e-2,
    target_acceptance_rate: float = 0.8,
    is_mass_matrix_diagonal: bool = True,
    max_num_doublings: int = 6,
    progress_bar: bool = False,
) -> dict[str, Any]:
    """
    Run BlackJAX NUTS with Stan window adaptation on the full-state target.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_likelihood = _build_full_likelihood(setup_result)
    initial_linear_position = _build_conditional_map_linear_init(
        setup_result, full_likelihood
    )
    logdensity_fn, transform, initial_position = _build_unconstrained_logdensity(
        full_likelihood,
        initial_linear_position=jnp.asarray(initial_linear_position),
    )

    rng_key = jax.random.PRNGKey(int(random_key))
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        initial_step_size=float(initial_step_size),
        target_acceptance_rate=float(target_acceptance_rate),
        progress_bar=bool(progress_bar),
        max_num_doublings=int(max_num_doublings),
    )

    import time

    t_warmup0 = time.perf_counter()
    (adapted_state, adapted_parameters), warmup_info = warmup.run(
        warmup_key, initial_position, num_steps=int(num_warmup)
    )
    warmup_seconds = time.perf_counter() - t_warmup0

    kernel = blackjax.nuts(logdensity_fn, **adapted_parameters).step

    t_sample0 = time.perf_counter()
    states, infos = _inference_loop(
        sample_key, kernel, adapted_state, int(num_samples)
    )
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        (states, infos),
    )
    sample_seconds = time.perf_counter() - t_sample0

    unconstrained_samples = np.asarray(jax.device_get(states.position))
    constrained_samples = np.asarray(
        jax.device_get(jax.vmap(lambda x: transform(x)[0])(states.position))
    )
    logdensity = np.asarray(jax.device_get(states.logdensity))
    acceptance_rate = np.asarray(jax.device_get(infos.acceptance_rate))
    num_integration_steps = np.asarray(jax.device_get(infos.num_integration_steps))
    num_expansions = np.asarray(jax.device_get(infos.num_trajectory_expansions))
    divergences = np.asarray(jax.device_get(infos.is_divergent))
    np.save(out_dir / "samples_unconstrained.npy", unconstrained_samples)
    np.save(out_dir / "samples_theta.npy", constrained_samples)
    np.save(out_dir / "logdensity.npy", logdensity)
    np.save(out_dir / "acceptance_rate.npy", acceptance_rate)

    inverse_mass_matrix = np.asarray(adapted_parameters["inverse_mass_matrix"])
    step_size = float(np.asarray(adapted_parameters["step_size"]))

    summary = {
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "initial_step_size": float(initial_step_size),
        "target_acceptance_rate": float(target_acceptance_rate),
        "is_mass_matrix_diagonal": bool(is_mass_matrix_diagonal),
        "max_num_doublings": int(max_num_doublings),
        "warmup_seconds": warmup_seconds,
        "sample_seconds": sample_seconds,
        "seconds_per_sample": sample_seconds / max(1, int(num_samples)),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "unconstrained_dim": int(initial_position.shape[0]),
        "theta_dim": int(full_likelihood.theta_dim),
        "n_power_params": int(full_likelihood.n_power_params),
        "n_linear_params": int(full_likelihood.n_linear_params),
        "t_shape": [int(x) for x in full_likelihood.t_matrix.shape],
        "used_conditional_map_initialization": True,
        "initial_linear_position_norm": float(np.linalg.norm(initial_linear_position)),
        "adapted_step_size": step_size,
        "inverse_mass_matrix_shape": list(inverse_mass_matrix.shape),
        "mean_logdensity": float(np.mean(logdensity)),
        "std_logdensity": float(np.std(logdensity)),
        "mean_acceptance_rate": float(np.mean(acceptance_rate)),
        "min_acceptance_rate": float(np.min(acceptance_rate)),
        "max_acceptance_rate": float(np.max(acceptance_rate)),
        "mean_num_integration_steps": float(np.mean(num_integration_steps)),
        "max_num_integration_steps": int(np.max(num_integration_steps)),
        "mean_num_trajectory_expansions": float(np.mean(num_expansions)),
        "divergence_count": int(np.sum(divergences)),
        "samples_theta_path": str(out_dir / "samples_theta.npy"),
        "samples_unconstrained_path": str(out_dir / "samples_unconstrained.npy"),
        "logdensity_path": str(out_dir / "logdensity.npy"),
        "acceptance_rate_path": str(out_dir / "acceptance_rate.npy"),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_marginalised_nuts(
    setup_result: SetupResult,
    out_dir: Path | str,
    *,
    random_key: int = 0,
    num_warmup: int = 100,
    num_samples: int = 50,
    initial_step_size: float = 1e-2,
    target_acceptance_rate: float = 0.8,
    is_mass_matrix_diagonal: bool = True,
    max_num_doublings: int = 6,
    progress_bar: bool = False,
    distributed_likelihood: bool = False,
    use_window_adaptation: bool = True,
) -> dict[str, Any]:
    """
    Run BlackJAX NUTS on the analytically marginalized posterior.

    ``distributed_likelihood`` is retained for call-site compatibility but no
    longer changes behavior: the marginalized likelihood always uses the JAXMg
    solver mesh, including on single-GPU runs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    marginal_likelihood = _build_marginalised_likelihood(setup_result)
    logdensity_fn, transform, initial_position = _build_unconstrained_logdensity_marginalised(
        marginal_likelihood
    )

    rng_key = jax.random.PRNGKey(int(random_key))
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    import time

    if use_window_adaptation:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            initial_step_size=float(initial_step_size),
            target_acceptance_rate=float(target_acceptance_rate),
            progress_bar=bool(progress_bar),
            max_num_doublings=int(max_num_doublings),
        )

        t_warmup0 = time.perf_counter()
        (adapted_state, adapted_parameters), warmup_info = warmup.run(
            warmup_key, initial_position, num_steps=int(num_warmup)
        )
        warmup_seconds = time.perf_counter() - t_warmup0
    else:
        if is_mass_matrix_diagonal:
            inverse_mass_matrix = jnp.ones_like(initial_position)
        else:
            inverse_mass_matrix = jnp.eye(initial_position.shape[0], dtype=jnp.float64)
        adapted_parameters = {
            "step_size": float(initial_step_size),
            "inverse_mass_matrix": inverse_mass_matrix,
            "max_num_doublings": int(max_num_doublings),
        }
        algorithm = blackjax.nuts(logdensity_fn, **adapted_parameters)
        t_warmup0 = time.perf_counter()
        adapted_state = algorithm.init(initial_position, rng_key=warmup_key)
        if int(num_warmup) > 0:
            warmup_states, _ = _inference_loop(
                warmup_key, algorithm.step, adapted_state, int(num_warmup)
            )
            adapted_state = jax.tree_util.tree_map(lambda x: x[-1], warmup_states)
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                adapted_state,
            )
        warmup_seconds = time.perf_counter() - t_warmup0
        warmup_info = {
            "step_size": float(initial_step_size),
            "inverse_mass_matrix": np.asarray(inverse_mass_matrix).tolist(),
            "mode": "fixed_step",
        }

    kernel = blackjax.nuts(logdensity_fn, **adapted_parameters).step

    t_sample0 = time.perf_counter()
    states, infos = _inference_loop(
        sample_key, kernel, adapted_state, int(num_samples)
    )
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        (states, infos),
    )
    sample_seconds = time.perf_counter() - t_sample0

    unconstrained_samples = np.asarray(jax.device_get(states.position))
    constrained_samples = np.asarray(
        jax.device_get(jax.vmap(lambda x: transform(x)[0])(states.position))
    )
    logdensity = np.asarray(jax.device_get(states.logdensity))
    acceptance_rate = np.asarray(jax.device_get(infos.acceptance_rate))
    num_integration_steps = np.asarray(jax.device_get(infos.num_integration_steps))
    num_expansions = np.asarray(jax.device_get(infos.num_trajectory_expansions))
    divergences = np.asarray(jax.device_get(infos.is_divergent))

    power_slice = slice(
        marginal_likelihood.power_param_start,
        marginal_likelihood.power_param_start + marginal_likelihood.n_power_params,
    )
    power_linear_samples = np.asarray(
        jax.device_get(
            jax.vmap(
                lambda th: theta_power_to_linear(
                    th[power_slice]
                )
            )(jnp.asarray(constrained_samples))
        )
    )

    np.save(out_dir / "samples_unconstrained.npy", unconstrained_samples)
    np.save(out_dir / "samples_theta.npy", constrained_samples)
    np.save(out_dir / "samples_power_linear.npy", power_linear_samples)
    np.save(out_dir / "logdensity.npy", logdensity)
    np.save(out_dir / "acceptance_rate.npy", acceptance_rate)

    inverse_mass_matrix = np.asarray(adapted_parameters["inverse_mass_matrix"])
    step_size = float(np.asarray(adapted_parameters["step_size"]))

    summary = {
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "initial_step_size": float(initial_step_size),
        "target_acceptance_rate": float(target_acceptance_rate),
        "is_mass_matrix_diagonal": bool(is_mass_matrix_diagonal),
        "max_num_doublings": int(max_num_doublings),
        "warmup_seconds": warmup_seconds,
        "sample_seconds": sample_seconds,
        "seconds_per_sample": sample_seconds / max(1, int(num_samples)),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "theta_dim": int(initial_position.shape[0]),
        "n_power_params": int(marginal_likelihood.n_power_params),
        "t_ninv_t_shape": [int(x) for x in marginal_likelihood.t_ninv_t.shape],
        "used_distributed_likelihood": True,
        "requested_distributed_likelihood": bool(distributed_likelihood),
        "used_window_adaptation": bool(use_window_adaptation),
        "adapted_step_size": step_size,
        "inverse_mass_matrix_shape": list(inverse_mass_matrix.shape),
        "mean_acceptance_rate": float(np.mean(acceptance_rate)),
        "mean_num_integration_steps": float(np.mean(num_integration_steps)),
        "mean_num_trajectory_expansions": float(np.mean(num_expansions)),
        "num_divergences": int(np.sum(divergences)),
        "samples_theta_path": str(out_dir / "samples_theta.npy"),
        "samples_power_linear_path": str(out_dir / "samples_power_linear.npy"),
        "logdensity_path": str(out_dir / "logdensity.npy"),
        "acceptance_rate_path": str(out_dir / "acceptance_rate.npy"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "warmup_info.json").write_text(
        json.dumps(
            {
                "step_size": step_size,
                "inverse_mass_matrix": inverse_mass_matrix.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    return summary
