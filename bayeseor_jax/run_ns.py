"""Nested-sampling runners for BayesEoR-JAX."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import json
import logging
from pathlib import Path
from typing import Any

import blackjax

import numpy as np
from anesthetic import NestedSamples
from blackjax.ns import adaptive as ns_adaptive
from blackjax.ns import nss as ns_nss
from blackjax.ns.utils import finalise, log_weights
from pymultinest.solve import solve
from tqdm.auto import tqdm

from .setup import SetupResult
from .utils import (
    block_until_ready,
    estimate_average_eval_time,
    validate_multinest_output_path,
)

LOGGER = logging.getLogger(__name__)


def run_blackjaxns(
    setup_result: SetupResult,
    out_dir: Path | str,
    random_key: int = 0,
    n_live: int | None = None,
    n_live_scale_factor: float = 25.0,
    num_inner_steps: int | None = None,
    num_inner_scale_factor: float = 3.0,
    num_delete: int | None = None,
    delete_scale_factor: float = 0.025,
    stop_threshold: float = -3.0,
) -> dict[str, Any]:
    """
    Run posterior inference with BlackJAX nested sampling.

    Parameters
    ----------
    setup_result : SetupResult
        Output of ``run_setup`` containing initialized prior/likelihood objects.
    out_dir : Path | str
        Directory where nested-sampling outputs and summary files are written.
    random_key : int, optional
        PRNG seed used to initialize nested sampling.
    n_live : int, optional
        Explicit live-point count. If omitted, computed from
        ``n_live_scale_factor * n_dim``.
    n_live_scale_factor : float, optional
        Live-point count scale factor applied to the inferred parameter count.
    num_inner_steps : int, optional
        Explicit number of inner MCMC steps. If omitted, computed from
        ``num_inner_scale_factor * n_dim``.
    num_inner_scale_factor : float, optional
        Number of inner MCMC steps scale factor applied to parameter count.
    num_delete : int, optional
        Explicit number of live points replaced per NS iteration. If omitted,
        computed from ``delete_scale_factor * n_live``.
    delete_scale_factor : float, optional
        Fraction of live points replaced at each nested-sampling iteration.
    stop_threshold : float, optional
        Termination threshold on ``logZ_live - logZ``.

    Returns
    -------
    dict[str, Any]
        Run summary containing dimensions, sampler settings, termination stats,
        evidence estimates, and output file paths.

    Notes
    -----
    The initial NS state is constructed with a scan-based evaluator to reduce
    peak memory during first likelihood/prior evaluation of live points.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_class = setup_result.prior_class
    likelihood_class = setup_result.likelihood_class
    n_dim = int(prior_class.bounds_log.shape[0])

    n_live = int(n_live) if n_live is not None else int(np.ceil(n_live_scale_factor * n_dim))
    num_inner_steps = (
        int(num_inner_steps)
        if num_inner_steps is not None
        else int(np.ceil(num_inner_scale_factor * n_dim))
    )
    num_delete = (
        max(1, int(num_delete))
        if num_delete is not None
        else max(1, int(np.floor(delete_scale_factor * n_live)))
    )

    ns_algo = blackjax.nss(
        logprior_fn=prior_class.logprior,
        loglikelihood_fn=likelihood_class.log_likelihood,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )
    rng_key = jax.random.PRNGKey(int(random_key))
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = prior_class.sample_from_priors(init_key, n_live)

    def scan_eval(fn):
        def eval_particles(particles):
            _, values = jax.lax.scan(
                lambda _, particle: (None, fn(particle)),
                None,
                particles,
            )
            return values

        return eval_particles

    state = ns_adaptive.init(
        particles=initial_particles,
        logprior_fn=scan_eval(prior_class.logprior),
        loglikelihood_fn=scan_eval(likelihood_class.log_likelihood),
        update_inner_kernel_params_fn=ns_nss.compute_covariance_from_particles,
    )

    @jax.jit
    def one_step(carry, _):
        state_in, key_in = carry
        key_in, subkey = jax.random.split(key_in)
        state_out, dead_info = ns_algo.step(subkey, state_in)
        return (state_out, key_in), dead_info

    dead = []
    n_iter = 0
    with tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not (float(state.logZ_live - state.logZ) < stop_threshold):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            n_iter += 1
            pbar.update(num_delete)

    with jax.default_device(jax.devices("cpu")[0]):
        final_state = finalise(state, dead)
        rng_key, weight_key = jax.random.split(rng_key)
        log_w = log_weights(weight_key, final_state, shape=100)
        logzs = jax.scipy.special.logsumexp(log_w, axis=0)

        particles = np.array(final_state.particles)
        logl = np.array(final_state.loglikelihood)
        logl_birth = np.array(final_state.loglikelihood_birth)

    param_names = [f"theta_{i}" for i in range(int(particles.shape[1]))]
    nested_samples = NestedSamples(
        data=particles,
        logL=logl,
        logL_birth=logl_birth,
        columns=param_names,
        labels={name: name for name in param_names},
    )
    nested_csv_path = out_dir / "nested_sampling_results.csv"
    nested_samples.to_csv(nested_csv_path)

    summary = {
        "sampler": "blackjaxns",
        "n_dim": n_dim,
        "n_live": n_live,
        "num_inner_steps": num_inner_steps,
        "num_delete": num_delete,
        "stop_threshold": float(stop_threshold),
        "iterations_completed": int(n_iter),
        "terminated_by_threshold": bool(float(state.logZ_live - state.logZ) < stop_threshold),
        "logZ_mean": float(np.mean(np.array(logzs))),
        "logZ_std": float(np.std(np.array(logzs))),
        "nested_sampling_results_csv": str(nested_csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Nested sampling summary: %s", json.dumps(summary))
    return summary


def run_multinest(
    setup_result: SetupResult,
    out_dir: Path | str,
    *,
    random_key: int | None = None,
    n_live_points: int | None = None,
    n_live_scale_factor: float = 25.0,
    calc_avg_eval: bool = False,
    avg_iters: int = 10,
    file_root: str = "data-",
    resume: bool | None = None,
    verbose: bool | None = None,
    importance_nested_sampling: bool | None = None,
    multimodal: bool | None = None,
    const_efficiency_mode: bool | None = None,
    evidence_tolerance: float | None = None,
    sampling_efficiency: float | None = None,
    max_iter: int | None = None,
) -> dict[str, Any]:
    """
    Run posterior inference with the Python MultiNest wrapper.

    This implementation is intentionally single-process at the sampler level.
    The expensive operation is still the likelihood evaluation itself, which
    can use multiple GPUs internally through the JAXMg-backed marginalized
    likelihood.

    By default this wrapper mirrors the old BayesEoR MultiNest runner closely:
    aside from the required output basename and resolved live-point count, it
    leaves the remaining MultiNest options unset so the underlying
    ``pymultinest.solve`` defaults apply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_class = setup_result.prior_class
    likelihood_class = setup_result.likelihood_class

    if bool(likelihood_class.intrinsic_noise_fitting):
        raise NotImplementedError(
            "run_multinest currently supports only the standard marginalized "
            "power-spectrum parameterization without intrinsic-noise fitting."
        )

    n_dim = int(prior_class.bounds_log.shape[0])
    n_live_points = (
        int(n_live_points)
        if n_live_points is not None
        else int(np.ceil(n_live_scale_factor * n_dim))
    )
    outputfiles_basename = validate_multinest_output_path(out_dir, file_root=file_root)

    theta_ref = 0.5 * (
        np.asarray(prior_class.lower, dtype=np.float64)
        + np.asarray(prior_class.upper, dtype=np.float64)
    )
    avg_eval_seconds = None
    if calc_avg_eval:
        avg_eval_seconds = estimate_average_eval_time(
            likelihood_class.log_likelihood,
            theta_ref=theta_ref,
            avg_iters=avg_iters,
        )

    def prior_transform(cube):
        return prior_class.transform_unit_cube(cube)

    def loglikelihood(theta):
        value = likelihood_class.log_likelihood(theta)
        return float(np.asarray(block_until_ready(value)))

    solve_kwargs: dict[str, Any] = {
        "LogLikelihood": loglikelihood,
        "Prior": prior_transform,
        "n_dims": n_dim,
        "outputfiles_basename": outputfiles_basename,
        "n_live_points": n_live_points,
    }
    if random_key is not None:
        solve_kwargs["seed"] = int(random_key)
    if resume is not None:
        solve_kwargs["resume"] = bool(resume)
    if verbose is not None:
        solve_kwargs["verbose"] = bool(verbose)
    if importance_nested_sampling is not None:
        solve_kwargs["importance_nested_sampling"] = bool(importance_nested_sampling)
    if multimodal is not None:
        solve_kwargs["multimodal"] = bool(multimodal)
    if const_efficiency_mode is not None:
        solve_kwargs["const_efficiency_mode"] = bool(const_efficiency_mode)
    if evidence_tolerance is not None:
        solve_kwargs["evidence_tolerance"] = float(evidence_tolerance)
    if sampling_efficiency is not None:
        solve_kwargs["sampling_efficiency"] = float(sampling_efficiency)
    if max_iter is not None:
        solve_kwargs["max_iter"] = int(max_iter)

    result = solve(
        **solve_kwargs,
    )

    summary = {
        "sampler": "multinest",
        "n_dim": n_dim,
        "n_live_points": n_live_points,
        "outputfiles_basename": outputfiles_basename,
        "multinest_kwargs_passed": sorted(
            key for key in solve_kwargs.keys()
            if key not in {"LogLikelihood", "Prior", "n_dims"}
        ),
    }
    if avg_eval_seconds is not None:
        summary["avg_eval_seconds"] = float(avg_eval_seconds)
        summary["avg_eval_iters"] = int(avg_iters)
    if random_key is not None:
        summary["random_key"] = int(random_key)
    if resume is not None:
        summary["resume"] = bool(resume)
    if verbose is not None:
        summary["verbose"] = bool(verbose)
    if importance_nested_sampling is not None:
        summary["importance_nested_sampling"] = bool(importance_nested_sampling)
    if multimodal is not None:
        summary["multimodal"] = bool(multimodal)
    if const_efficiency_mode is not None:
        summary["const_efficiency_mode"] = bool(const_efficiency_mode)
    if evidence_tolerance is not None:
        summary["evidence_tolerance"] = float(evidence_tolerance)
    if sampling_efficiency is not None:
        summary["sampling_efficiency"] = float(sampling_efficiency)
    if max_iter is not None:
        summary["max_iter"] = int(max_iter)
    if isinstance(result, dict):
        if "logZ" in result:
            summary["logZ"] = float(result["logZ"])
        if "logZerr" in result:
            summary["logZerr"] = float(result["logZerr"])
        if "samples" in result:
            summary["num_posterior_samples"] = int(np.asarray(result["samples"]).shape[0])
        summary["multinest_result_keys"] = sorted(result.keys())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("MultiNest summary: %s", json.dumps(summary))
    return summary


def run_nestedsampling(
    setup_result: SetupResult,
    out_dir: Path | str,
    *,
    sampler: str = "blackjaxns",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run nested sampling with the requested sampler backend.

    Parameters
    ----------
    sampler : {"blackjaxns", "multinest"}, optional
        Nested-sampling backend selector. Defaults to ``"blackjaxns"``.
    """
    sampler_key = sampler.lower()
    if sampler_key in {"blackjaxns", "blackjax", "blackjax-ns"}:
        return run_blackjaxns(setup_result=setup_result, out_dir=out_dir, **kwargs)
    if sampler_key == "multinest":
        return run_multinest(setup_result=setup_result, out_dir=out_dir, **kwargs)
    raise ValueError(
        "sampler must be one of {'blackjaxns', 'multinest'}, "
        f"got {sampler!r}"
    )
