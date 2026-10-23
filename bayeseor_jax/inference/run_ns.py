"""Sample power spectra with MultiNest or evaluate parameter-free models exactly."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax
import numpy as np
if TYPE_CHECKING:
    from ..setup import SetupResult
from .multinest_paths import (
    MultiNestPathManager,
    _cleanup_multinest_path,
    _prepare_multinest_path,
)

LOGGER = logging.getLogger(__name__)


def estimate_average_eval_time(
    likelihood_fn,
    theta_ref: np.ndarray,
    avg_iters: int,
) -> float:
    """
    Estimate the average scalar likelihood evaluation time.

    Parameters
    ----------
    likelihood_fn : callable
        Scalar log-likelihood callback.
    theta_ref : numpy.ndarray
        Reference parameter vector used for repeated timing calls.
    avg_iters : int
        Number of repeated evaluations.
    """
    start = time.perf_counter()
    for _ in range(int(avg_iters)):
        value = likelihood_fn(theta_ref)
        value = jax.block_until_ready(value)
        if not np.isfinite(float(np.asarray(value))):
            raise ValueError("Non-finite likelihood value encountered during timing.")
    return (time.perf_counter() - start) / max(1, int(avg_iters))


def _multinest_resume_checkpoints(outputfiles_basename: str) -> list[Path]:
    """Return candidate MultiNest checkpoint files for an output basename."""
    return [
        Path(f"{outputfiles_basename}resume.dat"),
        Path(f"{outputfiles_basename}phys_live.points"),
        Path(f"{outputfiles_basename}.resume"),
    ]


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
    verbose: bool = False,
    importance_nested_sampling: bool | None = None,
    multimodal: bool | None = None,
    const_efficiency_mode: bool | None = None,
    evidence_tolerance: float | None = None,
    sampling_efficiency: float | None = None,
    max_iter: int | None = None,
    print_rate: int = 100,
    center_likelihood: bool = False,
) -> dict[str, Any]:
    """
    Run MultiNest, or evaluate a parameter-free model analytically.

    A model with no sampled hyperparameters needs no nested sampling. Its
    conditional MAP coefficients and marginalised log-evidence target are
    evaluated in one solve and saved as ``map_coefficients.npy`` and
    ``summary.json``. The same fixed data/noise normalisation is omitted as
    in the sampled likelihood, so same-dataset model comparisons are unchanged.

    This implementation is intentionally single-process at the sampler level.
    The expensive operation is still the likelihood evaluation itself, which
    can use multiple GPUs internally through the JAXMg-backed marginalized
    likelihood.

    By default this wrapper mirrors the old BayesEoR MultiNest runner closely:
    aside from the required output basename and resolved live-point count, it
    leaves the remaining MultiNest options unset so the underlying
    ``pymultinest.solve`` defaults apply.

    Set ``center_likelihood=True`` to subtract the log likelihood at the
    prior midpoint before passing values to MultiNest. This avoids loss of
    precision in its evidence-error estimate for bright foregrounds. The
    offset is restored in the returned ``logZ`` and saved in summary.json;
    native MultiNest files retain centred values. Resuming requires the same
    centring setting and uses the recorded offset unchanged.
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
    if n_dim == 0:
        if resume:
            raise ValueError("An analytic foreground-only evaluation has no sampler checkpoint to resume")
        from .maxaposteriori import MaximumAPosteriori

        # Use the same compressed terms and prior as the marginalised likelihood.
        reconstruction = MaximumAPosteriori(
            t_ninv_t=likelihood_class.t_ninv_t, dbar=likelihood_class.dbar,
            d_ninv_d=likelihood_class.d_ninv_d, n_data=likelihood_class.n_data,
            phi_inv_class=setup_result.phi_inverse_class,
            mesh=likelihood_class.mesh, jaxmg_tile_size=likelihood_class.jaxmg_tile_size,
        )
        coefficients, log_evidence = reconstruction.map_estimate(np.empty(0))
        if not np.isfinite(log_evidence) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Non-finite analytic evidence or MAP coefficients")
        coefficients_path = out_dir / "map_coefficients.npy"
        np.save(coefficients_path, coefficients)
        summary = {
            "sampler": "analytic", "n_dim": 0,
            "model": setup_result.model,
            "logZ": log_evidence, "logZerr": 0.0,
            "evidence_normalisation": "fixed data/noise constants omitted",
            "map_coefficients_path": str(coefficients_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    # Foreground-only evaluation does not require a native MultiNest installation.
    from pymultinest.solve import solve

    n_live_points = (
        int(n_live_points)
        if n_live_points is not None
        else int(np.ceil(n_live_scale_factor * n_dim))
    )
    path_manager = MultiNestPathManager(out_dir, file_root=file_root)
    outputfiles_basename = _prepare_multinest_path(path_manager)
    resume_checkpoints = _multinest_resume_checkpoints(outputfiles_basename)
    existing_resume_files = [path for path in resume_checkpoints if path.exists()]
    if bool(resume) and not existing_resume_files:
        expected = ", ".join(path.name for path in resume_checkpoints)
        _cleanup_multinest_path(path_manager)
        raise FileNotFoundError(
            "Requested MultiNest resume, but no existing checkpoint files were "
            f"found in {out_dir}. Expected at least one of: {expected}"
        )

    theta_ref = 0.5 * (
        np.asarray(prior_class.lower, dtype=np.float64)
        + np.asarray(prior_class.upper, dtype=np.float64)
    )
    offset_path = out_dir / "likelihood_offset.json"
    if resume:
        offset_record = json.loads(offset_path.read_text()) if offset_path.exists() else {
            "center_likelihood": False, "log_likelihood_offset": 0.0,
        }
        if bool(offset_record["center_likelihood"]) != bool(center_likelihood):
            _cleanup_multinest_path(path_manager)
            raise ValueError("Resume must use the original likelihood-centring setting")
        log_likelihood_offset = float(offset_record["log_likelihood_offset"])
    else:
        log_likelihood_offset = (
            float(np.asarray(likelihood_class.log_likelihood(theta_ref)))
            if center_likelihood else 0.0
        )
    if not np.isfinite(log_likelihood_offset):
        _cleanup_multinest_path(path_manager)
        raise ValueError("Likelihood-centring offset must be finite")
    if not resume:
        offset_path.write_text(json.dumps({
            "center_likelihood": bool(center_likelihood),
            "log_likelihood_offset": log_likelihood_offset,
        }, indent=2) + "\n")
    avg_eval_seconds = None
    if calc_avg_eval:
        avg_eval_seconds = estimate_average_eval_time(
            likelihood_class.log_likelihood,
            theta_ref=theta_ref,
            avg_iters=avg_iters,
        )

    instantiation_time = time.time()
    loglikelihood_count = 0

    def prior_transform(cube):
        return prior_class.transform_unit_cube(cube)

    def loglikelihood(theta):
        nonlocal loglikelihood_count
        loglikelihood_count += 1
        start_call = time.time()
        value = likelihood_class.log_likelihood(theta)
        value = float(np.asarray(jax.block_until_ready(value)))
        if print_rate > 0 and loglikelihood_count % int(print_rate) == 0:
            print("count", loglikelihood_count, flush=True)
            print(
                "0 : Time since class instantiation: {}".format(
                    time.time() - instantiation_time
                ),
                flush=True,
            )
            print(
                "0 : Time for this likelihood call: {}".format(
                    time.time() - start_call
                ),
                flush=True,
            )
        return value - log_likelihood_offset

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
    _cleanup_multinest_path(path_manager)

    summary = {
        "sampler": "multinest",
        "model": setup_result.model,
        "n_dim": n_dim,
        "n_live_points": n_live_points,
        "log_likelihood_offset": log_likelihood_offset,
        "center_likelihood": bool(center_likelihood),
        "evidence_normalisation": "fixed data/noise constants omitted",
        "outputfiles_basename": outputfiles_basename,
        "logical_outputfiles_basename": path_manager.logical_outputfiles_basename,
        "used_short_multinest_path": path_manager.using_short_path,
        "multinest_short_path_link": (
            path_manager.short_link.as_posix()
            if path_manager.using_short_path
            else None
        ),
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
        summary["resume_checkpoint_files_found"] = [
            path.name for path in existing_resume_files
        ]
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
    summary["print_rate"] = int(print_rate)
    summary["loglikelihood_calls"] = int(loglikelihood_count)
    if isinstance(result, dict):
        if "logZ" in result:
            summary["logZ_centered"] = float(result["logZ"])
            summary["logZ"] = float(result["logZ"]) + log_likelihood_offset
        if "logZerr" in result:
            summary["logZerr"] = float(result["logZerr"])
        if "samples" in result:
            summary["num_posterior_samples"] = int(np.asarray(result["samples"]).shape[0])
        summary["multinest_result_keys"] = sorted(result.keys())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("MultiNest summary: %s", json.dumps(summary))
    return summary


run_nestedsampling = run_multinest
