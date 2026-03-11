"""BlackJAX nested-sampling runner for BayesEoR-JAX."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np
import blackjax
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise, log_weights
from tqdm.auto import tqdm

from .setup import SetupResult

LOGGER = logging.getLogger(__name__)


def run(
    setup_result: SetupResult,
    out_dir: Path | str,
    random_key: int = 0,
    n_live_scale_factor: float = 25.0,
    num_inner_scale_factor: float = 12.0,
    delete_scale_factor: float = 0.1,
    stop_threshold: float = -3.0,
) -> dict[str, Any]:
    """
    Run BlackJAX nested sampling using existing prior/likelihood calls.

    Notes
    -----
    - `logprior_fn` calls `Prior.logprior(theta)`.
    - `loglikelihood_fn` calls `Likelihood.log_likelihood(theta)`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_cfg = setup_result.prior_cfg
    likelihood_cfg = setup_result.likelihood_cfg
    n_dim = int(prior_cfg.bounds.shape[0])

    n_live = max(n_dim + 2, int(np.ceil(n_live_scale_factor * n_dim)))
    num_inner_steps = max(1, int(np.ceil(num_inner_scale_factor * n_dim)))
    num_delete = max(1, int(np.floor(delete_scale_factor * n_live)))
    num_delete = min(num_delete, n_live - 1)

    logprior_fn = prior_cfg.logprior
    loglikelihood_fn = likelihood_cfg.log_likelihood

    ns_algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    rng_key = jax.random.PRNGKey(int(random_key))
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = prior_cfg.sample_from_priors(init_key, n_live)
    state = ns_algo.init(initial_particles)

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

    # Save an Anesthetic-compatible chain file, matching the REACH pipeline output pattern.
    n_dim = int(particles.shape[1])
    param_names = [f"theta_{i}" for i in range(n_dim)]
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
