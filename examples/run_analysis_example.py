"""Minimal example script for running BayesEoR-JAX from a config file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax

from bayeseor_jax import * 


def run_example(
    config_path: str = "test_data/eor/config.yaml",
    do_sampling: bool = False,
    sampler: str | None = None,
):
    """
    Run setup (and optionally nested sampling) from a YAML config.

    Edit `config_path` and `do_sampling` in `main()` for quick local tests.
    """
    parser = BayesEoRParser()
    args_list = ["--config", config_path]
    if sampler is not None:
        args_list.extend(["--sampler", sampler])
    if do_sampling:
        args_list.append("--run")
    args = parser.parse_args(args_list)

    args_dict = dict(vars(args))
    setup_result = run_setup(**args_dict)

    if not args.run:
        print("Setup complete. Sampling disabled.")
        return setup_result

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_log_files(parser, args, out_dir=out_dir)

    ns_out_dir = out_dir / "nested_sampling"
    sampler_key = args.sampler.lower()
    ns_kwargs = {
        "sampler": args.sampler,
        "random_key": args.random_key,
        "n_live_scale_factor": args.n_live_scale_factor,
    }
    if sampler_key in {"blackjaxns", "blackjax", "blackjax-ns"}:
        ns_kwargs.update(
            {
                "num_inner_scale_factor": args.num_inner_scale_factor,
                "delete_scale_factor": args.delete_scale_factor,
                "stop_threshold": args.stop_threshold,
            }
        )
    summary = run_nestedsampling(
        setup_result=setup_result,
        out_dir=ns_out_dir,
        **ns_kwargs,
    )

    print("Run summary:")
    print(json.dumps(summary, indent=2))
    return setup_result, summary


def main() -> None:
    """Small runnable example with optional CLI overrides."""
    if os.environ.get("SLURM_PROCID") is not None and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        # Multi-node GH200 runs on Isambard currently need PXN disabled for the
        # distributed collectives used by JAX/XLA to avoid NCCL failures.
        os.environ.setdefault("NCCL_PXN_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_PXN_LEVEL", "0")
        jax.distributed.initialize()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="test_data/eor/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable nested sampling (default is setup-only).",
    )
    parser.add_argument(
        "--sampler",
        choices=("multinest", "blackjaxns"),
        default=None,
        help="Nested-sampling backend. Defaults to the config/package default.",
    )
    args = parser.parse_args()
    run_example(
        config_path=args.config,
        do_sampling=args.sample,
        sampler=args.sampler,
    )


if __name__ == "__main__":
    main()
