"""Minimal example script for running BayesEoR-JAX from a config file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils import iers

# The historical example observation needs no online Earth-orientation refresh.
iers.conf.auto_download = False

from bayeseor_jax import BayesEoRParser, DataContainer
from bayeseor_jax.analyze import (
    plot_power_spectra, plot_posteriors, plot_power_spectra_and_posteriors,
)


def _json_ready(value):
    """Convert NumPy/JAX-like values into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_output_analysis(
    sampler_dir: Path | str,
    output_dir: Path | str | None = None,
    label: str = "BayesEoR-JAX",
    metadata_dir: Path | str | None = None,
    expected_ps: float | None = None,
) -> dict:
    """
    Read sampler output and optional separate setup metadata, then save summaries and plots.

    expected_ps optionally supplies a flat injected dimensional power spectrum
    in mK² Mpc³ for the reference curves; it does not alter posterior samples.
    """
    sampler_dir = Path(sampler_dir)
    output_dir = Path(output_dir) if output_dir is not None else sampler_dir.parent
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    data = DataContainer(
        [sampler_dir],
        labels=[label],
        metadata_dirs=metadata_dir,
        expected_ps=expected_ps,
    )
    summary = {
        "sampler": "multinest",
        "sampler_dir": str(sampler_dir),
        "k_vals": _json_ready(data.k_vals[0]),
        "averages": _json_ready(data.avgs[0]),
        "medians": _json_ready(data.medians[0]),
        "credible_intervals": _json_ready(data.cred_intervals[0]),
    }
    if hasattr(data, "uplims"):
        summary["upper_limits"] = _json_ready(data.uplims[0])

    summary_path = analysis_dir / "posterior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, _ = plot_power_spectra(data, plot_priors=True)
    fig.savefig(analysis_dir / "power_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_posteriors(data, plot_priors=True)
    fig.savefig(analysis_dir / "posteriors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_power_spectra_and_posteriors(data, plot_priors=True)
    fig.savefig(analysis_dir / "power_spectrum_and_posteriors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Analysis summary:")
    print(json.dumps({"analysis_dir": str(analysis_dir), **summary}, indent=2))
    return summary


def run_example(
    config_path: str = "test_examples/eor/config.yaml",
    do_sampling: bool = False,
    extra_bayeseor_args: list[str] | None = None,
    nested_sampling_dirname: str = "nested_sampling",
    analysis_label: str = "BayesEoR-JAX",
):
    """
    Run setup (and optionally nested sampling) from a YAML config.

    Additional BayesEoR parser arguments can be supplied through
    `extra_bayeseor_args`, for example output and matrix-cache directories.
    """
    from bayeseor_jax import run_setup, run_nestedsampling

    parser = BayesEoRParser()
    args_list = ["--config", config_path]
    if do_sampling:
        args_list.append("--run")
    if extra_bayeseor_args:
        args_list.extend(extra_bayeseor_args)
    args = parser.parse_args(args_list)

    setup_result = run_setup(args)

    if not args.run:
        print("Setup complete. Sampling disabled.")
        return setup_result

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # run_setup already saved the resolved geometry; retain its inferred grid sizes.

    ns_out_dir = out_dir / nested_sampling_dirname
    summary = run_nestedsampling(
        setup_result=setup_result,
        out_dir=ns_out_dir,
        random_key=args.random_key,
        n_live_scale_factor=args.n_live_scale_factor,
        resume=args.resume,
    )

    print("Run summary:")
    print(json.dumps(summary, indent=2))

    if summary["sampler"] == "analytic":
        # There is no power-spectrum posterior to plot for a foreground-only model.
        return setup_result, summary, None

    analysis_summary = run_output_analysis(
        sampler_dir=ns_out_dir,
        output_dir=out_dir,
        label=analysis_label,
        metadata_dir=out_dir,
    )
    return setup_result, summary, analysis_summary


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
        default="test_examples/eor/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable nested sampling (default is setup-only).",
    )
    parser.add_argument(
        "--nested-sampling-dirname",
        default="nested_sampling",
        help="Directory name, under output_dir, for sampler products.",
    )
    parser.add_argument(
        "--analysis-label",
        default="BayesEoR-JAX",
        help="Label used in the analysis plots.",
    )
    args, extra_bayeseor_args = parser.parse_known_args()
    run_example(
        config_path=args.config,
        do_sampling=args.sample,
        extra_bayeseor_args=extra_bayeseor_args,
        nested_sampling_dirname=args.nested_sampling_dirname,
        analysis_label=args.analysis_label,
    )


if __name__ == "__main__":
    main()
