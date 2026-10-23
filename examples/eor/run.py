"""Run cosmological signal only: python -m examples.eor.run --sample."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.utils import iers

# The historical example observation needs no online Earth-orientation refresh.
iers.conf.auto_download = False

from bayeseor_jax import BayesEoRParser
from bayeseor_jax.analyze import save_analysis


def run_example(
    config_path: str = str(Path(__file__).with_name("config.yaml")),
    do_sampling: bool = False,
    extra_bayeseor_args: list[str] | None = None,
    nested_sampling_dirname: str = "nested_sampling",
    analysis_label: str = "Cosmological signal only",
    prepare_foreground: bool = False,
):
    """
    Run setup (and optionally nested sampling) from a YAML config.

    Additional BayesEoR parser arguments can be supplied through
    `extra_bayeseor_args`, for example output and matrix-cache directories.
    Set prepare_foreground to construct the KL basis from this same configuration
    during setup. Setup loads noiseless data and adds the configured seeded noise.
    Every run saves the actual noisy input and noise separately for reproducibility.
    """
    from bayeseor_jax import run_setup, run_nestedsampling

    parser = BayesEoRParser()
    args_list = ["--config", config_path]
    if do_sampling:
        args_list.append("--run")
    if extra_bayeseor_args:
        args_list.extend(extra_bayeseor_args)
    args = parser.parse_args(args_list)

    out_dir = Path(args.output_dir)
    ns_out_dir = out_dir / nested_sampling_dirname
    if args.run and ns_out_dir.exists() and any(ns_out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Sampler output already exists: {ns_out_dir}; use --resume or a fresh output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse stale matrix/KL caches when a user changes the example settings.
    manifest = _json_ready(asdict(args))
    for key in ("run", "resume", "quiet"):
        manifest.pop(key, None)
    manifest_path = out_dir / "run_config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Example configuration changed; choose fresh output, matrix and KL paths")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    setup_result = run_setup(args, build_foreground_kl=prepare_foreground and not args.resume)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "visibilities_noisy.npy", np.asarray(setup_result.vis_dict["vis_noisy"]))
    np.save(data_dir / "noise.npy", np.asarray(setup_result.vis_dict["noise"]))

    if not args.run:
        print("Setup complete. Sampling disabled.")
        return setup_result

    summary = run_nestedsampling(
        setup_result=setup_result,
        out_dir=ns_out_dir,
        random_key=args.random_key,
        n_live_scale_factor=args.n_live_scale_factor,
        resume=args.resume,
        center_likelihood=True,
    )

    print("Run summary:")
    print(json.dumps(summary, indent=2))

    if summary["sampler"] == "analytic":
        # There is no power-spectrum posterior to plot for a foreground-only model.
        return setup_result, summary, None

    analysis_summary = save_analysis(
        sampler_dir=ns_out_dir,
        output_dir=Path(config_path).resolve().parent / "plots",
        label=analysis_label,
        metadata_dir=out_dir,
        expected_ps=np.load(Path(__file__).parent / "data/injected_power_spectrum.npy", allow_pickle=False).item(),
    )
    return setup_result, summary, analysis_summary


def main() -> None:
    """Run this example using its own configuration and supplied reference spectrum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable nested sampling (default is setup-only).",
    )
    parser.add_argument("--prepare-kl", action="store_true", default=False, help="Build the configured foreground KL basis during setup.")
    parser.add_argument(
        "--nested-sampling-dirname",
        default="nested_sampling",
        help="Directory name, under output_dir, for sampler products.",
    )
    parser.add_argument(
        "--analysis-label",
        default="Cosmological signal only",
        help="Label used in the analysis plots.",
    )
    args, extra_bayeseor_args = parser.parse_known_args()
    run_example(
        config_path=args.config,
        do_sampling=args.sample,
        extra_bayeseor_args=extra_bayeseor_args,
        nested_sampling_dirname=args.nested_sampling_dirname,
        analysis_label=args.analysis_label,
        prepare_foreground=args.prepare_kl,
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
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
