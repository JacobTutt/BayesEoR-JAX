"""Minimal example script for running BayesEoR-JAX from a config file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.utils import iers

# The historical example observation needs no online Earth-orientation refresh.
iers.conf.auto_download = False

from bayeseor_jax import BayesEoRParser
from bayeseor_jax.analyze import save_analysis
from bayeseor_jax.model.cosmology.transforms import Cosmology


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


def run_example(
    config_path: str = "examples/eor/config.yaml",
    do_sampling: bool = False,
    extra_bayeseor_args: list[str] | None = None,
    nested_sampling_dirname: str = "nested_sampling",
    analysis_label: str = "BayesEoR-JAX",
    prepare_foreground: bool = False,
    expected_ps: float | None = None,
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

    # run_setup already saved the resolved geometry; retain its inferred grid sizes.

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
        expected_ps=expected_ps,
    )
    return setup_result, summary, analysis_summary


def main(default_config="examples/eor/config.yaml", *, prepare_foreground=False,
         label="BayesEoR-JAX", expected_ps=None) -> None:
    """Run one example, with defaults supplied by its independent entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable nested sampling (default is setup-only).",
    )
    parser.add_argument("--prepare-kl", action="store_true", default=prepare_foreground, help="Build the configured foreground KL basis during setup.")
    parser.add_argument("--expected-ps", type=float, default=expected_ps, help="Optional injected flat P(k), in mK^2 Mpc^3, for reference curves.")
    parser.add_argument(
        "--nested-sampling-dirname",
        default="nested_sampling",
        help="Directory name, under output_dir, for sampler products.",
    )
    parser.add_argument(
        "--analysis-label",
        default=label,
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
        expected_ps=args.expected_ps,
    )


def injected_power_spectrum(metadata):
    """Return P = sigma_pixel² V_voxel in mK² Mpc³ at the central frequency.

    Parameters
    ----------
    metadata : dict
        The supplied simulation.json, including frequencies, simulation
        HEALPix resolution and brightness-temperature standard deviation.

    Returns
    -------
    float
        Ensemble dimensional power of the injected independent-pixel field,
        not a fit or the realised power of this particular sky draw.
    """
    frequencies = np.asarray(metadata["frequency_hz"])
    cosmology = Cosmology()
    z = cosmology.f2z(frequencies.mean())
    solid_angle = 4*np.pi/(12*metadata["signal_nside"]**2)
    volume = solid_angle * (frequencies[1]-frequencies[0]) * cosmology.inst_to_cosmo_vol(z)
    return float(metadata["signal_pixel_std_mk"]**2 * volume)


if __name__ == "__main__":
    main()
