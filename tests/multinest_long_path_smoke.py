"""Small real-MultiNest smoke test for long output paths and resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bayeseor_jax.analyze.analyze import DataContainer
from bayeseor_jax.run_ns import run_multinest


class _Prior:
    lower = np.asarray([-5.0, -5.0], dtype=np.float64)
    upper = np.asarray([5.0, 5.0], dtype=np.float64)
    bounds_log = np.column_stack((lower, upper))

    @classmethod
    def transform_unit_cube(cls, cube):
        cube = np.asarray(cube, dtype=np.float64)
        return cls.lower + cube * (cls.upper - cls.lower)


class _Likelihood:
    intrinsic_noise_fitting = False

    @staticmethod
    def log_likelihood(theta):
        theta = np.asarray(theta, dtype=np.float64)
        return -0.5 * np.dot(theta, theta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    expected_longest_path = output_dir / "data-post_equal_weights.dat"
    if len(expected_longest_path.as_posix()) <= 100:
        raise AssertionError(
            f"Smoke output path is not long enough: {expected_longest_path}"
        )

    setup_result = SimpleNamespace(
        prior_class=_Prior(),
        likelihood_class=_Likelihood(),
    )
    first = run_multinest(
        setup_result,
        output_dir,
        random_key=13,
        n_live_points=20,
        resume=False,
        max_iter=100,
    )
    if not first["used_short_multinest_path"]:
        raise AssertionError("Long MultiNest path did not use a short symlink.")
    if len(f"{first['outputfiles_basename']}post_equal_weights.dat") > 100:
        raise AssertionError("The basename passed to MultiNest remained too long.")
    if os.path.lexists(first["multinest_short_path_link"]):
        raise AssertionError("Short MultiNest symlink was not cleaned up.")
    if not any(output_dir.glob("data-*")):
        raise AssertionError("MultiNest outputs did not land in the real directory.")

    second = run_multinest(
        setup_result,
        output_dir,
        random_key=13,
        n_live_points=20,
        resume=True,
        max_iter=200,
    )
    if not second.get("resume_checkpoint_files_found"):
        raise AssertionError("Resume did not discover a MultiNest checkpoint.")
    if os.path.lexists(second["multinest_short_path_link"]):
        raise AssertionError("Short MultiNest symlink remained after resume.")

    recorded = json.loads((output_dir / "summary.json").read_text())
    if recorded["logical_outputfiles_basename"] != (
        output_dir / "data-"
    ).as_posix():
        raise AssertionError("Summary did not retain the real logical basename.")
    chain_path = DataContainer._resolve_multinest_chain_path(output_dir)
    if chain_path != output_dir / "data-.txt":
        raise AssertionError(f"Analysis resolved the wrong chain: {chain_path}")
    print(json.dumps(recorded, indent=2))
    print("LONG_PATH_MULTINEST_SMOKE_PASSED")


if __name__ == "__main__":
    main()
