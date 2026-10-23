"""Run the two local examples through construction, inference and saved plots.

From the repository root: python -m test_examples.run_laptop_examples
Use --setup-only to build operators and likelihood products without MultiNest.
Both physical models use the shared nside in their YAML configuration.
"""

import argparse
import json
import os
from pathlib import Path

# These examples must not require GPU libraries or an HPC installation.
os.environ["JAX_PLATFORMS"] = "cpu"

from .plot_recoveries import injected_power_spectrum, export_recoveries
from .run_analysis_example import run_example


def main():
    """Run both configurations and retain data, matrices, chains and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true", help="Do not run nested sampling or posterior analysis.")
    args = parser.parse_args()
    metadata = json.loads(Path("test_examples/instrument/simulation.json").read_text())
    expected_ps = injected_power_spectrum(metadata)
    for name, label in (("eor", "21-cm only"), ("eor_fgs", "21-cm + foregrounds")):
        run_example(
            config_path=f"test_examples/{name}/config.yaml",
            do_sampling=not args.setup_only, prepare_foreground=name == "eor_fgs",
            analysis_label=label, expected_ps=expected_ps,
        )
    if not args.setup_only:
        export_recoveries(Path("outputs/laptop_recovery_plots_nside128"))


if __name__ == "__main__":
    main()
