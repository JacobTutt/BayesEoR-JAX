"""Run the signal-only example: python -m examples.eor.run --sample."""
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from ..plot_recoveries import injected_power_spectrum
from ..run_analysis_example import main


if __name__ == "__main__":
    folder = Path(__file__).parent
    metadata = json.loads((folder / "data/simulation.json").read_text())
    main(str(folder / "config.yaml"), label="21-cm only",
         expected_ps=injected_power_spectrum(metadata))
