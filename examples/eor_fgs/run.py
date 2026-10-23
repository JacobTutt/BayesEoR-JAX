"""Run joint inference: python -m examples.eor_fgs.run --sample."""
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from ..run_analysis_example import injected_power_spectrum, main


if __name__ == "__main__":
    folder = Path(__file__).parent
    metadata = json.loads((folder / "data/simulation.json").read_text())
    main(str(folder / "config.yaml"), prepare_foreground=True, label="Cosmological signal + foregrounds",
         expected_ps=injected_power_spectrum(metadata))
