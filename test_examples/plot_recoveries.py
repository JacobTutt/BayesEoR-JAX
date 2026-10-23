"""Plot both saved laptop recoveries against their injected white-noise spectrum.

Run from the repository root. This reads completed MultiNest samples; it does
not rerun inference, change posterior weights or repair evidence estimates.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from bayeseor_jax import DataContainer
from bayeseor_jax.analyze import plot_power_spectra_and_posteriors
from bayeseor_jax.model.cosmology.transforms import Cosmology


def injected_power_spectrum(metadata):
    """Return P = sigma_pixel² V_voxel in mK² Mpc³ at the central frequency.

    Parameters
    ----------
    metadata : dict
        The generator's simulation.json, including frequencies, simulation
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


def export_recoveries(output):
    """Save individual and overlaid recovery figures as PNG and PDF.

    Parameters
    ----------
    output : pathlib.Path
        Destination directory. Existing figures with these names are replaced.

    Returns
    -------
    list of pathlib.Path
        Paths to the three PNG previews; matching vector PDFs are also saved.
    """
    metadata = json.loads(Path("test_examples/instrument/simulation.json").read_text())
    power = injected_power_spectrum(metadata)
    roots = [Path("outputs/laptop_eor"), Path("outputs/laptop_eor_fgs")]
    labels = ["21-cm only", "21-cm + foregrounds"]
    summaries = [json.loads((root/"nested_sampling/summary.json").read_text()) for root in roots]
    output.mkdir(parents=True, exist_ok=True)
    previews = []
    for name, indices in (("21cm_only", [0]), ("21cm_and_foregrounds", [1]),
                          ("comparison", [0, 1])):
        data = DataContainer(
            [roots[i]/"nested_sampling" for i in indices],
            metadata_dirs=[roots[i] for i in indices],
            labels=[labels[i] for i in indices], expected_ps=power,
        )
        fig, (power_ax, posterior_axes) = plot_power_spectra_and_posteriors(
            data, plot_priors=True, bins=65, colors=[("#153b5c", "#b43c3c")[i] for i in indices],
        )
        fig.set_size_inches(11, 4.8)
        power_ax.get_subplotspec().get_gridspec().update(top=.88, bottom=.18)
        fig.suptitle("Posterior recovery — " + (labels[indices[0]] if len(indices) == 1 else "comparison"),
                     fontsize=13, y=.97)
        handles, legend_labels = power_ax.get_legend_handles_labels()
        power_ax.legend(handles, ["Injected spectrum" if label == "Reference" else label
                                  for label in legend_labels], loc="lower right", frameon=False)
        # Preserve the stored prior support, with consistent axes across figures.
        power_ax.set_ylim(0.1, 1e6)
        power_ax.xaxis.set_major_locator(FixedLocator(data.k_vals[0]))
        power_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        power_ax.xaxis.set_minor_formatter(NullFormatter())
        for ax in posterior_axes:
            ax.tick_params(axis="x", labelsize=9)
        note = "Central 68% credible intervals; dashed: injected ensemble spectrum; grey: prior support."
        if any(not np.isfinite(summaries[i].get("logZerr", np.nan)) for i in indices):
            note += "\nDiagnostic only: at least one run has a non-finite evidence uncertainty."
        fig.text(.5, .04, note, ha="center", va="center", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output/f"{name}.{extension}", dpi=250, bbox_inches="tight")
        plt.close(fig)
        previews.append(output/f"{name}.png")
    (output/"README.txt").write_text(
        f"Injected dimensional power: {power:.8g} mK^2 Mpc^3.\n"
        "Figures use the completed 15-time, 15-channel, 30-signed-baseline runs.\n"
        "Both model fields are 10 degrees, with Nside=64 inference grids.\n"
        f"Simulation footprint: {metadata.get('mask_definition', 'Independent angular selection at each resolution')}.\n"
        "Plot generation did not modify samples or weights, or rerun inference.\n"
        f"Reported evidence uncertainties: {[summary.get('logZerr') for summary in summaries]}.\n"
        "These plots alone do not establish validated recovery or reliable evidence.\n"
    )
    return previews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    for path in export_recoveries(arguments.output):
        print(path)
