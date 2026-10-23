# Examples

These two examples run locally on CPU: first infer the cosmological power
spectrum without foregrounds, then add foreground emission and construct its
KL representation before joint inference. Run the commands from the repository
root after following [installation](installation.md). Nested sampling requires
the native MultiNest library as well as PyMultiNest; the supplied Conda
environment installs both.

To run both examples, including matrix construction, inference and plots:

```bash
python -m test_examples.run_laptop_examples
```

Use `--setup-only` to load the data, add noise and build the matrices without
sampling. This entry point selects CPU JAX and runs the standard pipeline; it
does not use the separate-resolution diagnostic scripts.

## Observation and inputs

Both examples share the same HERA-like instrument and noise prescription:

| Setting | Value |
| --- | --- |
| Array | 37 antennas on a 14.6-m hexagonal grid |
| Baselines | All 30 signed baseline groups shorter than 40 m |
| Primary beam | 14.2-m Airy power beam |
| Frequency channels | 15 consecutive channels, 158.398–159.766 MHz |
| Channel spacing | 97.65625 kHz |
| Integrations | 15, separated by 21.51137 s |
| Foreground and cosmological fields | 10° diameter, selected over the drift scan |
| Inference image grid | HEALPix `Nside=128` for both components |
| Boundary-selection grid | `footprint_nside=64`, matching the saved simulation |
| Complex noise RMS | 0.006464 mK sr |

The times and frequencies are the central subsets of the 47-integration,
39-channel observation. There are **6,750 stored complex entries**, including
conjugate baselines, corresponding to **3,375 independent complex measurements**.
The conjugates do not provide additional independent data.

The cosmological model uses an automatically sized 11 × 11 transverse Fourier
grid and 14 nonzero line-of-sight modes: **1,680 marginalised coefficients**
and **six sampled power-spectrum amplitudes**. The joint example adds
**179 KL modes**, reduced from 1,248 foreground coefficients using `f=0.1`.

`nside` remains a single shared model resolution. The optional
`footprint_nside` controls only where the field boundary is selected: every
selected parent contributes all four of its `Nside=128` children. This avoids
changing the sky coverage when refining the model. It is not a separate
foreground or cosmological resolution setting.

The checked-in inputs contain only two noiseless visibility vectors and shared
baseline/redundancy metadata. Setup generates reproducible thermal noise using
the same seed for both examples. No full-sky maps, model matrices or sampler
outputs are stored in Git.

## 1. Cosmological signal only

The input is an independent-pixel, zero-mean Gaussian 21-cm brightness field
with a pixel standard deviation of 6.4 mK, simulated at `Nside=256`.
The time-union footprint is selected on a reference grid at `Nside=64`.
The simulation includes all finer pixels belonging to those selected coarse
pixels before integration through the beam and full three-dimensional fringe
response. It does not select a new circular boundary at the simulation resolution.

```bash
export JAX_PLATFORMS=cpu
python test_examples/run_analysis_example.py \
  --config test_examples/eor/config.yaml --sample
```

Omit `--sample` to build the matrices and likelihood without running MultiNest.
Outputs are written to `outputs/laptop_eor_nside128/`.

## 2. Cosmological signal and foregrounds

This dataset adds GSM2008 diffuse emission and a statistical extragalactic/
catalogued bright-source population within the same 10° field. The local
GSM input is sampled at `Nside=128`; point sources retain their individual
positions. Both inherit the same `Nside=64` footprint: every selected coarse
pixel contributes its four `Nside=128` children, and sources are included only
when they lie inside a selected coarse pixel. Both inference components use
`Nside=128` over this same footprint. Matching the footprint does not by itself
guarantee exact representation of subpixel point-source structure.

First build the prior- and noise-weighted foreground operator and reduce it
using native CPU JAX. Then run joint inference:

```bash
export JAX_PLATFORMS=cpu
python test_examples/run_analysis_example.py \
  --config test_examples/eor_fgs/config.yaml --prepare-kl --sample
```

The reduced operator is saved under `outputs/laptop_eor_fgs_nside128/kl/`; the
configuration uses it with a standard-normal foreground prior. Preparation
uses the same physical prior widths and noise covariance as inference.
The joint model marginalises 1,859 coefficients while sampling the six
power-spectrum amplitudes.

For a separate KL-preparation step, run
`python test_examples/eor_fgs/prepare_kl.py --config test_examples/eor_fgs/config.yaml`
and omit `--prepare-kl` from the inference command. Both steps read the same
configuration, including the shared image resolution and footprint.

These are workflow examples with 60 live points, not validated scientific
constraints. Both simulated components are restricted to 10°: these inputs
do not demonstrate recovery from unmodelled emission outside the field or
from the horizon. The reduced bandwidth also probes different bins from
the full observing band.

## Results and rerunning

Each run writes:

- `run_config.json` and `args.json`: requested and resolved settings.
- `data/`: the actual noisy visibility vector and added noise realisation.
- `matrices/`: forward operators and marginal-likelihood products.
- `nested_sampling/`: posterior samples, evidence and run summary.
- `analysis/`: posterior summaries and power-spectrum/posterior figures.

The joint example additionally writes the weighted foreground operator, KL
spectrum and truncation diagnostics. The combined entry point saves overlaid
PNG/PDF figures under `outputs/laptop_recovery_plots_nside128/`.

The example runner centres the log likelihood before passing it to MultiNest
to avoid numerical loss of precision in the evidence uncertainty. The offset
is recorded in `nested_sampling/likelihood_offset.json` and `summary.json`.
The summary's `logZ` and the analysis loader restore the offset; native
MultiNest text files retain centred values. Fixed data/noise normalisation
constants remain omitted, as in the underlying likelihood.
To compare both completed posteriors with the injected ensemble spectrum:

```bash
python test_examples/plot_recoveries.py --output outputs/recovery_plots
```

This writes individual and overlaid PNG/PDF figures without rerunning inference.
See [posterior analysis](outputs-analysis.md) for comparing results.
The `outputs/` directory is ignored by Git.

Give changed observations or models their own output and matrix directories.
Rebuild the KL basis if its instrument, field, foreground prior or noise changes;
it is not interchangeable between configurations.
The runner refuses to replace existing sampler products. Use `--resume` with
the individual run command to continue the same run, or choose fresh output,
matrix and KL paths for another configuration.

## Regenerating the inputs locally

Running the examples needs no external sky files. Regeneration uses the local
ICRS GSM and point-source `skyh5` files from the sky-generation workflow:

```bash
python test_examples/generate_data.py \
  --gsm-sky /path/to/gsm/fov-360.skyh5 \
  --point-sky /path/to/gleam_ateam/fov-360.skyh5 \
  --output test_examples
```

This replaces the two visibility vectors and shared instrument files. It
selects only the required channels and sky directions, integrates their
visibilities locally, and records the input histories, units, observation
coordinates and simulation seed in `test_examples/instrument/simulation.json`.
No HPC connection or job submission is involved.
