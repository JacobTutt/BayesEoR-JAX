# Examples

These two examples run locally on CPU: first infer the cosmological power
spectrum without foregrounds, then add foreground emission and construct its
KL representation before joint inference. Run the commands from the repository
root after following [installation](installation.md). Nested sampling requires
the native MultiNest library as well as PyMultiNest; the supplied Conda
environment installs both.

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
| Inference image grid | HEALPix `Nside=64` for both components |
| Complex noise RMS | 0.006464 mK sr |

The times and frequencies are the central subsets of the 47-integration,
39-channel observation. There are **6,750 stored complex entries**, including
conjugate baselines, corresponding to **3,375 independent complex measurements**.
The conjugates do not provide additional independent data.

The cosmological model uses an automatically sized 11 × 11 transverse Fourier
grid and 14 nonzero line-of-sight modes: **1,680 marginalised coefficients**
and **six sampled power-spectrum amplitudes**. The joint example adds
**165 KL modes**, reduced from 312 foreground coefficients using `f=0.1`.

The checked-in inputs contain only two noiseless visibility vectors and shared
baseline/redundancy metadata. Setup generates reproducible thermal noise using
the same seed for both examples. No full-sky maps, model matrices or sampler
outputs are stored in Git.

## 1. Cosmological signal only

The input is an independent-pixel, zero-mean Gaussian 21-cm brightness field
with a pixel standard deviation of 6.4 mK, simulated at `Nside=256`.
The time-union footprint is selected on the inference grid at `Nside=64`.
The simulation includes all finer pixels belonging to those selected coarse
pixels before integration through the beam and full three-dimensional fringe
response. It does not select a new circular boundary at the simulation resolution.

```bash
export JAX_PLATFORMS=cpu
python test_examples/run_analysis_example.py \
  --config test_examples/eor/config.yaml --sample
```

Omit `--sample` to build the matrices and likelihood without running MultiNest.
Outputs are written to `outputs/laptop_eor/`.

## 2. Cosmological signal and foregrounds

This dataset adds GSM2008 diffuse emission and a statistical extragalactic/
catalogued bright-source population within the same 10° field. The local
GSM input is sampled at `Nside=128`; point sources retain their individual
positions. Both inherit the same `Nside=64` footprint: every selected coarse
pixel contributes its four `Nside=128` children, and sources are included only
when they lie inside a selected coarse pixel. The inference image grid remains
`Nside=64`. Matching the footprint removes a boundary-selection mismatch but
does not guarantee exact representation of subpixel brightness structure.

First build the prior- and noise-weighted foreground operator and reduce it
using native CPU JAX. Then run joint inference:

```bash
export JAX_PLATFORMS=cpu
python test_examples/eor_fgs/prepare_kl.py
python test_examples/run_analysis_example.py \
  --config test_examples/eor_fgs/config.yaml --sample
```

The reduced operator is saved under `outputs/laptop_eor_fgs/kl/`; the
configuration uses it with a standard-normal foreground prior. Preparation
uses the same physical prior widths and noise covariance as inference.
The joint model marginalises 1,845 coefficients while sampling the six
power-spectrum amplitudes.

These are workflow examples with 60 live points, not validated scientific
constraints. Both simulated components are restricted to 10°: these inputs
do not demonstrate recovery from unmodelled emission outside the field or
from the horizon. The reduced bandwidth also probes different bins from
the full observing band.

## Results and rerunning

Each run writes:

- `matrices/`: forward operators and marginal-likelihood products.
- `nested_sampling/`: posterior samples, evidence and run summary.
- `analysis/`: posterior summaries and power-spectrum/posterior figures.

The joint example additionally writes the KL spectrum and truncation diagnostics.
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
