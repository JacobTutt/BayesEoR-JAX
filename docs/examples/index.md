# Examples

These two independent examples run locally on CPU using supplied noiseless data:
infer the cosmological power spectrum alone, or construct a foreground KL basis
and perform joint inference. Each has its own data, configuration and entry point.
Run the commands from the repository
root after following [installation](../installation.md). Nested sampling requires
the native MultiNest library as well as PyMultiNest; the supplied Conda
environment installs both.

Choose either example, including noise generation, matrix construction, inference
and plots:

```bash
python -m examples.eor.run --sample
# Or, independently:
python -m examples.eor_fgs.run --sample
```

Omit `--sample` to load the supplied data, add noise and build the matrices without
sampling. Both entry points select CPU JAX by default and use the standard pipeline.
No sky-simulation script, external sky files or scheduler scripts are required.

## Observation and inputs

The observing setup follows the wide-field study of
Tutt et al. (2026, manuscript), *A Karhunen–Loève Latent Space for Bayesian
Uncertainty Quantification and Evidence Evaluation in Interferometric Visibilities*,
using the same HERA-like baseline selection, Airy beam and noise prescription.
These are reduced-observation demonstrations, not reproductions of its full-sky
results: they use 15 rather than 47 integrations, 15 rather than 39 channels,
and restrict both simulated components to a 10° field.

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
The shared `Nside=128` used here is finer than the manuscript's `Nside=64`
inference grid; the simulated signal remains at `Nside=256`.

Each example's `data/` folder contains its own noiseless visibility vector,
baseline/redundancy arrays and `simulation.json` provenance record. Setup generates
reproducible thermal noise using the same seed for both examples. No full-sky maps, model matrices or sampler
outputs are stored in Git.

## 1. Cosmological signal only

The input is an independent-pixel, zero-mean Gaussian 21-cm brightness field
with a pixel standard deviation of 6.4 mK, simulated at `Nside=256`.
The time-union footprint is selected on a reference grid at `Nside=64`.
The simulation includes all finer pixels belonging to those selected coarse
pixels before integration through the beam and full three-dimensional fringe
response. It does not select a new circular boundary at the simulation resolution.

```bash
python -m examples.eor.run --sample
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
python -m examples.eor_fgs.run --sample
```

The reduced operator is saved under `outputs/laptop_eor_fgs_nside128/kl/`; the
configuration uses it with a standard-normal foreground prior. Preparation
uses the same physical prior widths and noise covariance as inference.
The joint model marginalises 1,859 coefficients while sampling the six
power-spectrum amplitudes.

The joint entry point calls `run_setup(config, build_foreground_kl=True)`;
no separate preparation script or signal-only run is needed. Setup builds and
reduces the weighted operator using the resolved observation, physical prior
and the same noise covariance as the likelihood.

## Power-spectrum priors

Both supplied configurations currently assign **the same bounds to each of the
six independent amplitudes**:

\[
\rho_s=\log_{10}(\Delta_{21}^2(k_s)/\mathrm{mK}^2)
\sim\mathcal U(-1,6).
\]

These are log-uniform priors on power, not uniform densities in linear power.
The shaded boxes show each bin's support, read from the completed run's
`args.json`; their equal heights reflect the actual configuration. They are not
the bin-dependent priors used in the manuscript. To use different bounds, edit
each pair in `config.yaml` and run inference into a fresh output directory;
changing a plot cannot change the prior used to obtain its posterior.

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

The pipeline's `save_analysis` runs automatically after sampling, saving
`posterior_summary.json` and one combined power-spectrum/posterior PNG in
`examples/eor/plots/` or `examples/eor_fgs/plots/`, next to the corresponding
configuration and input data. Plotting is not reimplemented in the examples.

The joint example additionally writes the weighted foreground operator, KL
spectrum and truncation diagnostics. Each entry point produces its own figures.

The example runner centres the log likelihood before passing it to MultiNest
to avoid numerical loss of precision in the evidence uncertainty. The offset
is recorded in `nested_sampling/likelihood_offset.json` and `summary.json`.
The summary's `logZ` and the analysis loader restore the offset; native
MultiNest text files retain centred values. Fixed data/noise normalisation
constants remain omitted, as in the underlying likelihood.
To regenerate figures from completed samples, use the same pipeline function:

```python
from bayeseor_jax.analyze import save_analysis

save_analysis(
    "outputs/laptop_eor_nside128/nested_sampling",
    output_dir="examples/eor/plots",
    metadata_dir="outputs/laptop_eor_nside128",
    label="Cosmological signal only",
)
```

This reads existing samples without rerunning inference. Pass `expected_ps`
to include a known reference spectrum. Each example loads its supplied
`data/injected_power_spectrum.npy`, a scalar array containing the flat
injected dimensional power in mK² Mpc³. The plot functions also accept a
multi-run `DataContainer` for posterior comparisons.
See [posterior analysis](../inference/posterior-analysis.md) for comparing results.
The `outputs/` directory is ignored by Git.

Give changed observations or models their own output and matrix directories.
Rebuild the KL basis if its instrument, field, foreground prior or noise changes;
it is not interchangeable between configurations.
The runner refuses to replace existing sampler products. Use `--resume` with
the individual run command to continue the same run, or choose fresh output,
matrix and KL paths for another configuration.

Sky simulation is outside the scope of these examples. The supplied
`data/simulation.json` records the sky construction, seeds, footprint, units,
channel frequencies and integration times needed to interpret each dataset.
