# Cosmological signal and foregrounds

Load supplied noiseless signal-plus-foreground visibilities, add thermal noise,
construct the foreground KL basis and jointly infer the 21-cm power spectrum.
This is a separate run with its own data and configuration; the signal-only
example does not need to be run first. No sky simulation or Slurm is required.

## Run

From the repository root, after installing the CPU environment and MultiNest:

```bash
python -m examples.eor_fgs.run --sample
```

Omit `--sample` to stop after KL preparation, data loading, noise generation
and matrix construction. The entry point defaults to CPU JAX and performs KL
preparation automatically through `run_setup(config, build_foreground_kl=True)`,
using the same resolved observation, physical prior and noise model as inference.
The example contains no separate matrix-construction or decomposition logic.

```text
eor_fgs/
├── config.yaml
├── run.py
└── data/
    ├── visibilities.npy
    ├── uvw_model.npy
    ├── redundancy_model.npy
    └── simulation.json
```

## Supplied observation and sky

The setup follows [Tutt et al. (2026, manuscript)](../../docs/references.md#wide-field-foreground-compression),
*A Karhunen–Loève Latent Space for Bayesian Uncertainty Quantification and
Evidence Evaluation in Interferometric Visibilities*, with reduced time,
frequency and sky coverage for a local demonstration.

| Setting | Value |
| --- | --- |
| Instrument | 37-antenna HERA-like hexagonal layout, 14.6-m spacing |
| Baselines | All 30 signed groups below 40 m; 15 independent plus conjugates |
| Beam | 14.2-m Airy power beam |
| Frequencies | 15 channels, 158.3984375–159.765625 MHz, spaced by 97.65625 kHz |
| Times | 15 integrations spaced by 21.51137 s, centred on JD 2459999.097166265 |
| Data | 6,750 complex entries in mK sr; 3,375 independent measurements |
| Both sky components | 10° time-union footprint selected at HEALPix `Nside=64` |
| Added noise | Complex RMS 0.006464 mK sr; seed 742123 |

The cosmological contribution is the same independent-pixel Gaussian realisation
as in the signal-only example: standard deviation 6.4 mK at `Nside=256`, with
flat ensemble dimensional power `P ≈ 8.84 × 10⁴ mK² Mpc³`. Foregrounds add
GSM2008 diffuse emission at `Nside=128` and a statistical extragalactic plus
catalogued A-team source population at their individual positions. All components
were integrated through the beam and full three-dimensional fringe response.

Sky membership was defined on the `Nside=64` grid: finer pixels inherit their
parent's selection, and point sources must lie inside a selected parent pixel.
`data/simulation.json` records the exact footprint, times, frequencies, seeds
and source provenance. No external sky files are needed to run inference.

## Inference configuration and results

Both model components share `Nside=128`, while `footprint_nside=64` preserves
the data's sky boundary. The foreground model has three spectral coefficients
per pixel, using a constant and power laws with indices 2.63 and 2.82.
The physical prior and noise weights enter a native-JAX KL construction with
discarded-RMS threshold `f=0.1`, retaining 179 modes from 1,248 coefficients.
The retained foreground coefficients have standard-normal priors.

Combined with 1,680 cosmological coefficients, the likelihood marginalises
1,859 coefficients and samples six power-spectrum amplitudes. MultiNest uses
60 live points and seed 42. Each bin currently has the **same** independent
prior, `log10(Δ² / mK²) ~ Uniform(-1, 6)`. Grey boxes show these actual bounds,
not a uniform density in linear power or the manuscript's bin-dependent priors.

Outputs go to `outputs/laptop_eor_fgs_nside128/`: noisy data, noise draw, KL
diagnostics, matrices, posterior samples and evidence. The weighted operator
`A_fg.npy` is saved with the likelihood matrices; the reduced operator and SVD
diagnostics are saved under `kl/`. The pipeline's
`bayeseor_jax.analyze.save_analysis` runs automatically after sampling and writes
the posterior summary and three titled figures (PNG and PDF) to this example's
`plots/` folder. No separate plotting script is required.
Use `--resume` only for the same run; changed settings require fresh output,
matrix and KL paths to avoid stale results.

These 15 times and channels are central subsets of the manuscript's 47-time,
39-channel observation. Both simulated components are restricted to 10°, so
this is not a full-sky validation or a demonstration of horizon recovery.
The shared footprint does not guarantee exact representation of point-source
subpixel structure. The small live-point budget illustrates the workflow,
not validated scientific constraints. See the [examples guide](../../docs/examples.md)
for output conventions.

![Pipeline power-spectrum and posterior plots](plots/power_spectrum_and_posteriors.png)
