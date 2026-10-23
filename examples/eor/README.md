# Cosmological signal only

Load the supplied noiseless 21-cm visibilities, add seeded thermal noise, build
the cosmological forward model and infer its power spectrum. This example is
independent of the foreground example and requires no sky simulation or Slurm.

## Run

From the repository root, after installing the CPU environment and MultiNest:

```bash
python -m examples.eor.run --sample
```

Omit `--sample` for data loading, noise generation and matrix construction only.
The entry point defaults to CPU JAX. Settings are in `config.yaml`; inputs are
in this example's `data/` folder:

```text
eor/
├── config.yaml
├── run.py
└── data/
    ├── visibilities.npy
    ├── uvw_model.npy
    ├── redundancy_model.npy
    └── simulation.json
```

## Supplied observation

The setup follows [Tutt et al. (2026, manuscript)](../../docs/references.md#wide-field-foreground-compression),
*A Karhunen–Loève Latent Space for Bayesian Uncertainty Quantification and
Evidence Evaluation in Interferometric Visibilities*, with reduced time and
frequency coverage for a local demonstration.

| Setting | Value |
| --- | --- |
| Instrument | 37-antenna HERA-like hexagonal layout, 14.6-m spacing |
| Baselines | All 30 signed groups below 40 m; 15 independent plus conjugates |
| Beam | 14.2-m Airy power beam |
| Frequencies | 15 channels, 158.3984375–159.765625 MHz, spaced by 97.65625 kHz |
| Times | 15 integrations spaced by 21.51137 s, centred on JD 2459999.097166265 |
| Data | 6,750 complex entries in mK sr; 3,375 independent measurements |
| Field | 10° time-union footprint selected at HEALPix `Nside=64` |
| Added noise | Complex RMS 0.006464 mK sr; seed 742123 |

The signal was generated as independent zero-mean Gaussian pixel amplitudes
with standard deviation 6.4 mK at `Nside=256`, independently in each channel.
All children of the selected `Nside=64` pixels were integrated through the
beam and three-dimensional fringe response. The ensemble dimensional spectrum
is flat, `P ≈ 8.84 × 10⁴ mK² Mpc³`; the plots show its corresponding Δ²(k).
`data/simulation.json` records the exact times, frequencies, footprint and seed.

## Inference configuration and results

The model uses `Nside=128` with `footprint_nside=64` to preserve the supplied
sky boundary. Automatic Fourier-grid sizing gives 11 × 11 transverse modes
and 14 nonzero line-of-sight modes, or 1,680 marginalised coefficients.
MultiNest samples six power-spectrum amplitudes with 60 live points and seed 42.
Each bin currently has the **same** independent prior,
`log10(Δ² / mK²) ~ Uniform(-1, 6)`. The grey boxes show this support, not a
uniform density in linear power. These are not the manuscript's bin-dependent priors.

Outputs go to `outputs/laptop_eor_nside128/`: the noisy data and noise draw,
matrices, posterior samples, evidence summary and three analysis plots.
Existing sampler outputs are protected; use `--resume` for the same run or
fresh output and matrix paths for changed settings.

This uses the central 15 times and channels of the manuscript's 47-time,
39-channel setup, not its full observation or full-sky foreground analysis.
The small live-point budget is for demonstrating the workflow, not establishing
validated scientific constraints. See the [examples guide](../../docs/examples.md)
for output conventions and optional comparisons of completed runs.
