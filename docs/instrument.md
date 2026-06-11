# Instrument Model

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

The instrument model describes how sky emission is sampled into interferometric visibilities. In BayesEoR-JAX this includes baseline coordinates, time and frequency selection, the primary beam, redundancy, and noise.

## Visibility Inputs

`data_path` can point to either:

- A pyuvdata-compatible visibility file, such as `.uvh5`, `.uvfits`, or `.ms`.
- A preprocessed NumPy-compatible file produced by the BayesEoR-JAX preprocessing path.

Raw pyuvdata ingestion supports selection and preprocessing controls including:

- `antenna_selection`
- `bl_cutoff`
- `freq_idx_min`, `freq_min`, `freq_center`, `n_freqs_select`
- `jd_idx_min`, `jd_min`, `jd_center`, `n_times_select`
- `form_pi` and `pi_norm`
- `redundant_avg`
- `calc_noise`

The visibility vector follows the package's internal ordering after preprocessing. When preparing external preprocessed inputs, record baseline, frequency, time, conjugation, and polarization conventions explicitly.

## Frequency And Time

The frequency axis is configured with:

```yaml
n_freq: 38
f0_mhz: 158.304048743
df_mhz: 0.23761898685801028
```

For raw visibility files, data-driven frequency values can override or validate these settings. For preprocessed inputs, these values define the model axis.

Time geometry is controlled with:

```yaml
n_t: 1
d_t: 10.0
jd_center: 2461040.5
```

`d_t` is required for multi-time model construction when the time axis cannot be inferred from raw data.

## Beam Models

The `beam_type` parameter can be one of the analytic beam models or a path to a pyuvdata-compatible beam file.

Supported analytic names include:

- `uniform`
- `gaussian`
- `airy`
- `gausscosine`
- `taperairy`
- `tanhairy`

Common beam parameters are:

- `peak_amp`
- `fwhm_deg`
- `diam`
- `cosfreq`
- `tanh_freq`
- `tanh_sl_red`
- `pol`
- `freq_interp_kind`

Example Gaussian beam configuration:

```yaml
beam_type: gaussian
fwhm_deg: 9.306821090681533
telescope_latlonalt: [-30.72152777777791, 21.428305555555557, 1073.0000000093132]
```

Beam sidelobes are scientifically important. If the beam has non-negligible response outside the EoR model FoV, the foreground FoV may need to be larger than the EoR FoV.

## Noise

Noise can be supplied or generated through:

- `sigma`: visibility-noise standard deviation.
- `noise_seed`: random seed used when generating Gaussian noise.
- `noise_data_path`: preprocessed noise vector.
- `effective_noise_path`: per-visibility effective noise array.
- `calc_noise`: empirical noise estimate from even/odd differencing.

The marginalised likelihood uses noise weighting through `N^{-1}`. Document whether a run used simulated Gaussian noise, precomputed noise, or an empirical noise estimate.

## Instrument Model Files

For preprocessed workflows, an instrument-model directory can contain arrays such as baseline coordinates and redundancy metadata. The most important items are:

- `uvw_model.npy`
- `redundancy_model.npy`
- optional antenna-pair or phasing metadata

Keep these files versioned or archived with the run configuration. A power-spectrum posterior is only interpretable together with the instrument model used to produce it.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
