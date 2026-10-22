# Instrument Model

The instrument model describes how sky emission is sampled into interferometric visibilities. In BayesEoR-JAX this includes baseline coordinates, time and frequency selection, the primary beam, redundancy, and noise.

## Visibility Inputs

Visibility ingestion and noise preparation live in `bayeseor_jax/data/`:

- `vis.py`: loading, selection, unit/polarisation conversion and vector ordering.
- `noise.py`: simulated noise realisations.

The diagonal noise covariance used by the model and likelihood is defined
separately in `bayeseor_jax/model/instrument/noise.py`.

`get_vis_data` accepts these lowercase `data_path` suffixes:

- `.uvh5` or `.uvfits` files, or `.ms` Measurement Set directories, read through
  pyuvdata with its corresponding reader dependencies installed. Raw inputs
  must be calibrated in Jy or K sr and are converted to mK sr.
- `.npy` visibility arrays, either plain arrays or trusted BayesEoR dictionary
  payloads with a `data` entry. These must already be in mK sr and require
  `inst_model`, `n_freq`, `f0_mhz`, `df_mhz`, `n_t`, `d_t` and a `jd_min` or
  `jd_center`. Stored dictionary metadata do not replace these arguments.

Raw pyuvdata ingestion supports selection and preprocessing controls including:

- `antenna_selection`
- `bl_cutoff`
- `freq_idx_min`, `freq_min`, `freq_center`, `n_freqs_select`
- `jd_idx_min`, `jd_min`, `jd_center`, `n_times_select`
- `form_pi` and `pi_norm`
- `redundant_avg`
- `calc_noise`

Vectors are flattened with time slowest, frequency next and baseline fastest.
Each time/frequency block contains the original baselines followed by conjugate
copies at reversed baselines. These copies are not independent observations.
NumPy inputs bypass preprocessing and must already follow this convention.
Fully flagged raw baselines are removed; partial flags are not propagated into
per-visibility likelihood weights by this loader.

Use `from bayeseor_jax.data import get_vis_data, generate_gaussian_noise` for data
preparation and `from bayeseor_jax.model.instrument.noise import build_noise_variance`
for likelihood weights. Existing top-level visibility helpers remain available;
direct imports from `bayeseor_jax.vis` should now use `bayeseor_jax.data.vis`.

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

Supplied or estimated noise is returned alongside the existing data, not added
again. Without either, `get_vis_data` treats its input as noiseless and adds a
synthetic realisation using `sigma`. `build_noise_variance` constructs covariance
diagonal entries separately and never generates or adds noise.

## Instrument Model Files

For preprocessed workflows, an instrument-model directory can contain arrays such as baseline coordinates and redundancy metadata. The most important items are:

- `uvw_model.npy`
- `redundancy_model.npy`
- optional antenna-pair or phasing metadata

Keep these files versioned or archived with the run configuration. A power-spectrum posterior is only interpretable together with the instrument model used to produce it.
