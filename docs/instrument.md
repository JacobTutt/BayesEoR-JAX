# Instrument Model

The instrument model describes how sky emission is sampled into interferometric visibilities. In BayesEoR-JAX this includes baseline coordinates, time and frequency selection, the primary beam, redundancy, and noise.

Visibility loading, supported file formats, row ordering and noise injection
are described in [Data](data.md). This page describes the observation axes,
primary beam and instrument metadata used by the forward model.

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

## Noise covariance

The likelihood uses a diagonal visibility-noise covariance, defined separately
from noise-realisation generation. See [noise handling](data.md#noise-already-present-versus-noise-to-add)
for how supplied, estimated and simulated noise enter the pipeline.

## Instrument Model Files

For preprocessed workflows, an instrument-model directory can contain arrays such as baseline coordinates and redundancy metadata. The most important items are:

- `uvw_model.npy`
- `redundancy_model.npy`
- optional antenna-pair or phasing metadata

Keep these files versioned or archived with the run configuration. A power-spectrum posterior is only interpretable together with the instrument model used to produce it.
