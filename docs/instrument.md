# Instrument and observation

The instrument model specifies how the sky is sampled into the
[visibilities](visibilities.md): baseline coordinates, observing times and
frequencies, telescope location and primary-beam response. These settings must
describe the same observation as the supplied data.

Raw visibility files supply their selected baseline, frequency and time metadata
during ingestion. For preprocessed NumPy vectors, provide these explicitly using
`inst_model` and the observation settings below. The telescope location and beam
are specified in the run configuration in either case.

## Instrument model files

The array layout enters the forward model through its sampled baseline vectors,
not an antenna-position table alone. For preprocessed data, set

```yaml
inst_model: inputs/instrument/
```

The directory contains:

| File | Contents |
| --- | --- |
| `uvw_model.npy` | Baseline coordinates in **metres**, shaped `(n_t, n_bls, 3)`. |
| `redundancy_model.npy` | Number of measurements averaged into each baseline, shaped `(n_t, n_bls)` or `(n_t, n_bls, 1)`. Use one for unaveraged baselines. |
| `antpairs.npy` | Optional antenna-pair labels in baseline order. |

Alternatively, `instrument_model.npy` can contain a dictionary with `uvw_model`
and `redundancy_model` entries and optional `antpairs`; this combined file takes
precedence over the separate arrays. Dictionary files should come from trusted
sources because loading them uses pickle.

Despite the `uvw` filename, these coordinates are physical baseline lengths,
not wavelengths. Their coordinate frame, orientation and ordering must match
the unphased visibility convention used by the forward model. The baseline axis
includes the reversed-baseline conjugate copies described in
[visibility ordering](data.md#visibility-units-and-ordering); do not supply
only the original half for a conjugate-augmented data vector.

For raw inputs, preprocessing extracts this geometry from the selected data;
`save_model: true` saves the resulting instrument arrays to `save_dir`.

## Frequency and time

The frequency axis is configured with:

```yaml
n_freq: 38
f0_mhz: 158.304048743
df_mhz: 0.23761898685801028
```

For preprocessed inputs, these define the frequencies as
$\nu_f=(\texttt{f0\_mhz}+f\,\texttt{df\_mhz})\times10^6$ Hz, starting at
$f=0$. Raw inputs instead provide the selected frequencies through ingestion;
their selection controls are described in [input preparation](data.md#selecting-and-preparing-raw-visibilities).

Time geometry is controlled with:

```yaml
n_t: 1
d_t: 10.0
jd_center: 2461040.5
```

`n_t` is the number of integrations, `d_t` their spacing in seconds, and
`jd_center` anchors the time grid in Julian date. Alternatively, `jd_min`
specifies the first integration. Preprocessed inputs require `d_t` even for a
single integration; raw inputs supply their selected timestamps.

Set the telescope's geodetic latitude and longitude in degrees and altitude
in metres:

```yaml
telescope_latlonalt: [-30.72152777777791, 21.428305555555557, 1073.0000000093132]
```

Together with the observation times, this location determines the sky directions
relative to the instrument and the horizon. The values above are an example,
not a restriction to a particular array.

## Primary beam

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

Continue to [input preparation](data.md) for supported visibility formats,
selection, units and noise handling. Keep the instrument files and configuration
with each run so its inferred power spectrum can be traced to the observation
and response used to model it.
