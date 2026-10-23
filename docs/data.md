# Visibilities and input preparation

BayesEoR analyses complex interferometric visibilities indexed by baseline,
frequency and observing time. The data preparation stage loads these measurements
and the corresponding observation metadata, then arranges them in the same order
as the rows of the forward operator. It does not calibrate the observations or
construct a sky simulation.

Set `data_path` in the run configuration. During
[setup](configuration.md#typed-configuration-workflow),
[get_vis_data](api/data-analysis.md#bayeseor_jax.data.vis.get_vis_data)
returns the visibility vector, noise information, baseline coordinates,
redundancies, frequencies and observing times used by the model.

## Supported inputs

| Format | Input | Preparation |
| --- | --- | --- |
| `.uvh5` | UVH5 file | Read with pyuvdata, select and convert to the analysis convention. |
| `.uvfits` | UVFITS file | Same preprocessing route. |
| `.ms` | Measurement Set directory | Same route, with the required pyuvdata reader dependencies installed. |
| `.npy` | Preprocessed visibility vector | Load directly; supply matching instrument and observation metadata. |

The loader selects its reader from these lowercase suffixes. Raw visibility
files must already be calibrated and use Jy or K sr; preprocessing converts
them to **mK sr**. NumPy arrays must already use mK sr and the ordering below.
The same interfaces accept simulated or observed visibilities.

NumPy inputs can be plain arrays or BayesEoR dictionary payloads containing
a `data` entry. Dictionary files use pickle: load them only from trusted sources.

## Selecting and preparing raw visibilities

For pyuvdata inputs, the available controls include:

- **Baselines:** `antenna_selection` and `bl_cutoff`.
- **Frequency window:** `freq_idx_min`, `freq_min` or `freq_center`, with
  `n_freqs_select` specifying the window width.
- **Time window:** `jd_idx_min`, `jd_min` or `jd_center`, with `n_times_select`.
- **Polarisation:** form pseudo-Stokes I with `form_pi`, or select `pol`.
- **Redundancy:** optionally average redundant baselines with `redundant_avg`.

Selection widths must be paired with a starting index, lower bound or centre.
Frequencies supplied to the raw selection options are in Hz; Julian dates
specify time windows. The default pseudo-Stokes combination is
`pI = pi_norm * (XX + YY)`, with `pi_norm = 1`, not one half.

Preprocessing unprojects the data before vectorisation. The selected observation
provides the actual frequency/time axes and baseline coordinates for setup.
Fully flagged baselines are removed, but partial flags are **not** converted
into per-visibility likelihood weights. Flag handling must therefore be checked
before using a dataset for inference.

## Visibility units and ordering

The flattened data vector has time varying slowest, frequency next and baseline
fastest. Within each time/frequency block, the selected baselines are followed
by their conjugate copies at reversed baseline coordinates:

$$
\boldsymbol d_{tf}
=\begin{pmatrix}\boldsymbol v_{tf}\\\boldsymbol v_{tf}^{*}\end{pmatrix},
\qquad
\boldsymbol b\longrightarrow-\boldsymbol b.
$$

These copies are not independent measurements. The
[marginalised likelihood](inference.md#complex-storage-and-the-factors-of-one-half)
uses this convention explicitly. Data, noise vectors and operator rows must
match in both ordering and units; raw ingestion constructs this arrangement,
whereas NumPy inputs bypass those conversions.

## Supplying preprocessed arrays

A `.npy` input requires `inst_model`, pointing to baseline and redundancy
metadata, and explicit observation settings:

```yaml
data_path: inputs/visibilities.npy
inst_model: inputs/instrument/
n_freq: 38
f0_mhz: 158.304048743
df_mhz: 0.23761898685801028
n_t: 1
d_t: 10.0
jd_center: 2461040.5
noise_data_path: inputs/noise.npy
```

These numbers illustrate the required fields; they must describe your actual
data. Supply either `jd_min` or `jd_center`. Metadata stored inside a visibility
dictionary do not replace these settings. Instrument files are described under
[instrument model files](instrument.md#instrument-model-files).

## Noise already present versus noise to add

Noise handling determines whether the input vector is modified:

| Configuration | Data passed to inference |
| --- | --- |
| `noise_data_path` supplied | Input unchanged; the separate noise vector is returned, not added. |
| Raw-data noise estimate available through `calc_noise` | Input unchanged; reuse the even/odd time-difference estimate. |
| Neither supplied nor estimated noise | Treat input as noiseless and add a synthetic realisation using `sigma` and `noise_seed`. |

Supplied noise takes precedence over estimated noise. `calc_noise` does not
estimate noise from preprocessed NumPy arrays. Do not use the synthetic-noise
route unintentionally on measurements that already contain noise.

The likelihood weights are constructed separately by `build_noise_variance`.
With redundancy $R_{tb}$, its default variance is
$N_{tfb}=\sigma^2/R_{tb}$, where $\sigma$ is the complex visibility RMS.
`effective_noise_path` instead supplies per-visibility amplitudes whose squared
magnitudes define the diagonal variance. It changes the weights, not the loader's
decision to add noise, and does not replace `noise_data_path`.

## Passing prepared data into an analysis

`run_setup(config)` uses the loader's `vis_noisy` vector for likelihood
preparation. The `model` setting selects which components are inferred, not
which components exist in the input: foreground-only and joint models can
therefore be applied to the same dataset.

For signal-injection studies, prepare the combined signal and foreground
visibilities in a supported input format before setup. The loader does not
itself inject a cosmological sky component; its optional addition is thermal
noise. Continue to [Examples](examples.md) for the supplied datasets and commands.
