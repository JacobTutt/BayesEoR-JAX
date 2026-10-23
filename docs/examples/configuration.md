# Configuration

BayesEoR-JAX uses `jsonargparse` to support YAML configuration files and command-line overrides. The parser is `BayesEoRParser`.

YAML keys use underscores, while CLI flags use dashes:

```yaml
n_u_eor: 15
fov_eor: 12.9
```

```bash
python -m examples.eor.run --n-u-eor 15 --fov-eor 12.9
```

For reproducible analyses, prefer YAML files committed or archived with the outputs.

## Code responsibilities

- `configuration.py` defines `ModelConfig`, `RunConfig` and the YAML/CLI parser.
- `setup.py` coordinates data loading, physical-model construction and likelihood preparation; it does not run the sampler.
- `model/` owns the physical geometry and operators, including automatic Fourier-grid sizing in `model/cosmology/fourier_geometry.py`.
- `inference/` owns priors, likelihood evaluation and sampling, including MultiNest output-path handling.
- `io.py` provides shared array loading/saving and run-metadata output.

The public imports `from bayeseor_jax import BayesEoRParser, run_setup` are
unchanged. Direct imports from the former `bayeseor_jax.params` and
`bayeseor_jax.utils` modules should use the modules above instead;
`parse_uprior_inds` now belongs to `inference/prior.py` and remains available
at package level.

## Typed configuration workflow

The parser returns a `RunConfig`, which setup accepts directly:

```python
from dataclasses import replace
from bayeseor_jax import BayesEoRParser, RunConfig, run_setup

config = BayesEoRParser().parse_args([
    "--config", "examples/eor_fgs/config.yaml",
])
result = run_setup(config)

# A new analysis gets a new configuration, leaving the original unchanged.
foreground_only = replace(config, model="fg", array_save_directory="matrices_fg")
# Direct Python construction also works: RunConfig(data_path=..., ...).
```

`RunConfig` contains requested settings. Selected observation axes and derived
image/Fourier grids remain separate: setup does not overwrite `config.n_freq`,
`config.n_t` or other requests when it resolves them from the data. The original
configuration remains available as `result.config`, while actual inputs are
available through `result.vis_dict` and the builder's geometry objects.

`BuildMatrices` accepts the model settings plus resolved arrays and geometry:

```python
from bayeseor_jax import BuildMatrices, ModelConfig

builder = BuildMatrices(
    ModelConfig(model="eor+fg", beam_type="airy", diam=14.2),
    frequencies_hz=frequencies_hz,
    uvw_array_m=uvw_array_m,
    image_geometry=image_geometry,
    fourier_geometry=fourier_geometry,
)
```

A `RunConfig` extends `ModelConfig` and can be passed directly to this builder.
The builder no longer reconstructs geometry from a second list of nominal
settings. Low-level operator functions continue to take explicit arrays.

Migration: replace `run_setup(**settings)` with `run_setup(RunConfig(**settings))`;
replace long `BuildMatrices(...)` keyword lists with model settings and prepared
geometry as above. Unknown configuration keys now raise an error instead of
being silently discarded by setup. The CLI output-directory default is `./BayesEoROutputs/`,
while direct Python configuration retains the setup default `./`.

## Choosing a HEALPix resolution

`nside` is shared by the foreground and cosmological image grids. For fixed
coverage across resolution changes, optionally set `footprint_nside` to a
coarser power-of-two resolution: the time-union field is selected on that grid,
then every child pixel is modelled at `nside`. For example, `nside: 128` with
`footprint_nside: 64` refines both models without changing the coarse-pixel sky
boundary. With `footprint_nside: null` (the default), selection uses `nside`.

When `nside` is omitted, setup uses the longest selected three-dimensional
baseline and highest selected observing frequency to estimate a shared
foreground and cosmological image resolution. An explicit `nside` is retained,
with a warning if it is below the estimate. The same calculation is available
directly:

```python
from bayeseor_jax import required_healpix_nside

nside = required_healpix_nside(
    longest_baseline_m=40.0,
    highest_frequency_hz=159.082e6,
)  # 64
```

The helper requires the nominal pixel width `sqrt(pi/3) / nside` to be no
larger than `lambda_min / (2 * b_max)`, then rounds up to a power of two.
Set `samples_per_fringe` above two to request finer sampling. This is an
estimate based on the [nominal HEALPix pixel size](https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.nside2resol.html),
not a guarantee of visibility convergence for arbitrary sky structure, beams
or horizon masks. Changes to resolution also require reconsidering the
pixel-domain prior. Setup records the selected `nside` and `nside_required`
estimate in `args.json`, without modifying the input configuration.

## Minimal Structure

Most runs need these groups:

```yaml
model: eor
data_path: examples/eor/data/visibilities.npy
inst_model: examples/eor/data
array_save_directory: outputs/my_example/matrices
output_dir: outputs/my_example

fov_eor: 10.0
nside: 128
footprint_nside: 64

n_freq: 15
f0_mhz: 158.3984375
df_mhz: 0.09765625
n_t: 15
d_t: 21.511369943618774
jd_center: 2459999.097166265

telescope_latlonalt: [-30.72152777777791, 21.428305555555557, 1073.0000000093132]
beam_type: airy
diam: 14.2

sigma: 0.006464

priors: [[-1, 6], [-1, 6], [-1, 6], [-1, 6], [-1, 6], [-1, 6]]
```

The complete shipped examples are:

- `examples/eor/config.yaml`
- `examples/eor_fgs/config.yaml`

## General And Output

| Key | Purpose |
| --- | --- |
| `config` | Path to a YAML configuration file. |
| `quiet` | Reduce printed output. |
| `clobber` | Overwrite existing cached outputs. |
| `run` | Run sampling after setup when using parser-driven scripts. |
| `array_save_directory` | Directory for cached matrices such as `T.npy` and `T_Ninv_T.npy`. |
| `output_dir` | Directory for run metadata and sampler outputs. |

Use separate `array_save_directory` values for model configurations that should not share cached matrices.

## EoR Grid

| Key | Purpose |
| --- | --- |
| `n_u_eor`, `n_v_eor` | Cosmological Fourier-grid dimensions, each inferred independently when omitted. Smaller explicit values produce warnings. |
| `du_eor`, `dv_eor` | Optional explicit UV spacing. Defaults derive from `fov_eor`. |
| `fov_eor` | Circular EoR field of view in degrees. |
| `n_eta` | Number of line-of-sight Fourier modes. Defaults to `n_freq`. |
| `nside` | Shared HEALPix resolution; inferred from the selected baselines and frequencies when omitted. |

These parameters control k-bin coverage and dense matrix size.
The transverse estimates cover the selected baseline coordinates at the highest
frequency plus the beam-aperture buffer, using spacings set by the angular
extent of the cosmological field. Larger explicit grids are accepted without a
warning. Fourier-grid checks are not needed for foreground-only inference.

## Foregrounds

| Key | Purpose |
| --- | --- |
| `fov_fg` | Foreground model field of view. Defaults to `fov_eor`. |
| `beta_fg` | Spectral indices for foreground power-law modes. |

Use a larger `fov_fg` when beam sidelobes or horizon emission are scientifically relevant.

## Frequency And Time

| Key | Purpose |
| --- | --- |
| `n_freq` | Number of frequency channels. |
| `f0_mhz` | Minimum frequency in MHz. |
| `df_mhz` | Frequency channel width in MHz. |
| `n_t` | Number of time integrations. |
| `d_t` | Integration time in seconds. |
| `jd_center` | Central Julian date for centred time selection or model construction. |

For raw pyuvdata inputs, selection options may define the actual subset used.

## Beam And Instrument

| Key | Purpose |
| --- | --- |
| `telescope_latlonalt` | Telescope latitude, longitude, altitude. |
| `beam_type` | Analytic beam name or UVBeam file path. |
| `peak_amp` | Peak beam amplitude. |
| `fwhm_deg` | Gaussian or gausscosine beam width. |
| `diam` | Antenna diameter for Airy-like beams. |
| `cosfreq` | Cosine frequency for gausscosine beams. |
| `tanh_freq`, `tanh_sl_red` | Taper/tanh Airy parameters. |
| `pol` | Beam polarization. |
| `freq_interp_kind` | Frequency interpolation mode for beam files. |
| `inst_model` | Preprocessed instrument-model directory. |

## Data And Noise

| Key | Purpose |
| --- | --- |
| `data_path` | Visibility data path. |
| `antenna_selection` | pyuvdata antenna selection string. |
| `bl_cutoff` | Baseline length cutoff in meters. |
| `freq_idx_min`, `freq_min`, `freq_center`, `n_freqs_select` | Frequency downselection controls. |
| `jd_idx_min`, `jd_min`, `jd_center`, `n_times_select` | Time downselection controls. |
| `form_pi`, `pi_norm` | Pseudo-Stokes I formation. |
| `redundant_avg`, `uniform_redundancy` | Redundant averaging controls. |
| `calc_noise` | Estimate noise from even/odd differencing. |
| `sigma` | Gaussian visibility-noise standard deviation. |
| `noise_data_path` | Preprocessed noise vector. |
| `effective_noise_path` | Per-visibility effective-noise array. |
| `save_vis`, `save_model`, `save_dir` | Save preprocessed data/model products. |

## Priors And Sampler Controls

| Key | Purpose |
| --- | --- |
| `priors` | Power-spectrum prior ranges, one `[min, max]` pair per k bin. |
| `uniform_priors` | Mask for linear-uniform physical priors. |
| `foreground_prior_std_mk` | Physical prior width for each foreground spectral coefficient field. |
| `foreground_kl_path` | Optional saved, unwhitened KL foreground operator; selects unit coefficient widths instead of the original physical prior. |
| `use_intrinsic_noise_fitting` | Include a global intrinsic noise amplitude. |
| `n_live_scale_factor` | Live points as a multiple of parameter dimension. |
| `random_key` | Sampler seed. |

Record sampler controls with every run because they affect both runtime and posterior quality.
