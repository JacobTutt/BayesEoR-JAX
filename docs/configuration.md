# Configuration

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX uses `jsonargparse` to support YAML configuration files and command-line overrides. The parser is `BayesEoRParser`.

YAML keys use underscores, while CLI flags use dashes:

```yaml
n_u_eor: 15
fov_eor: 12.9
```

```bash
python examples/run_analysis_example.py --n-u-eor 15 --fov-eor 12.9
```

For reproducible analyses, prefer YAML files committed or archived with the outputs.

## Minimal Structure

Most runs need these groups:

```yaml
data_path: test_data/eor/eor.uvh5

n_u_eor: 15
fov_eor: 12.9080728652
nside: 128

n_freq: 38
f0_mhz: 158.304048743
df_mhz: 0.23761898685801028

telescope_latlonalt: [-30.72152777777791, 21.428305555555557, 1073.0000000093132]
beam_type: gaussian
fwhm_deg: 9.306821090681533

sigma: 0.00615864342588761

priors:
  - [-2.0, 2.0]
  - [-1.2, 2.8]
```

The complete shipped examples are:

- `test_data/eor/config.yaml`
- `test_data/eor_fgs/config.yaml`

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
| `n_u_eor`, `n_v_eor` | EoR UV grid shape. `n_v_eor` defaults to `n_u_eor`. |
| `du_eor`, `dv_eor` | Optional explicit UV spacing. Defaults derive from `fov_eor`. |
| `fov_eor` | Circular EoR field of view in degrees. |
| `n_eta` | Number of line-of-sight Fourier modes. Defaults to `n_freq`. |
| `nside` | HEALPix resolution used for image-domain sky coordinates. |

These parameters control k-bin coverage and dense matrix size.

## Foregrounds

| Key | Purpose |
| --- | --- |
| `n_u_fg`, `n_v_fg` | Foreground UV grid shape. Defaults inherit the EoR grid. |
| `du_fg`, `dv_fg` | Optional foreground UV spacing. Defaults derive from `fov_fg`. |
| `fov_fg` | Foreground model field of view. Defaults to `fov_eor`. |
| `n_modes_fg` | Number of foreground or LSSM modes. |
| `n_powerlaws_fg` | Number of power-law foreground modes. |
| `beta_fg` | Spectral indices for foreground power-law modes. |
| `fit_for_monopole` | Include a dedicated `(u, v) = (0, 0)` foreground coefficient. |
| `fg_model_domain` | `uv` or `image`. |

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
| `inverse_lw_power` | Inverse prior power for foreground/LSSM coefficients. |
| `use_intrinsic_noise_fitting` | Include a global intrinsic noise amplitude. |
| `n_live_scale_factor` | Live points as a multiple of parameter dimension. |
| `num_inner_scale_factor` | BlackJAX inner steps as a multiple of dimension. |
| `delete_scale_factor` | Fraction of live points replaced per nested-sampling step. |
| `stop_threshold` | Nested-sampling termination threshold. |
| `random_key` | JAX PRNG seed. |

Record sampler controls with every run because they affect both runtime and posterior quality.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
