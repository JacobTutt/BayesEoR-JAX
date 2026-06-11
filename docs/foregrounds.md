# Foregrounds

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

Foreground emission is much brighter than the 21-cm EoR signal and is spectrally smoother. BayesEoR-JAX models foreground-like large spectral scale structure jointly with the EoR signal so that uncertainty and covariance between components propagate into the power-spectrum posterior.

## Large Spectral Scale Model

The foreground model includes smooth spectral modes controlled by:

- `n_modes_fg`: number of foreground or large spectral scale modes.
- `n_powerlaws_fg`: number of power-law spectral modes. If omitted, this is derived from `beta_fg` when foreground modes are enabled.
- `beta_fg`: brightness-temperature spectral indices for foreground power laws.
- `fit_for_monopole`: whether to include a dedicated `(u, v) = (0, 0)` coefficient.

In the original BayesEoR literature these models connect to polynomial and power-law descriptions of diffuse Galactic synchrotron emission, extragalactic source populations, and free-free-like components. In BayesEoR-JAX the important user choice is whether the selected foreground basis is expressive enough over the angular and spectral region seen by the instrument.

## UV-Domain Foregrounds

The default foreground spatial basis is UV-domain:

```yaml
fg_model_domain: uv
n_modes_fg: 2
beta_fg: [2.63, 2.82]
fit_for_monopole: true
```

In this mode the non-monopole foreground block assigns coefficient sets to pixels in the foreground UV model grid. If `n_u_fg`, `n_v_fg`, or `fov_fg` are omitted, they inherit the EoR model geometry.

Use UV-domain foregrounds for smaller, traditional BayesEoR-style tests and for parity with many original examples.

## Image-Domain Foregrounds

BayesEoR-JAX also supports an image-domain foreground path:

```yaml
fg_model_domain: image
n_modes_fg: 2
beta_fg: [2.63, 2.82]
fov_fg: 60.0
```

This path is intended for wide-field foreground modelling experiments where off-axis foregrounds and beam sidelobes can bias EoR power-spectrum estimates. It is especially relevant to all-sky modelling questions discussed by Burba, Sims & Pober (2023).

Image-domain foregrounds can be much larger than UV-domain models. Full-resolution wide-field models can produce very large dense matrices, so production runs should plan memory and resolution carefully.

## Field Of View

BayesEoR-JAX separates:

- `fov_eor`: the EoR model field of view.
- `fov_fg`: the foreground model field of view.

The foreground FoV may need to be larger than the EoR FoV because foreground emission outside the target field can enter through sidelobes and chromatic instrumental response. A foreground model that is too narrow can leave structured residuals and bias low-k power-spectrum bins.

## Current Caveats

The image-domain foreground prior currently approximates expensive dense image-space correlations by retaining the UV-implied marginal variances. This is useful for scaling experiments, but it should be documented in scientific analyses that depend on detailed foreground prior structure.

When comparing UV-domain and image-domain runs, record:

- `fg_model_domain`
- `fov_fg`
- `n_u_fg`, `n_v_fg`
- `nside`
- beam type and beam parameters
- foreground spectral indices
- matrix dimensions and memory requirements

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
