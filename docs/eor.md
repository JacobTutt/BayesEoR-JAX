# EoR Model

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

The EoR model represents the cosmological 21-cm signal in a 3D model space that is connected to observed visibilities through the forward model.

## Model Space

BayesEoR-JAX builds a rectilinear model grid over transverse Fourier coordinates and line-of-sight modes:

- `u` and `v` describe angular Fourier modes on the sky.
- `eta` describes Fourier modes along the frequency axis.
- Cosmological conversions map these to physical k coordinates.

The key EoR grid parameters are:

- `n_u_eor`: number of EoR model pixels along the `u` axis.
- `n_v_eor`: number of EoR model pixels along the `v` axis. If omitted, it defaults to `n_u_eor`.
- `du_eor` and `dv_eor`: explicit UV spacings. If omitted, they are derived from `fov_eor`.
- `n_eta`: number of line-of-sight Fourier modes. If omitted, it defaults to `n_freq`.
- `fov_eor`: circular EoR field of view in degrees.

The EoR model excludes the `eta = 0` mode. Foreground models handle the spectrally smooth component.

## k Bins

The EoR coefficients are grouped into spherical k bins. The sampler does not infer every EoR coefficient directly in the marginalised likelihood. Instead, it samples power-spectrum amplitudes that determine the prior variance of EoR modes in each k bin.

The default reported quantity is:

```text
Delta^2(k) = k^3 P(k) / (2 pi^2)
```

where `P(k)` is the dimensional power spectrum. The output metadata records the representative k value, bin edges, and number of model modes contributing to each bin.

## Priors

Power-spectrum priors are supplied as a list of `[min, max]` ranges in log-space:

```yaml
priors:
  - [-2.0, 2.0]
  - [-1.2, 2.8]
  - [-0.7, 3.3]
```

The number of prior ranges should match the number of non-empty k bins produced by the chosen model grid and mask.

Use `uniform_priors` to mark bins that should use the old BayesEoR-style linear-uniform physical prior. The sampler coordinates remain log-space; the correction is applied through the likelihood.

## Configuration Example

```yaml
n_u_eor: 15
fov_eor: 12.9080728652
nside: 128

n_freq: 38
f0_mhz: 158.304048743
df_mhz: 0.23761898685801028

priors:
  - [-2.0, 2.0]
  - [-1.2, 2.8]
  - [-0.7, 3.3]
```

The shipped example configs in `test_data/eor/` and `test_data/eor_fgs/` show complete working parameter sets.

## Choosing Grid Size

Increasing `n_u_eor`, `n_v_eor`, `n_eta`, or `fov_eor` increases the number of latent EoR modes and the size of the cached matrix products. This can rapidly dominate memory use because the marginalised likelihood works with dense parameter-space matrices.

Use small grids for installation checks and method development. Production runs should document the intended FoV, k-bin coverage, and memory budget before submission.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
