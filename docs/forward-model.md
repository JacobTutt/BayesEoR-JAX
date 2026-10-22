# Forward Model

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX models visibility data through a linear forward operator:

```text
d = T x + n
```

where `d` is the visibility vector, `x` contains latent EoR and foreground coefficients, `T` maps those coefficients into observed visibility space, and `n` is noise.

## What `T` Contains

The operator `T` combines:

- EoR model coordinates in `(u, v, eta)` space.
- Foreground coefficients on a spherical HEALPix image grid.
- EoR Fourier and foreground spectral basis functions.
- Primary beam response.
- Baseline sampling and time dependence.

The two physical operators are constructed explicitly as

```text
T_21 = F^-1 P F' F_z
T_fg = F^-1 P Q_z
T    = [T_21  T_fg].
```

All transformations between Fourier, image, frequency, and visibility spaces
are defined in `forward_operators.py`; the geometry modules contain only their
coordinate definitions.

`build_sky_to_visibility_operator` forms the shared beam-weighted fringe
response `F^-1 P`. `build_T_21_core` and `build_T_fg_core` then apply the
component operators. The public `BuildMatrices.build_T_21`, `build_T_fg`, and `build_T`
methods distribute time ranges across JAX processes, construct frequency-row
blocks on CPU, and write the result directly to a shared NumPy memmap. This
keeps the exact equation-level model while avoiding a full operator allocation
on any GPU.

## Cached Matrix Products

The marginalised likelihood does not rebuild the full model at every sampler step. Setup constructs and caches matrix products that are reused during inference:

- `T.npy`: dense forward-model operator.
- `Ninv.npy`: inverse-noise representation.
- `T_Ninv_T.npy`: `T^H N^{-1} T`.
- `dbar.npy`: `T^H N^{-1} d`.
- `d_ninv_d.npy`: `d^H N^{-1} d`.

These files are written under `array_save_directory`. If they already exist and `clobber` is false, the setup path can reuse them.

## Setup Flow

`run_setup` performs the high-level orchestration:

1. Load raw or preprocessed visibility data.
2. Resolve model-grid defaults from the configuration.
3. Build k-bin metadata and power-spectrum normalisation terms.
4. Build or load cached matrix products.
5. Construct the prior, precision model, and marginalised likelihood.
6. Return a `SetupResult` used by downstream samplers.

This separation matters because setup can be used as a fast configuration check before launching an expensive sampler.

## Relation To Inference

The marginalised likelihood samples power-spectrum parameters, not every latent sky coefficient. For a proposed power spectrum `theta`, the EoR prior precision changes, producing:

```text
Sigma(theta) = T^H N^{-1} T + Phi^{-1}(theta)
```

The likelihood then evaluates dense linear algebra involving `Sigma(theta)`, including solves and log determinants. This is where the JAXMg downstream branch is currently used.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
