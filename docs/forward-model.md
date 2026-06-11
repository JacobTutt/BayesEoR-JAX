# Forward Model

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX models visibility data through a linear forward operator:

```text
d = T x + n
```

where `d` is the visibility vector, `x` contains latent EoR and foreground coefficients, `T` maps those coefficients into observed visibility space, and `n` is noise.

## What `T` Contains

The operator `T` combines:

- EoR model coordinates in `(u, v, eta)` space.
- Foreground model coordinates in UV or image space.
- Frequency-axis transforms.
- HEALPix sky coordinates for image-domain modelling.
- Primary beam response.
- Baseline sampling and time dependence.
- Noise weighting used by the likelihood.

The implementation entry point is `BuildMatrices`, which coordinates the UV model spaces, LMN/HEALPix sky coordinates, beam model, NUDFT kernels, and matrix products.

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

## Documentation Diagram To Add

A useful final diagram for this page should show:

```text
EoR coefficients      Foreground coefficients
        \                  /
         \                /
          -> model vector x
                  |
                  v
      sky/frequency/instrument operator T
                  |
                  v
          visibility prediction T x
                  |
                  v
      likelihood against observed data d
```

For a second, more technical diagram, show the cached products `T^H N^{-1} T`, `T^H N^{-1} d`, and `d^H N^{-1} d` feeding the marginalised likelihood.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
