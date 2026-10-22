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
are defined in `model/operators.py`; the geometry modules contain only their
coordinate definitions.

There are two explicit interfaces to the same mathematics:

- `T_21(...)` and `T_fg(...)` evaluate a supplied block and return a JAX array.
  They compose `build_sky_to_visibility_operator` with the component factors,
  without choosing hardware, reading files, or applying priors or KL reduction.
- `T_21_build_save(...)` and `T_fg_build_save(...)` evaluate those functions on
  CPU in distributed time-frequency blocks, returning paths to complete
  disk-backed component matrices. `BuildMatrices` exposes methods with these
  same names using its configured observation and model.

The pure functions can be used directly inside compiled computations:

```python
import jax
from bayeseor_jax.model.operators import T_21, T_fg

# Inputs have already been selected for matching times, frequencies and baselines.
# F_prime, F_z and Q contain the chosen component-mode columns.
T21_block = jax.jit(T_21)(lmn_21, uvw_lambda, beam_21, pixel_area_sr, F_prime, F_z)
Tfg_block = jax.jit(T_fg)(lmn_fg, uvw_lambda, beam_fg, pixel_area_sr, Q)
```

Here `lmn` is `(time, 3, pixel)`, `uvw_lambda` is `(frequency, 3, baseline)`
and the horizon-masked beam is `(time, frequency, pixel)`. Cosmological pixel
summation must still cover its full model field; selecting fewer pixels is not
equivalent to selecting fewer Fourier coefficients. Array shapes determine the
compiled block dimensions. Calling a pure function does materialise its requested
matrix block; it does not automatically provide a matrix-free solver.

For disk-backed construction, every participating process makes the same calls:

```python
from bayeseor_jax.model.construction.forward import (
    T_21_build_save, T_fg_build_save, assemble_joint_operator,
)

path_21 = T_21_build_save(
    lmn_21, uvw_lambda, beam_21, pixel_area_sr, F_prime, F_z,
    output_path="arrays/T_21.npy",
)
path_fg = T_fg_build_save(
    lmn_fg, uvw_lambda, beam_fg, pixel_area_sr, Q,
    output_path="arrays/T_fg.npy",
)
path_joint = assemble_joint_operator(path_21, path_fg, output_path="arrays/T.npy")
```

Construction discovers the existing JAX process ranks, a local CPU device and
the available logical CPU count, respecting affinity and Slurm's per-task count
where supplied. It does **not** launch workers, initialise distributed JAX or
configure XLA thread pools; CPU binding and thread budgets belong to the job
launcher. A JAX CPU device is not a physical CPU core.

Each process receives a disjoint contiguous range of flattened time-frequency
blocks, with baseline varying fastest within each block. All processes need
access to the same shared filesystem. Completed blocks are flushed before the
temporary file is published under its final name. Joint assembly copies the
saved component columns in blocks rather than rebuilding them or loading the
full arrays onto a GPU. This stage remains independent of KL reduction.

## Code Organisation

The physical model and its construction live together under `bayeseor_jax/model/`:

```text
model/
├── instrument/
│   ├── beam.py                 # Analytic and interpolated primary-beam responses
│   ├── baselines.py            # Sampled baselines, redundancy and antenna pairs
│   └── noise.py                # Instrumental thermal-noise generation
├── observation/
│   └── image_geometry.py       # HEALPix selection and time-dependent directions
├── cosmology/
│   ├── fourier_geometry.py     # Cosmological (u, v, eta) coefficient grid
│   ├── transforms.py           # Frequency and comoving-distance conversions
│   └── power_spectrum.py       # Physical k coordinates and binning
├── operators.py                # Numerical mappings between model spaces
└── construction/
    ├── builder.py              # BuildMatrices orchestration
    ├── forward.py              # CPU construction and saved-component assembly
    ├── storage.py              # Saved arrays and shared memory-mapped files
    └── distribution.py         # Device meshes and visibility-row partitions
```

The mathematical operators do not manage files or device allocation. The builder
combines the physical ingredients and delegates those execution details to the
construction helpers. Numerical builders return arrays; methods that construct
large saved operators return their file paths.

Public imports such as `from bayeseor_jax import BuildMatrices` remain unchanged.
The former `build_matricies` package has been removed; direct module imports
should use the paths above. The old `model.instrument`, `model.noise`,
`model.cosmology_transforms`, and `model.k_cube` implementations now live in
`model.instrument.baselines`, `model.instrument.noise`, `model.cosmology.transforms`,
and `model.cosmology.power_spectrum`, respectively.
The former saving methods `BuildMatrices.build_T_21` and `build_T_fg` are now
named `T_21_build_save` and `T_fg_build_save` to distinguish them from pure
operator evaluation. `BuildMatrices.build_T` assembles the separate files,
building missing components first.

## Cached Matrix Products

The marginalised likelihood does not rebuild the full model at every sampler step. Setup constructs and caches matrix products that are reused during inference:

- `T_21.npy`: cosmological Fourier-coefficient operator.
- `T_fg.npy`: original foreground spectral-amplitude operator.
- `T.npy`: joint operator assembled from those saved component columns.
- `Ninv.npy`: inverse-noise representation.
- `T_Ninv_T.npy`: `T^H N^{-1} T`.
- `dbar.npy`: `T^H N^{-1} d`.
- `d_ninv_d.npy`: `d^H N^{-1} d`.

These files are written under `array_save_directory`. If they already exist and `clobber` is false, the setup path can reuse them.
Reuse assumes identical observation and model settings; a filename alone does
not validate that configuration. Replacing a component through `BuildMatrices`
invalidates its joint matrix, Gram matrix and data projection. The standalone
save functions do not manage caches belonging to other analyses. Keeping the
two component files as well as the assembled joint matrix increases disk usage
but allows independent construction and reuse.

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
