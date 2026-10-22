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
  same names using its configured observation and model. The weighted variant,
  `T_fg_weighted_build_save(...)`, uses this same CPU writer to construct the
  KL input directly, without saving an unweighted foreground matrix first.

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
from bayeseor_jax.model.construction.operator_construction import (
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

## Prior- and Noise-Weighted Foreground Mapping

The physical operator is kept separate from the uncertainty model. For KL
construction, `T_fg_weighted` evaluates

```text
A = W T_fg L,
Sigma_q = L L^H,     W N W^H = I.
```

Choosing Hermitian covariance roots gives `W = N^(-1/2)` and
`L = Sigma_q^(1/2)`. Consequently `A A^H` is the prior-predictive foreground
covariance in noise-whitened visibility space. These functions apply the supplied
factors only; the decomposition and truncation are separate stages.

For independent coefficients and noise, pass vectors instead of dense matrices:

```python
import jax.numpy as jnp
from bayeseor_jax import T_fg_weighted, weight_foreground_operator

# Repeat each component's physical prior width over its pixels, not vice versa.
prior_std = jnp.repeat(component_prior_std, lmn_fg.shape[-1])
# Ninv_diagonal contains inverse variances E[|n_i|^2]^-1, not inverse RMS.
noise_whitener = jnp.sqrt(jnp.real(Ninv_diagonal))
A = jax.jit(T_fg_weighted)(
    lmn_fg, uvw_lambda, beam_fg, pixel_area_sr, Q,
    prior_sqrt=prior_std, noise_whitener=noise_whitener,
)
# Or apply the identical weighting to a previously constructed matrix/block:
A_from_cached = weight_foreground_operator(Tfg_block, prior_std, noise_whitener)
```

The factor arrays must match the supplied matrix rows and columns. Per-pixel
prior widths and per-visibility noise levels need not be uniform. Dense square
factors support correlations, but mix their respective axes: independently
weighting diagonal sub-blocks is then generally incorrect. The caller supplies
valid covariance factors in consistent physical units. A nonzero prior mean
is propagated separately as a mean visibility contribution, not included in `A`.
Neither function saves an array or selects a device; the existing physical
operator caches and inference setup are unchanged.

For disk-backed construction with diagonal priors and noise, use the same
weights with the CPU block writer:

```python
from bayeseor_jax import T_fg_weighted_build_save

weighted_path = T_fg_weighted_build_save(
    lmn_fg, uvw_lambda, beam_fg, pixel_area_sr, Q,
    prior_sqrt=prior_std, noise_whitener=noise_whitener,
    output_path="arrays/A_fg.npy",
)
# With an existing BuildMatrices configuration, the equivalent call is:
# weighted_path = builder.T_fg_weighted_build_save(
#     prior_sqrt=prior_std, noise_whitener=noise_whitener,
# )
```

All participating processes call this together. Each evaluates its assigned
time-frequency blocks on CPU, applies the prior weights to the columns and
the corresponding noise weights to the rows, then writes directly to the
shared weighted matrix. No complete unweighted matrix is constructed, saved or
reread. The weight vectors can vary across coefficients and visibilities;
dense covariance factors are not supported by this block writer. No reduction
is performed until the separate decomposition call below.

## Distributed Foreground SVD and Truncation

`T_fg_svd_build_save` takes the **already prior- and noise-weighted** matrix
`A = N^(-1/2) T_fg Sigma_q^(1/2)` saved as a complex128 `.npy` file. It loads
the matrix directly into distributed GPU shards, runs JAXMG GESVD, chooses the
retained rank and saves the reduced operator in one call:

```python
from bayeseor_jax import T_fg_svd_build_save, assemble_joint_operator

reduced_path = T_fg_svd_build_save(
    weighted_path, "arrays/Ninv.npy",
    output_path="arrays/T_fg_reduced.npy", f=0.1, tile_size=128,
)
joint_path = assemble_joint_operator(
    "arrays/T_21.npy", reduced_path, output_path="arrays/T.npy",
)
```

`Ninv.npy` must contain the same positive diagonal inverse noise used to
construct `A`. It is used to undo whitening, **not** to apply it again.
The prior is already absorbed in `A` and is not supplied a second time.
For `A = U diag(s) Vh`, the routine selects the smallest `r` satisfying

```text
sqrt(sum(s[r:]**2) / N_d) <= f
```

and writes `T_fg_reduced = N^(1/2) U[:, :r] diag(s[:r])`. The output coefficients
therefore require a **standard-normal foreground prior**, not the original
physical prior widths. The bound is on average prior-predictive RMS in whitened
visibility space, not on every visibility or realised sky. A zero-column operator
is permitted if the whole contribution lies below the threshold; `f=0` retains
all modes with nonzero computed singular values.

The launcher must initialise distributed JAX with **one Python process per GPU**,
and every process calls the function. A near-square mesh and tile-aligned zero
padding make both rectangular inputs and thin left-vector outputs shardable.
Padding is removed from saved arrays and excluded from `N_d` and rank selection.
Only retained local pieces of `U` are copied back to shared storage; `Vh` is not
computed. The input, full thin `U` and solver workspace must still fit the GPUs.
This routine currently assumes diagonal noise even though the pure weighting
function also permits dense noise factors.

Alongside the operator it saves `<stem>.singular_values.npy` and
`<stem>.svd.json`, recording the original/padded dimensions, process grid,
retained rank, achieved discarded RMS and timings. Solver failures are checked
before replacing any completed operator. Each call recomputes the SVD; there is
no implicit reuse that could ignore a changed `f`.

To use the completed reduction in inference, supply its path to `run_setup`
or set the equivalent configuration key:

```yaml
foreground_kl_path: arrays/T_fg_reduced.npy
```

Setup skips construction of the original foreground matrix, concatenates the
saved reduced operator with `T_21`, and prepares the same likelihood products.
It assigns one unit prior width to each retained foreground column, including
an empty foreground block if no modes were retained. The original
`foreground_prior_std_mk` setting is ignored in this case: those physical
widths were already absorbed during weighted construction. Setup does not run
the SVD or overwrite the supplied reduced operator.

The caller must supply an **unwhitened** reduced operator for the intended
observation, visibility ordering and physical foreground prior, not `A_fg.npy`.
Shapes alone cannot verify this provenance. Changes to the construction noise
or truncation threshold require reconsidering the retained basis; setup does
not certify its truncation accuracy for a different noise model. Switching
operator paths invalidates the cached joint matrix and its Gram product.

Without `foreground_kl_path`, setup uses the original physical foreground
matrix and repeats each component's physical prior width across its pixels.
Both representations feed the same `PhiInverse` interface:
`foreground_prior_std` is an explicit vector with one width per foreground
column. Direct callers should use this vector instead of the former
`foreground_prior_std_mk` and `n_fg_spatial_modes` constructor arguments.
The cosmological power-spectrum prior and likelihood algebra are unchanged.

This needs a JAXMG build exposing `gesvd` with `compute_vh=False` and
`return_status=True`. There is no implicit CPU or approximate-SVD fallback.
`tests/gpu/run_foreground_svd.py` provides a small native multi-GPU check using
synthetic tall and wide complex matrices, nonuniform noise and several thresholds.

## Code Organisation

The physical model and its construction live together under `bayeseor_jax/model/`:

```text
model/
├── instrument/
│   ├── beam.py                 # Analytic and interpolated primary-beam responses
│   ├── baselines.py            # Sampled baselines, redundancy and antenna pairs
│   └── noise.py                # Thermal-noise variances and realisations
├── observation/
│   └── image_geometry.py       # HEALPix selection and time-dependent directions
├── cosmology/
│   ├── fourier_geometry.py     # Cosmological (u, v, eta) coefficient grid
│   ├── transforms.py           # Frequency and comoving-distance conversions
│   └── power_spectrum.py       # Physical k coordinates and binning
├── operators.py                # Numerical mappings between model spaces
└── construction/
    ├── model_builder.py        # Physical input preparation and component writers
    ├── operator_construction.py # CPU construction and saved-component assembly
    ├── kl_reduction.py         # Distributed KL reduction of a weighted operator
    ├── likelihood_products.py  # Noise-weighted Gram matrix and data contractions
    ├── storage.py              # Cache checks, saved arrays and publication
    └── parallel_layout.py      # CPU resources, device meshes and row partitions
```

The mathematical operators do not manage files or device allocation. The builder
combines the physical ingredients and delegates those execution details to the
construction helpers. Numerical builders return arrays; methods that construct
large saved operators return their file paths.

Public imports such as `from bayeseor_jax import BuildMatrices` remain unchanged.
Within `model.construction`, the modules formerly named `builder`, `forward`,
`reduce_kl_construction`, `likelihood` and `distribution` are now `model_builder`,
`operator_construction`, `kl_reduction`, `likelihood_products` and `parallel_layout`,
respectively. Direct module imports must use these new paths; function names
and numerical behaviour are unchanged.
The former `build_matricies` package has been removed; direct module imports
should use the paths above. The old `model.instrument`, `model.noise`,
`model.cosmology_transforms`, and `model.k_cube` implementations now live in
`model.instrument.baselines`, `model.instrument.noise`, `model.cosmology.transforms`,
and `model.cosmology.power_spectrum`, respectively.
The former saving methods `BuildMatrices.build_T_21` and `build_T_fg` are now
named `T_21_build_save` and `T_fg_build_save` to distinguish them from pure
operator evaluation. `BuildMatrices` now prepares physical inputs and exposes
those component writers plus `T_fg_weighted_build_save`, whose weights are
explicit arguments rather than constructor settings. The former `build_T`, noise-building,
likelihood-product and stack-building methods have been removed. Their stages
are explicit functions rather than methods that silently construct dependencies.
Noise settings, `clobber` and Gram-block tuning no longer belong to its constructor.

## Assembly and Likelihood Preparation

Once the components exist, select the foreground representation and assemble it
with the cosmological operator. Neither assembly nor likelihood preparation
rebuilds a missing component:

```python
from bayeseor_jax.model.instrument.noise import build_noise_variance
from bayeseor_jax.model.construction.storage import save_array
from bayeseor_jax.model.construction.likelihood_products import (
    save_likelihood_products,
)

# Use path_fg for the original representation, or a separately saved reduced one.
selected_fg = path_fg
path_joint = assemble_joint_operator(path_21, selected_fg, output_path="arrays/T.npy")
variance = build_noise_variance(n_time, n_frequency, n_baseline,
                                sigma=noise_rms, redundancy=redundancy)
path_ninv = save_array("arrays/Ninv.npy", 1.0 / variance)
save_likelihood_products(path_joint, path_ninv, data, output_directory="arrays")
```

These are collective calls when distributed JAX is active. Component writers
run on CPU; the likelihood-product writers use the default device mesh, normally
GPUs. `save_likelihood_products` loads T and inverse noise once across all devices,
keeps each process's visibility stripe resident, and writes the Gram matrix in
coefficient blocks. It reuses a valid cached Gram matrix but always refreshes
the data projection and scalar data power. It does not stream the input from
disk during each contraction. The individual `save_gram_matrix`,
`save_data_projection` and `save_data_power` functions remain available.

For in-memory use, `build_gram_matrix`, `build_data_projection` and
`build_data_power` are pure JAX contractions with no storage or placement policy.
The supplied foreground operator must use the same visibility rows as `T_21`.
Its coefficient prior must also match the chosen representation: physical prior
widths for the original basis, or a standard-normal prior when those widths have
already been absorbed into a reduced operator. The SVD stage above is explicit;
`run_setup(foreground_kl_path=...)` selects a saved reduced operator and its
standard-normal prior. Without that argument, setup uses the original representation.

## Cached Matrix Products

The marginalised likelihood does not rebuild the full model at every sampler step. Setup constructs and caches matrix products that are reused during inference:

- `T_21.npy`: cosmological Fourier-coefficient operator.
- `T_fg.npy`: original foreground spectral-amplitude operator.
- `T.npy`: joint operator assembled from those saved component columns.
- `Ninv.npy`: inverse-noise representation.
- `T_Ninv_T.npy`: `T^H N^{-1} T`.
- `dbar.npy`: `T^H N^{-1} d`.
- `d_ninv_d.npy`: `d^H N^{-1} d`.

These files are written under `array_save_directory`. `ensure_cached` centralises
reuse: the physical components are reused when present unless `clobber` is true,
while derived products have sidecar metadata recording their input paths, file
sizes and modification times. Changing the selected component or rewriting an
input causes its dependent products to be rebuilt on the next cache check.
Noise is compared with the requested variance vector, and the two data-dependent
terms are refreshed on every setup call. Writers publish completed temporary
files without first deleting an existing output.

Physical-component reuse still requires identical observation and model settings;
it does not hash the configuration. Use a separate directory or `clobber=True`
when those settings change. The individual product writers always write their
requested output; `save_likelihood_products` handles Gram-cache reuse. Keeping both components
and the assembled joint matrix increases disk usage but permits independent reuse.

## Setup Flow

`run_setup` performs the high-level orchestration:

1. Load raw or preprocessed visibility data.
2. Resolve model-grid defaults from the configuration.
3. Build k-bin metadata and power-spectrum normalisation terms.
4. Construct or reuse `T_21` and `T_fg` independently, then assemble `T`.
5. Construct the inverse-noise diagonal independently of the coefficient basis.
6. Prepare the Gram matrix and the data projection/scalar from those saved inputs.
7. Construct the prior, precision model, and marginalised likelihood.
8. Return a `SetupResult` used by downstream samplers.

This separation matters because setup can be used as a fast configuration check before launching an expensive sampler.

## Relation To Inference

The marginalised likelihood samples power-spectrum parameters, not every latent sky coefficient. For a proposed power spectrum `theta`, the EoR prior precision changes, producing:

```text
Sigma(theta) = T^H N^{-1} T + Phi^{-1}(theta)
```

The likelihood then evaluates dense linear algebra involving `Sigma(theta)`, including solves and log determinants. This is where the JAXMg downstream branch is currently used.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
