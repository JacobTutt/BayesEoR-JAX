# HPC And Scaling

BayesEoR-JAX is designed for accelerator-backed workflows, but the practical scaling limit is usually dense matrix size and likelihood linear algebra rather than Python overhead.

## JAX Devices

Always check the backend on the node that will run the analysis:

```bash
python - <<'PY'
import jax
print(jax.devices())
print(jax.default_backend())
PY
```

For GPU jobs, this should report CUDA devices and a `gpu` backend. If it reports CPU only, the job is not using the intended JAX/CUDA stack.

## Precision

BayesEoR-JAX uses double precision in the core model and likelihood path. Several modules enable:

```python
jax.config.update("jax_enable_x64", True)
```

For scripts and notebooks, set precision before creating arrays or importing heavy JAX code:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

or in the shell:

```bash
export JAX_ENABLE_X64=1
```

## Compilation Cost

JAX compiles kernels on first use for a given shape and dtype. The first setup or likelihood call can therefore be slower than later calls. This is expected and should not be confused with steady-state evaluation time.

Shape changes usually trigger recompilation. Avoid changing model sizes inside loops.

## Matrix Size

The main memory pressure comes from dense parameter-space products such as:

```text
T^H N^{-1} T
```

The dimension grows with:

- EoR UV grid size;
- line-of-sight modes;
- foreground image-domain grid size;
- number of foreground spectral components;
- field of view;
- HEALPix resolution.

Wide-field image-domain foreground operators can become very large. Both physical
components are therefore constructed in CPU time-frequency blocks and written
directly to disk rather than materialised on a GPU. Joint assembly copies the
selected component files without rebuilding them.

Likelihood preparation is separate: `save_likelihood_products` distributes visibility
rows over the existing device mesh and contracts coefficient blocks on the
default backend, normally GPUs. The local input stripe must fit on those devices;
blocking bounds the output workspace but does not stream the input from disk.
Block widths are aligned with the device count, with zero padding removed before
the completed matrix is published. The operator and inverse noise are loaded once
for the Gram matrix, data projection and scalar data power. A valid cached Gram
matrix is reused when only the observations change; one device uses the same path.

## JAXMg

Single-device likelihood and MAP calculations use native JAX on CPU or GPU.
Multi-GPU inference uses JAXMG for dense Hermitian positive-definite solves and
log determinants; GPU KL construction uses its SVD routine.

The `hpc` extra installs `jaxmg[cuda12]` from PyPI on supported Linux x86_64
and aarch64 systems. Use Python 3.12+ and a compatible NVIDIA driver; no
cluster-specific wheels or extras are needed. See [Installation](installation.md).

Distributed JAXMG execution requires one Python process per GPU, with
distributed JAX initialised before creating the device mesh. Scheduler and
network settings belong in the job scripts. For CSD3 examples and MultiNest
shared-library setup, see [CSD3 Getting Started](csd3.md).

## Practical Run Advice

- Run the example setup on an interactive GPU node before submitting a long sampler job.
- Use short output paths when running MultiNest.
- Keep matrix caches on fast project storage.
- Use distinct matrix directories for distinct model configurations.
- Record GPU model, driver, CUDA, JAX, JAXMg, and commit hash in run notes.
- Check that likelihood preparation and sampling use the intended GPU backend;
  physical-component construction deliberately runs on CPU.
