# HPC And Scaling

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

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

Likelihood preparation is separate: `save_gram_matrix` distributes visibility
rows over the existing device mesh and contracts coefficient blocks on the
default backend, normally GPUs. The local input stripe must fit on those devices;
blocking bounds the output workspace but does not stream the input from disk.
Block widths are aligned with the device count, with zero padding removed before
the completed matrix is published.

## JAXMg

The marginalised likelihood uses JAXMg for dense Hermitian positive-definite solves and log determinants. The JAXMg fork used by this project adds log-determinant support needed by BayesEoR-JAX.

The provided extras install prebuilt JAXMg wheels for specific hardware families:

- `csd3`: Linux x86_64 CUDA wheel.
- `isambard`: Linux ARM64 CUDA wheel.

Unsupported systems should build JAXMg from source, then install BayesEoR-JAX without one of these extras.

## CSD3-Like Systems

The CSD3 wheel was built for a CUDA 12 style x86_64 environment. For source builds on CSD3, previous working builds used:

- `gcc/11`
- `cuda/12.1`
- `cudnn/8.9_cuda-12.1`

The CSD3 cuDNN module exposed versioned headers and libraries, so the source build needed a local compatibility directory for generic cuDNN header/library names. That is a JAXMg build issue, not a BayesEoR-JAX runtime setting.

For the recommended CSD3 conda environment, MultiNest shared-library setup, and one-GPU nested-sampling scripts, see [CSD3 Getting Started](csd3.md).

## Isambard-Like Systems

The Isambard wheel targets Linux ARM64 CUDA systems. Use:

```bash
pip install "bayeseor-jax[isambard] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

For multi-node JAX jobs, scripts may need distributed initialisation and NCCL environment settings appropriate to the cluster. Keep those settings in job scripts, not in portable package documentation.

## Practical Run Advice

- Run the example setup on an interactive GPU node before submitting a long sampler job.
- Use short output paths when running MultiNest.
- Keep matrix caches on fast project storage.
- Use distinct matrix directories for distinct model configurations.
- Record GPU model, driver, CUDA, JAX, JAXMg, and commit hash in run notes.
- Check that likelihood preparation and sampling use the intended GPU backend;
  physical-component construction deliberately runs on CPU.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
