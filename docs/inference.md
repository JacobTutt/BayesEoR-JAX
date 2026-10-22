# Inference

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX uses MultiNest with the analytically marginalised likelihood to infer power-spectrum parameters and evaluate Bayesian evidence.

## Marginalised Likelihood

The latent EoR and foreground coefficients are analytically marginalised. The sampler explores power-spectrum parameters, `theta`, which define the EoR prior covariance.

For each proposed `theta`, the likelihood forms:

```text
Sigma(theta) = T^H N^{-1} T + Phi^{-1}(theta)
```

where:

- `T^H N^{-1} T` is cached during setup.
- `Phi^{-1}(theta)` is the latent prior precision.
- `T^H N^{-1} d` and `d^H N^{-1} d` are also cached.

Evaluating the likelihood requires dense positive-definite linear algebra, including a Cholesky factorization, solves, and a log determinant. BayesEoR-JAX uses JAX and the downstream JAXMg branch for this path.

## MultiNest

MultiNest is the supported nested sampler. It is available directly through:

```python
from bayeseor_jax import run_multinest
```

The generic name `run_nestedsampling` is an alias for the same function:

```python
from bayeseor_jax import run_nestedsampling

summary = run_nestedsampling(
    setup_result,
    out_dir="outputs/nested_sampling",
)
```

MultiNest is often useful when likelihood evaluations are expensive and evidence estimates are required. It requires the native MultiNest library as well as `pymultinest`.

## MAP Helper

`MaximumAPosteriori` provides a maximum-a-posteriori analysis path. Use it for reconstruction and diagnostic workflows rather than as the default power-spectrum posterior estimator.

## Example Script

The shipped example runs setup by default:

```bash
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml
```

Add `--sample` to run nested sampling:

```bash
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml --sample
```

The example calls `run_multinest`. After sampling, it runs the example analysis step and writes posterior summaries and diagnostic plots. For serious runs, create a dedicated configuration and output directory rather than using the default example paths.

Record the live-point count, random seed and stopping settings with each scientific result.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
