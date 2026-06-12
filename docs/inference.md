# Inference

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX currently exposes several inference paths. The main production path is the marginalised likelihood for power-spectrum parameters, with MultiNest used by default to estimate posterior distributions and Bayesian evidence.

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

MultiNest is the default nested-sampling backend. It is available directly through:

```python
from bayeseor_jax import run_multinest
```

or through the backend selector:

```python
from bayeseor_jax import run_nestedsampling

summary = run_nestedsampling(
    setup_result,
    out_dir="outputs/nested_sampling",
)
```

MultiNest is often useful when likelihood evaluations are expensive and evidence estimates are required. It requires the native MultiNest library as well as `pymultinest`.

## BlackJAX Nested Sampling

The BlackJAX nested-sampling backend is experimental for the marginalised JAXMg likelihood. The current JAXMg FFI solve/log-determinant path is scalar and can fail under BlackJAX batching transforms such as `vmap`. Use MultiNest for the production pipeline.

The backend remains available for development through:

```python
from bayeseor_jax import run_blackjaxns
```

or:

```python
summary = run_nestedsampling(
    setup_result,
    out_dir="outputs/nested_sampling",
    sampler="blackjaxns",
)
```

Important controls are:

- `n_live_scale_factor`
- `num_inner_scale_factor`
- `delete_scale_factor`
- `stop_threshold`
- `random_key`

BlackJAX outputs are written in an Anesthetic-style CSV file named `nested_sampling_results.csv`, together with `summary.json`.

## HMC And NUTS

The package also exposes:

- `run_full_nuts`
- `run_marginalised_nuts`

These are useful for development and high-dimensional posterior experiments. They should be documented as advanced workflows until their recommended configuration and outputs are standardised.

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

The example uses the package-level `run_nestedsampling` selector, which defaults to MultiNest. After sampling, it runs the example analysis step and writes posterior summaries and diagnostic plots. For serious runs, create a dedicated configuration and output directory rather than using the default example paths.

## Choosing A Sampler

Use MultiNest when:

- you need a mature nested-sampling backend;
- evidence estimates are central to the analysis;
- you are running the JAXMg-backed marginalised likelihood.

Use BlackJAX nested sampling when:

- you want a JAX-native experimental backend;
- you are comparing sampler behaviour;
- you are working on likelihood paths that are compatible with JAX batching transforms.

Document the sampler, live-point settings, random seed, and stopping criterion in any scientific result.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
