# Inference

`MarginalisedLikelihood` evaluates the
[marginalised likelihood](../inference/marginalisation.md) for a set of power-spectrum
amplitudes, and `run_multinest` samples it. `MaximumAPosteriori` recovers the
[conditional coefficients](../inference/map.md) from the same products. All three are built
from the `SetupResult` returned by [`run_setup`](setup.md).

```python
from bayeseor_jax import run_nestedsampling

summary = run_nestedsampling(setup, out_dir="outputs/multinest")
log_l = setup.likelihood_class.log_likelihood(rho)
```

## Sampling

`run_nestedsampling` is an alias of `run_multinest`.

::: bayeseor_jax.inference.run_ns.run_multinest
    options:
      heading_level: 3

## Likelihood and MAP

::: bayeseor_jax.inference.likelihood_marg.MarginalisedLikelihood
    options:
      heading_level: 3

::: bayeseor_jax.inference.maxaposteriori.MaximumAPosteriori
    options:
      heading_level: 3

## Priors and noise

The power-spectrum prior, the [coefficient prior](../inference/power-spectrum.md#3-set-the-coefficient-prior)
and the [noise covariance](../inference/marginalisation.md#noise-covariance).

::: bayeseor_jax.inference.prior.Prior
    options:
      heading_level: 3

::: bayeseor_jax.inference.phi.PhiInverse
    options:
      heading_level: 3

::: bayeseor_jax.model.instrument.noise.build_noise_variance
    options:
      heading_level: 3
