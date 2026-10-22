# Inference

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX uses MultiNest with the analytically marginalised likelihood to infer power-spectrum parameters and evaluate Bayesian evidence.

The sampler runner and conditional MAP solver live in `inference/run_ns.py` and
`inference/maxaposteriori.py`, alongside the priors and likelihood. They are
available from `bayeseor_jax.inference`; the existing top-level imports from
`bayeseor_jax` remain supported. Direct imports from the former root-level modules
must use these new paths.

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

### Complex storage and the factors of one half

The visibility loader appends the conjugate of each measured visibility, together
with its reversed baseline. Thus `d = (v, v.conj())`, up to row ordering, has
`N_d = 2*m` complex entries but only `m` independent complex measurements.
Noise variance means `E[abs(n)**2]`; each independent real or imaginary component
has half that variance. Conjugate rows carry the same noise variance.

Consequently, `d^H N^{-1} d` counts each independent measurement twice. The
correct Gaussian exponent is `-0.5*d^H N^{-1} d`, not `-d^H N^{-1} d`.
Complex storage does not make the conjugate entries independent.

The foreground brightness coefficients are real; cosmological Fourier partners
obey `a(-k) = a(k).conj()` with equal prior variances. Writing the calculation in
independent real coordinates gives the same marginalised target:

```text
log L = -0.5 * (log|Sigma| + log|Phi|
               + d^H N^{-1} d - dbar^H Sigma^{-1} dbar)
        -0.5 * (log|N| + N_d*log(pi))
```

Here the density is with respect to the real and imaginary components of the
`m` independent visibilities. KL reduction preserves this result through the
induced visibility covariance; it does not turn the latent sky into an
independent proper-complex field. Supplying independent complex observations
without their conjugate counterparts requires a different formulation, not
simply the same cached products. Saved `.npy` data bypass raw-data preprocessing,
so their convention must already match their operators and noise weights.

### Evidence normalisation

For fixed noise, `log_likelihood` returns the expression above without

```text
c_fixed = -0.5 * (d^H N^{-1} d + log|N| + N_d*log(pi)).
```

Add `c_fixed` to the reported log evidence to recover its absolute normalisation.
It cancels when comparing foreground-only and joint models of the same dataset
with the same noise covariance and prior convention, even if their coefficient
counts differ. It does not generally cancel between different datasets.

When fitting a noise amplitude `alpha`, the code retains the varying data and
noise-determinant terms. Its omitted constant is instead
`c_noise = -0.5*(log|N| + N_d*log(pi))`, where `N` is the reference covariance
before multiplication by `alpha**2`. Restore the appropriate constant before
comparing fixed-noise and fitted-noise models. These statements assume the
normalised log-uniform power priors; the legacy `uniform_priors` Jacobian option
also requires its prior-normalisation constants for absolute evidence.

The regression tests compare the original and KL coefficient-space targets
against a normalised Gaussian density on independent real/imaginary data,
including truncated bases, foreground-only models and fitted noise amplitudes.

### Residual-noise checks

For independent Gaussian noise before any fitting or sample rescaling, the
augmented statistic `Q = n^H N^{-1} n` follows
`2*Gamma(N_d/2, 1)`, equivalently `chi2(N_d)`, not `Gamma(N_d, 1)`.
Both expressions have mean `N_d`, but their variances differ by a factor of two.
The existing noise generator also rescales each draw by its measured standard
deviation before enforcing conjugacy, so its finite-sample distribution is not
exactly this Gaussian reference. A calibrated residual test must account for
that generation procedure and the subsequent fitting; likelihood algebra alone
does not validate a residual-tail probability.

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
