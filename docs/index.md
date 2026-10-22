# BayesEoR-JAX

BayesEoR-JAX provides physically motivated Bayesian inference of the cosmological
21-cm power spectrum from interferometric visibilities. It combines a cosmological
Fourier model with spherical image-domain foregrounds and the instrumental response:

$$
\boldsymbol{d}
= \mathbf{T}_{21}\boldsymbol{a}
+ \mathbf{T}_{\rm FG}\boldsymbol{q}
+ \boldsymbol{n}.
$$

The linear coefficients are analytically marginalised, leaving the power-spectrum
amplitudes to be sampled with MultiNest. A prior- and noise-informed
Karhunen–Loève (KL) reduction compresses the foreground representation before inference.

## Get started

Start with [installation](installation.md) and the [workflow guide](basics.md).
The [configuration reference](configuration.md) describes the available settings,
and the [API reference](api/index.md) documents the Python functions and classes.

## Workflow

1. Configure the observation, instrument, cosmological grid and foreground prior.
2. Construct the cosmological and foreground visibility operators.
3. Optionally reduce the foreground operator using its noise-weighted prior covariance.
4. Assemble the joint operator and marginal-likelihood products.
5. Recover the power-spectrum posterior and Bayesian evidence, then inspect the results.

The pipeline supports cosmological-only, foreground-only and joint analyses.
Foreground-only models with fixed priors have an analytically evaluated evidence
and require no posterior sampling.

## Computation

Native JAX supports CPU and single-GPU calculations. The optional JAXMG backend
supports distributed GPU calculations; using a GPU does not require JAXMG.

See the [forward model](forward-model.md), [inference guide](inference.md) and
[computing guide](hpc-scaling.md) for the mathematical construction and execution options.
