---
title: BayesEoR
---

<h1 class="bayeseor-wordmark" id="bayeseor-jax" aria-label="BayesEoR">
  <img class="brand-light" src="assets/branding/wordmark-light.svg" alt="" width="695" height="120">
  <img class="brand-dark" src="assets/branding/wordmark-dark.svg" alt="" width="695" height="120">
</h1>

[![Complete BayesEoR workflow: cosmological and foreground coefficients are mapped through the instrument into visibilities, followed by marginalisation, nested sampling, power-spectrum recovery and evidence evaluation.](assets/diagrams/workflow.png#only-light)](assets/diagrams/workflow.png)
[![Complete BayesEoR workflow: cosmological and foreground coefficients are mapped through the instrument into visibilities, followed by marginalisation, nested sampling, power-spectrum recovery and evidence evaluation.](assets/diagrams/workflow-dark.png#only-dark)](assets/diagrams/workflow-dark.png)

Recovering a cosmological 21-cm signal requires more than a flexible description
of the measured data. Foreground emission and the cosmological signal are both
propagated through the chromatic instrumental response, producing visibility
structures that can be difficult to distinguish. A model must therefore describe
not only these contributions, but also the uncertainty in their separation.

## Physically motivated Bayesian forward modelling

The BayesEoR philosophy is to parameterise the physical sky, propagate it through
the interferometric measurement equation, and infer the cosmological power spectrum
while accounting jointly for foreground uncertainty. The data model is

$$
\boldsymbol d
= \underbrace{\mathbf T_{21}\boldsymbol a}_{\text{cosmological signal}}
+ \underbrace{\mathbf T_{\rm FG}\boldsymbol q}_{\text{foreground emission}}
+ \boldsymbol n.
$$

Here, $\boldsymbol a$ describes the cosmological field in Fourier space,
$\boldsymbol q$ describes foreground brightness in the spherical image domain,
and the operators $\mathbf T_{21}$ and $\mathbf T_{\rm FG}$ propagate their
contributions into the measured visibilities. Priors on these coefficients
describe physical uncertainty rather than simply supplying additional freedom
to fit the data.

BayesEoR-JAX preserves this connection to the underlying sky while making the
calculation tractable. Linear sky coefficients are analytically marginalised,
and a prior- and noise-informed Karhunen–Loève (KL) basis can reduce the foreground
coefficient space before inference. The sampled cosmological power-spectrum
amplitudes determine the signal prior, while Bayesian-evidence evaluation
supports comparison and validation of competing models.

## Complete workflow

1. **Define the physical components.** Choose the cosmological Fourier grid and
   foreground image-domain amplitudes, together with their priors.
2. **Propagate the sky into visibilities.** Apply the spectral, image, primary-beam
   and three-dimensional fringe operators for the selected observation.
3. **Represent foreground uncertainty efficiently.** Optionally construct a KL
   basis retaining its noise-relevant prior-predictive covariance.
4. **Form the joint likelihood.** Combine the component operators and noise model,
   then analytically marginalise the linear coefficients.
5. **Infer and assess the model.** Recover the power-spectrum posterior and Bayesian
   evidence, and inspect the reconstructed visibilities and residuals.

## Follow the workflow

- [Data](visibilities.md): interferometric measurements, instrument configuration and visibility inputs.
- [Model overview](model.md): the shared measurement equation and operator conventions.
- [EoR model](eor.md): from $(u,v,\eta)$ coefficients to cosmological visibilities.
- [Foreground model](foregrounds.md): from image-domain spectral amplitudes to wide-field visibilities.
- [KL Reduced Basis](kl-reduced-basis.md): from physical foreground uncertainty to a reduced operator.
- [Inference](joint-inference.md): parameterisation, marginalisation, posterior sampling and evidence evaluation.
- [Examples](examples.md): runnable scripts for setup and inference.

## Get started

Begin with [installation](installation.md) and the [example scripts](examples.md),
then use the [configuration reference](configuration.md) to define an analysis.
The [Python API](api/index.md) links the mathematical steps to their implementations.

Native JAX supports CPU and single-GPU inference. Optional JAXMG routines provide
distributed GPU calculations where needed; see [computing and scaling](hpc-scaling.md).
