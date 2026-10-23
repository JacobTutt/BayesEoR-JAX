---
title: BayesEoR
---

<h1 class="bayeseor-wordmark" id="bayeseor-jax" aria-label="BayesEoR">
  <img class="brand-light" src="assets/branding/wordmark-light.svg" alt="" width="695" height="120">
  <img class="brand-dark" src="assets/branding/wordmark-dark.svg" alt="" width="695" height="120">
</h1>

**BayesEoR** is a physically motivated forward-modelling framework for Bayesian
recovery of the 21-cm power spectrum from interferometric visibilities. Every
contribution to the data is modelled jointly and propagated through the instrument
response, so the recovered power spectrum and Bayesian evidence reflect their
combined uncertainty.

This version is written in JAX and runs on CPU or GPU, with optional distributed GPU linear algebra through
[JAXMg](https://flatironinstitute.github.io/jaxmg/latest/).

[![The BayesEoR workflow: cosmological and foreground coefficients are mapped through the instrument into visibilities, followed by marginalisation, nested sampling, power-spectrum recovery and evidence evaluation.](assets/diagrams/workflow.png#only-light)](assets/diagrams/workflow.png)
[![The BayesEoR workflow: cosmological and foreground coefficients are mapped through the instrument into visibilities, followed by marginalisation, nested sampling, power-spectrum recovery and evidence evaluation.](assets/diagrams/workflow-dark.png#only-dark)](assets/diagrams/workflow-dark.png)

## How BayesEoR works

In interferometric visibilities, the 21-cm signal is a faint, low signal-to-noise
contribution, buried beneath Galactic and extragalactic foregrounds roughly
$10^4$–$10^5$ times brighter. The chromatic response of the instrument then maps
both into highly degenerate structures, making its extraction a formidable
inference problem. Any foreground structure a model fails to describe can
be absorbed by the signal and mistaken for a detection, so both must be inferred
together within a joint posterior, given by Bayes' theorem:

$$
P(\boldsymbol\theta\mid\boldsymbol d,\mathcal M)
= \frac{\mathcal L(\boldsymbol d\mid\boldsymbol\theta,\mathcal M)\,
        \pi(\boldsymbol\theta\mid\mathcal M)}
       {\mathcal Z(\boldsymbol d\mid\mathcal M)}.
$$

Unlike most other approaches, BayesEoR evaluates the Bayesian evidence,
$\mathcal Z$, alongside the posterior. Comparing evidences between models with and
without a 21-cm component quantifies the statistical significance of a detection,
and allows the framework to be rigorously validated against spurious detections.

## Using these docs

- [Installation](installation.md) and [Getting started](basics.md): set up the
  package and run a first analysis.
- [Data](data/index.md): the visibilities, instrument and observation inputs.
- [Model](model/index.md): how the 21-cm signal and foregrounds are mapped into
  visibilities.
- [Inference](inference/index.md): marginalisation, sampling, and recovering the
  power spectrum and evidence.
- [Examples](examples/index.md): runnable scripts and the configuration reference.
- [API](api/index.md): the Python reference, generated from the source.
