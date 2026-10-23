# Inference

BayesEoR accelerates joint posterior inference and Bayesian-evidence evaluation by
analytically marginalising over the individual sky coefficients, and sampling only
the power spectrum of the 21-cm field. These few power-spectrum amplitudes define
the prior on the coefficients in each $k$ bin, so the sampler explores tens of
parameters rather than tens of thousands.

[![Inference workflow: modelled data and observation, likelihood, prior covariance, marginalised likelihood, nested sampling, and power-spectrum and evidence outputs.](../assets/diagrams/inference-workflow.png#only-light)](../assets/diagrams/inference-workflow.png)
[![Inference workflow: modelled data and observation, likelihood, prior covariance, marginalised likelihood, nested sampling, and power-spectrum and evidence outputs.](../assets/diagrams/inference-workflow-dark.png#only-dark)](../assets/diagrams/inference-workflow-dark.png)

## Overview

- [Δ² parameterisation](power-spectrum.md): how the EoR coefficients are grouped
  into $k$ bins, whose power-spectrum amplitudes set their prior and are the
  parameters sampled.
- [Analytical marginalisation](marginalisation.md): integrating out the
  coefficients to give a likelihood for the power spectrum alone.
- [Maximum a posteriori](map.md): recovering the coefficients and model
  visibilities for a fixed prior, and the foreground-only evidence.
- [Running inference](running.md): choosing the model and exploring the
  posterior and evidence with nested sampling.
- [Posterior analysis](posterior-analysis.md): reading, summarising and plotting
  the results.
