# Bayesian forward model

In order to rigorously separate the components of the measured visibilities,
BayesEoR models each of them at its physical origin, predicting the visibilities
that a given sky would produce for comparison with the data. This section describes how the 21-cm signal and the foregrounds
are each modelled, and how both are mapped through the instrument into a single
model for the measured visibilities.

[![Combined forward model: the EoR and foreground components enter their respective Fourier and image spaces, then propagate through the shared beam and fringe response into modelled data.](assets/diagrams/joint-model-workflow.png#only-light)](assets/diagrams/joint-model-workflow.png)
[![Combined forward model: the EoR and foreground components enter their respective Fourier and image spaces, then propagate through the shared beam and fringe response into modelled data.](assets/diagrams/joint-model-workflow-dark.png#only-dark)](assets/diagrams/joint-model-workflow-dark.png)

## The interferometric measurement equation

At full fidelity, every direction on the sky and every frequency is propagated
into the data through the continuous measurement equation, which relates each
[visibility](visibilities.md) to the sky brightness, $I$, weighted by the primary
beam, $P$:

$$
V_{ij}(\nu,t)
=\int_{\Omega_{\rm vis}(t)}
P(\hat{\boldsymbol s},\nu)\,
I(\hat{\boldsymbol s},\nu,t)\,
\exp\!\left[-2\pi i\,\boldsymbol u_{ij}(\nu)\cdot\hat{\boldsymbol s}\right]
\mathrm d\Omega.
$$

BayesEoR's goal is to solve the inverse of this problem. However, parameterising
the brightness of every direction on the sky at every frequency would quickly
become computationally intractable, and would give no way to separate the
foregrounds from the cosmological signal. BayesEoR instead describes each
component in a basis suited to its physics, and maps both into the visibilities
through linear operators:

$$
\boldsymbol m = \mathbf T_{21}\,\boldsymbol a + \mathbf T_{\rm FG}\,\boldsymbol q,
\qquad
\boldsymbol d = \boldsymbol m + \boldsymbol n.
$$

## Overview

- [EoR model](eor.md): the 21-cm signal, $\boldsymbol a$, is described by Fourier
  coefficients on a three-dimensional grid, which expose its line-of-sight
  structure and connect directly to the power spectrum.
- [Foreground model](foregrounds.md): the foregrounds, $\boldsymbol q$, are
  described by smooth spectral amplitudes in each pixel of a spherical image grid,
  out to the horizon if required.
- [KL Reduced Basis](kl-reduced-basis.md): a latent compression, rooted in this
  physics, can be used to reduce the model before inference.
- [Construction and execution](forward-model.md): how the operators are built,
  stored and run.
