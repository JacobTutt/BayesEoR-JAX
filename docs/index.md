# BayesEoR-JAX Documentation

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Docs Index](index.md) · [Original BayesEoR](https://github.com/PSims/BayesEoR)

BayesEoR-JAX is the JAX-compatible implementation and extension of BayesEoR.
The documentation is being built as a Markdown-first guide for users running
BayesEoR workflows on modern GPU systems.

## Current Pages

- [Installation](installation.md): environment setup, JAX runtime checks,
  JAXMg installation, and supported site-specific wheel installs.

## Planned Guide Structure

The documentation will grow around the main user workflow:

- configuration files and command-line entry points
- visibility input formats and preprocessing
- model-space and matrix construction
- priors, likelihoods, and latent-space precision models
- nested-sampling and JAX-native inference workflows
- example analyses using the included test data
- HPC notes for CSD3, Isambard, and other supported GPU systems

Research notes and derivations should live under dedicated subdirectories once
they are promoted into the public documentation. The top-level docs should stay
focused on installation, running analyses, and understanding the main pipeline.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Docs Index](index.md) · [Original BayesEoR](https://github.com/PSims/BayesEoR)
