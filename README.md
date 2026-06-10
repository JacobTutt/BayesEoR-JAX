# BayesEoR-JAX

[Overview](#overview) | [Installation](docs/installation.md) | [Documentation](docs/index.md) | [Original BayesEoR](https://github.com/PSims/BayesEoR)

## Overview

BayesEoR-JAX is a JAX-compatible development version of
[BayesEoR](https://github.com/PSims/BayesEoR), a Bayesian pipeline for
estimating the redshifted 21-cm power spectrum from interferometric data. This
version keeps the BayesEoR modelling workflow while moving the computationally
expensive linear-algebra path onto JAX-compatible GPU infrastructure.

The project is intended for large GPU systems where matrix construction,
likelihood evaluation, and posterior sampling need to be integrated with modern
JAX workflows. The current package is under active development and should be
treated as the JAX implementation and extension of the original BayesEoR code
rather than a drop-in replacement for every legacy workflow.

## Documentation

- [Installation](docs/installation.md): Python environments, JAX, JAXMg, and
  site-specific wheel installs for CSD3 and Isambard.
- [Documentation index](docs/index.md): the current documentation map and
  planned user-guide structure.
- [Original BayesEoR](https://github.com/PSims/BayesEoR): the reference
  implementation and methodology background.

## Quick Install

For a standard development install without a prebuilt JAXMg wheel:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e .
```

JAXMg is required by the current JAX likelihood path. On systems where a
prebuilt JAXMg wheel is available, install the matching optional extra:

```bash
pip install -e ".[csd3]"
```

or:

```bash
pip install -e ".[isambard]"
```

See [Installation](docs/installation.md) for full source-build and direct
GitHub install commands.

## Citation

Users should cite the relevant BayesEoR methodology papers associated with the
original project. Formal citation guidance for BayesEoR-JAX will be expanded as
the JAX implementation stabilizes.

[Overview](#overview) | [Installation](docs/installation.md) | [Documentation](docs/index.md) | [Original BayesEoR](https://github.com/PSims/BayesEoR)
