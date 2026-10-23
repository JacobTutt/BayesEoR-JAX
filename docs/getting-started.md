# Getting started

The quickest way to get started is to run one of the two examples. Each infers a
21-cm power spectrum from a small simulated observation, and takes a few minutes
on a laptop.

## 1. Install

Create and activate the CPU environment, as described in
[installation](installation.md):

```bash
conda env create -f environment.yml
conda activate bayeseor
```

## 2. Run an example

From the repository root, run either example:

=== "21-cm only"

    ```bash
    python -m examples.eor.run --sample
    ```

    Infers the power spectrum of a simulated 21-cm signal, with no foregrounds.
    See [21-cm only](examples/eor.md).

=== "21-cm and foregrounds"

    ```bash
    python -m examples.eor_fgs.run --sample
    ```

    Adds bright foregrounds, compresses them into a KL reduced basis, and infers
    the power spectrum jointly. See [21-cm and foregrounds](examples/eor-fgs.md).

Without `--sample`, each example stops after setup, which is a quick check that
the installation works.

## 3. Look at the results

Each example writes its power-spectrum and posterior figure to its `plots/`
folder, for example `examples/eor/plots/power_spectrum_and_posteriors.png`, and
its full outputs under `outputs/`. See
[posterior analysis](inference/posterior-analysis.md) for reading them yourself.

## Next steps

To analyse your own data, start from an example's `config.yaml` and change its
settings:

- [Data](data/index.md): point it at your visibilities and describe the
  instrument and observation.
- [Model](model/index.md): choose the EoR and foreground fields and resolution.
- [Inference](inference/index.md): set the power-spectrum priors and run the
  sampler.
