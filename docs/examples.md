# Running the example scripts

The repository includes a single analysis runner and two simulated visibility
datasets. Run the commands below from the repository root after following
[installation](installation.md).

| Example | Configuration | Data |
| --- | --- | --- |
| Cosmological signal only | [eor/config.yaml](https://github.com/JacobTutt/BayesEoR-JAX/blob/main/test_examples/eor/config.yaml) | `test_examples/eor/eor.uvh5` |
| Cosmological signal and foregrounds | [eor_fgs/config.yaml](https://github.com/JacobTutt/BayesEoR-JAX/blob/main/test_examples/eor_fgs/config.yaml) | `test_examples/eor_fgs/eor_and_fgs.uvh5` |

Both use [run_analysis_example.py](https://github.com/JacobTutt/BayesEoR-JAX/blob/main/test_examples/run_analysis_example.py).
The input simulations are noiseless; their configurations specify `sigma` so
the loader adds synthetic noise. These are scientific examples, not minimal
memory benchmarks: matrix construction can still be expensive on a laptop.

## 1. Prepare an analysis without sampling

```bash
python test_examples/run_analysis_example.py \
  --config test_examples/eor/config.yaml \
  --array-save-directory outputs/eor/matrices \
  --output-dir outputs/eor
```

The default runs `run_setup`: it loads the data, constructs or reuses the
model matrices, and prepares the prior and marginalised likelihood. It does
not start MultiNest. Separate matrix and output directories keep distinct
analyses from overwriting or reusing incompatible products.

## 2. Recover the cosmological power spectrum

Add `--sample` to evaluate the posterior and Bayesian evidence:

```bash
python test_examples/run_analysis_example.py \
  --config test_examples/eor/config.yaml \
  --array-save-directory outputs/eor/matrices \
  --output-dir outputs/eor \
  --sample
```

Sampling requires both the native MultiNest library and PyMultiNest. The script
then writes posterior summaries and power-spectrum/posterior plots automatically.

## 3. Run joint cosmological and foreground inference

```bash
python test_examples/run_analysis_example.py \
  --config test_examples/eor_fgs/config.yaml \
  --model eor+fg \
  --array-save-directory outputs/joint/matrices \
  --output-dir outputs/joint \
  --sample
```

This uses the foreground model alongside the cosmological component while
sampling only the remaining hyperparameters. To use a preconstructed KL basis,
add `--foreground-kl-path path/to/T_fg_reduced.npy`; that operator must match
the observation, row ordering and foreground prior. Setup does not construct
the KL basis automatically. See [KL Reduced Basis](kl-reduced-basis.md).

## 4. Evaluate a foreground-only hypothesis

The same runner can analyse the joint dataset using only a foreground model:

```bash
python test_examples/run_analysis_example.py \
  --config test_examples/eor_fgs/config.yaml \
  --model fg \
  --array-save-directory outputs/foreground_only/matrices \
  --output-dir outputs/foreground_only \
  --sample
```

Here `--sample` enables the inference stage, but with fixed foreground priors
and noise no sampled parameters remain. The runner evaluates the evidence and
MAP coefficients analytically, without calling MultiNest or producing a
power-spectrum posterior. This example changes the hypothesis, not the dataset:
the input still contains a cosmological signal.

## Outputs and custom configurations

Sampler or analytic results are written under `output_dir/nested_sampling/`.
Sampled runs additionally produce `output_dir/analysis/`, containing
`posterior_summary.json`, `power_spectrum.png`, `posteriors.png` and
`power_spectrum_and_posteriors.png`. See [posterior analysis](outputs-analysis.md)
for reading and comparing results.

Copy an example YAML file to start a new analysis. Use the
[configuration reference](configuration.md) for available settings and
[Data](data.md) for input conventions. Give changed physical models, priors,
observations or datasets their own matrix/output directories; use `--clobber`
only when intentionally replacing existing products.
