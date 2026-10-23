# Running inference

With setup complete, BayesEoR explores the power-spectrum posterior with
[MultiNest](https://github.com/JohannesBuchner/MultiNest), an implementation of
nested sampling that allows simultaneous posterior exploration and Bayesian-evidence
evaluation. Each
likelihood call is a single [marginalised likelihood](marginalisation.md)
evaluation, so the sampler only ever sees the power-spectrum amplitudes.

## Choosing the model

`model` selects which components are fitted. It describes the inference model,
not the contents of the data, so the same dataset can be analysed with each.

`eor+fg`
:   The joint model, and the default.

    - **Operator:** $[\mathbf T_{21}\ \ \mathbf T'_{\rm FG}]$.
    - **Sampled:** the power-spectrum amplitudes, with MultiNest.

`eor`
:   The EoR alone.

    - **Operator:** $\mathbf T_{21}$.
    - **Sampled:** the power-spectrum amplitudes, with MultiNest.
    - **Note:** foreground settings are ignored, and `foreground_kl_path` is
      rejected.

`fg`
:   The foregrounds alone, as the null hypothesis for model comparison.

    - **Operator:** $\mathbf T'_{\rm FG}$, physical or
      [reduced](kl-reduced-basis.md).
    - **Sampled:** nothing. The evidence and MAP coefficients are evaluated in one
      solve; see [foreground-only evidence](map.md#foreground-only-evidence).
    - **Note:** `priors` and `fov_eor` are not needed.

Use a separate output directory for each model, so their results and metadata are
kept apart.

## Nested sampling

`run_nestedsampling` runs MultiNest on a completed setup, or evaluates the
foreground-only evidence directly when there is nothing to sample.

=== "Maths"

    With the coefficients marginalised, the remaining posterior and evidence are

    $$
    P(\boldsymbol\rho\mid\boldsymbol d,\mathcal M)
    =\frac{\mathcal L(\boldsymbol d\mid\boldsymbol\rho,\mathcal M)\,
    \pi(\boldsymbol\rho\mid\mathcal M)}{\mathcal Z_{\mathcal M}},
    \qquad
    \mathcal Z_{\mathcal M}
    =\int\mathcal L(\boldsymbol d\mid\boldsymbol\rho,\mathcal M)\,
    \pi(\boldsymbol\rho\mid\mathcal M)\,\mathrm d\boldsymbol\rho.
    $$

    Comparing $\mathcal Z$ between the joint and foreground-only models quantifies
    the support for a 21-cm signal. Applying the same comparison to signal-free
    data tests whether the EoR component is absorbing unmodelled foreground
    structure instead.

=== "Code"

    ```python
    from bayeseor_jax import BayesEoRParser, run_setup, run_nestedsampling

    config = BayesEoRParser().parse_args(["--config", "config.yaml"])
    setup = run_setup(config)
    summary = run_nestedsampling(
        setup,
        out_dir=f"{config.output_dir}/multinest",
        random_key=config.random_key,
        n_live_scale_factor=config.n_live_scale_factor,
        resume=config.resume,
    )
    ```

    The sampler settings below are read from the configuration and passed to this
    call; the [example script](#example-script) does this for you.

    MultiNest needs both the native library and `pymultinest`, which the
    [Conda environments](installation.md) install.

`n_live_scale_factor`
:   Sets the number of live points relative to the number of sampled parameters.

    - **Default:** 25, so live points $= 25\,N_{\rm dim}$.

`random_key`
:   The sampler's random seed.

    - **Default:** 0.

`resume`
:   Continues an existing MultiNest run from `output_dir`.

    - **Default:** `false`.
    - **Note:** reports an error if no checkpoint files are found.

`output_dir`
:   Where the sampler outputs and run metadata are written.

## Hardware

MultiNest runs in a single process, while each likelihood evaluation uses the
available JAX devices. On one CPU or GPU, the likelihood uses native JAX; across
several GPUs, it uses [JAXMg](https://flatironinstitute.github.io/jaxmg/latest/),
with one process per GPU. To restrict a run to one GPU, set
`CUDA_VISIBLE_DEVICES=0`. See [HPC and scaling](hpc-scaling.md).

## Example script

The example script runs setup, and with `--sample`, nested sampling followed by a
short posterior analysis:

```bash
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml --sample
```

For scientific runs, use a dedicated configuration and output directory, and
record the live-point count, seed and stopping settings with each result.
