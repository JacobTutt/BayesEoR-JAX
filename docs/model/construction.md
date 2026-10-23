# Construction and execution

Setup turns a configuration and its data into the arrays used by the inference.
It builds the forward operators described in the previous pages, saves them to
disk, and precomputes the fixed matrix products needed by the likelihood. As this
happens once, before sampling, a setup run is also a quick way to check a
configuration.

## Setup flow

`run_setup(config)` performs the following steps without modifying `config`.
Pass `build_foreground_kl=True` to construct a KL basis at the configured
`foreground_kl_path` using the resolved physical model and noise; otherwise
an existing basis is loaded unchanged:

1. Load and prepare the [visibility data](../data/visibility-data.md).
2. Resolve the model geometry: the image pixels, the Fourier grid and, when the
   EoR is included, its $k$ bins.
3. Build or reuse $\mathbf T_{21}$ and $\mathbf T_{\rm FG}$, or load a
   [reduced basis](kl-reduced-basis.md), and assemble the joint operator,
   $\mathbf T = [\mathbf T_{21}\ \ \mathbf T_{\rm FG}]$.
4. Build the inverse [noise covariance](../inference/marginalisation.md#noise-covariance).
5. Precompute the likelihood products.
6. Construct the priors and the marginalised likelihood, and return a
   `SetupResult` ready for sampling.

Because the sky coefficients are marginalised analytically, the likelihood depends
on the data and operators only through a few fixed matrix products, which are
computed here once rather than at every sampler step. Their mathematical
formulation is presented in [analytical marginalisation](../inference/marginalisation.md).

## Saved arrays

Setup writes its arrays to `array_save_directory`, and reuses them on later runs
where possible.

`T_21.npy`, `T_fg.npy`
:   The EoR and foreground operators.

    - **Shape:** $(N_{\rm d}, N_{21})$ and $(N_{\rm d}, N_{\rm FG})$.
    - **Reuse:** kept whenever present, unless `clobber` is set.

`T.npy`
:   The joint operator, assembled from the saved component columns.

    - **Shape:** $(N_{\rm d}, N_{21} + N_{\rm FG})$.

`Ninv.npy`
:   The inverse noise variance of each visibility.

    - **Shape:** $(N_{\rm d},)$.
    - **Reuse:** rebuilt whenever the noise model changes.

`T_Ninv_T.npy`, `dbar.npy`, `d_ninv_d.npy`
:   The likelihood products, $\mathbf T^\dagger\mathbf N^{-1}\mathbf T$,
    $\mathbf T^\dagger\mathbf N^{-1}\boldsymbol d$ and
    $\boldsymbol d^\dagger\mathbf N^{-1}\boldsymbol d$.

    - **Reuse:** the first is rebuilt if `T.npy` or `Ninv.npy` change; the two
      data products are recomputed on every run.

`array_save_directory`
:   Where the arrays are written.

    - **Default:** `./matrices/`.

`clobber`
:   Rebuilds every array instead of reusing it.

    - **Default:** `false`.

!!! warning

    Reuse checks that the saved files exist and are consistent with each other,
    but not that they came from the same configuration. When the observation or
    model settings change, use a new `array_save_directory` or set `clobber`.

## Execution

The two expensive stages of setup run on different hardware.

**Operator construction** runs on CPU. The time-frequency blocks of each operator
are divided between the JAX processes, which all write to the same file, so every
process needs the same shared filesystem. Each process uses the CPUs available to
it, respecting CPU affinity and Slurm's per-task count. The job launcher, not
BayesEoR, is responsible for starting the processes and setting thread counts.

**Likelihood products and inference** run on the default JAX devices, normally
GPUs. A single device uses native JAX; across several GPUs,
[JAXMg](https://flatironinstitute.github.io/jaxmg/latest/) requires one Python
process per GPU, with distributed JAX initialised before setup. The
[KL decomposition](kl-reduced-basis.md) follows the same rule.

See [HPC and scaling](../technical/hpc-scaling.md) for memory requirements and practical run
advice.

## Using the operators directly

The operators can also be built outside `run_setup`, either as arrays in memory
or as saved files.

=== "In memory"

    `T_21` and `T_fg` evaluate a supplied block of the operator and return a JAX
    array. They do not read files, choose hardware or apply priors, so they can be
    used inside compiled code.

    ```python
    import jax
    from bayeseor_jax.model.operators import T_21, T_fg

    # Inputs already selected for matching times, frequencies and baselines.
    T21_block = jax.jit(T_21)(lmn_21, uvw_lambda, beam_21, pixel_area_sr, F_prime, F_z)
    Tfg_block = jax.jit(T_fg)(lmn_fg, uvw_lambda, beam_fg, pixel_area_sr, Q)
    ```

    Here `lmn` is `(time, 3, pixel)`, `uvw_lambda` is `(frequency, 3, baseline)`
    and the beam is `(time, frequency, pixel)`.

=== "Disk-backed"

    The `_build_save` functions evaluate the same operators on CPU in
    time-frequency blocks and return the paths of the saved matrices. Every
    participating process makes the same calls.

    ```python
    from bayeseor_jax.model.construction.operator_construction import (
        T_21_build_save, T_fg_build_save, assemble_joint_operator,
    )

    path_21 = T_21_build_save(
        lmn_21, uvw_lambda, beam_21, pixel_area_sr, F_prime, F_z,
        output_path="arrays/T_21.npy",
    )
    path_fg = T_fg_build_save(
        lmn_fg, uvw_lambda, beam_fg, pixel_area_sr, Q,
        output_path="arrays/T_fg.npy",
    )
    path_joint = assemble_joint_operator(path_21, path_fg, output_path="arrays/T.npy")
    ```

    `BuildMatrices` provides the same methods, using its configured observation
    and model.
