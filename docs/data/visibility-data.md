# Visibility data

Set `data_path` to the visibilities to analyse. The loader reads them, converts
them to the analysis units and arranges them in the same order as the rows of the
forward model. It does not calibrate the data or simulate a sky, and the
[instrumental](instrument.md) and [observation](observation.md) settings must
describe the same data.

## Input formats

The reader is chosen from the file suffix. Simulated and observed visibilities are
handled in the same way.

=== "Visibility files (.uvh5, .uvfits, .ms)"

    ```yaml
    data_path: inputs/observation.uvh5
    ```

    Files are read with pyuvdata, so `.ms` Measurement Sets need pyuvdata's
    optional `casacore` dependency. The data should be:

    - **Calibrated**, in Jy or K sr. They are converted to mK sr, using the
      Rayleigh–Jeans relation for each channel.
    - **Drift-scan**, with the same baselines at every selected time. Phased data
      are unprojected before use.
    - **Flagged with care.** Fully flagged baselines are removed, but partial
      flags are **not** turned into per-visibility weights, so check flagging
      before inference.

    BayesEoR's sky model is scalar, so the data are reduced to one polarisation:

    `form_pi`
    :   Forms pseudo-Stokes I visibilities.

        - **Default:** `true`.
        - **Requires:** XX and YY in the file, or an existing pI polarisation,
          which is kept as it is.

    `pi_norm`
    :   The normalisation in $\mathrm{pI} = N(\mathrm{XX} + \mathrm{YY})$.

        - **Default:** 1, not one half.

    `pol`
    :   The polarisation to keep when `form_pi: false`.

        - **Default:** `xx`.
        - **Note:** the same setting selects the polarisation of a
          [beam file](instrument.md#primary-beam).

    `save_vis`
    :   Writes the prepared vector to `save_dir` as `vis_vector.npy`, plus
        `noise_vector.npy` if `calc_noise` is set.

        - **Default:** `false`.
        - **Reuse:** with `save_model: true`, the outputs can be passed back in
          as a `.npy` input.

=== "Preprocessed vectors (.npy)"

    A `.npy` file holds the complex data vector itself, already in **mK sr** and
    in the [data vector ordering](#data-vector-ordering) below. It can be a plain
    array, or a BayesEoR dictionary with the vector under `data`. Dictionary files
    are loaded with pickle, so only load them from trusted sources.

    The file carries no metadata, so the instrument model and observation axes
    must be given in full:

    ```yaml
    data_path: inputs/visibilities.npy
    inst_model: inputs/instrument/
    n_freq: 38
    f0_mhz: 158.304048743
    df_mhz: 0.23761898685801028
    n_t: 1
    d_t: 10.0
    jd_center: 2461040.5
    ```

    See [array layout](instrument.md#array-layout) for `inst_model`, and
    [observation configuration](observation.md) for the axes. Metadata stored
    inside a dictionary file are not used.

## Data vector ordering

The data are a single complex vector with time varying slowest, then frequency,
then baseline. Within each time and frequency, the $N_{\rm bls}$ original
baselines are followed by their reversed copies, which hold the conjugate
visibilities:

$$
\boldsymbol d_{tf}
=\begin{pmatrix}\boldsymbol v_{tf}\\\boldsymbol v_{tf}^{*}\end{pmatrix},
\qquad
\text{row} = (t\,N_\nu + f)\,2N_{\rm bls} + b.
$$

The vector therefore has length $2N_tN_\nu N_{\rm bls}$. The conjugate copies are
not independent measurements; the
[marginalised likelihood](../inference/marginalisation.md#3-evaluate-in-coefficient-space)
accounts for this. Visibility files are arranged this way automatically, but a
`.npy` vector, its noise vectors and the [instrument model](instrument.md#array-layout)
must already follow it.

## Noise

The loader decides whether the data already contain noise. The first option that
applies is used:

`noise_data_path`
:   A noise vector for data that already contain noise.

    - **Effect:** the data are used unchanged.
    - **Format:** a `.npy` vector in mK sr, in the data vector ordering.

`calc_noise`
:   Estimates the noise of a visibility file from differences between even and
    odd integrations.

    - **Default:** `false`.
    - **Effect:** the data are used unchanged.
    - **Requires:** a visibility file with more than one integration; it has no
      effect on `.npy` inputs.

`sigma` and `noise_seed`
:   If neither of the above is set, the data are treated as noiseless and a
    Gaussian noise realisation is added.

    - **`sigma`:** the complex noise RMS per visibility before redundancy
      averaging, in mK sr. Each baseline's noise is scaled by $1/\sqrt{R}$, and
      the reversed copies receive the conjugate noise.
    - **`noise_seed`:** the random seed. Defaults to `742123`.
    - **Note:** `sigma: 0` adds no noise.

!!! warning

    Without `noise_data_path` or `calc_noise`, noise **is added**. Do not leave
    them unset for measurements that already contain noise.

This only determines the data. The noise covariance that weights the likelihood
is set separately; see [noise covariance](../inference/marginalisation.md#noise-covariance).
