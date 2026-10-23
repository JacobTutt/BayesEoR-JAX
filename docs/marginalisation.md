# Analytical marginalisation

The forward model is linear in the sky coefficients, and both their prior and the
noise are Gaussian, so the coefficients can be integrated out exactly. This leaves
a likelihood for the power spectrum alone, in which the uncertainty in every
coefficient, and the degeneracies between the EoR and foregrounds, are fully
propagated.

## 1. Combine the components

The [EoR](eor.md) and [foreground](foregrounds.md) contributions combine into a
single linear model over the joint coefficient vector, $\boldsymbol b$, with a
block-diagonal Gaussian prior.

=== "Maths"

    $$
    \boldsymbol m
    =\underbrace{\begin{bmatrix}\mathbf T_{21}&\mathbf T'_{\rm FG}\end{bmatrix}}_{\mathbf T}
    \underbrace{\begin{pmatrix}\boldsymbol a\\\boldsymbol q'\end{pmatrix}}_{\boldsymbol b},
    \qquad
    \mathbf\Psi(\boldsymbol\varphi)
    =\begin{pmatrix}
    \mathbf\Phi(\boldsymbol\varphi)&\mathbf 0\\
    \mathbf 0&\mathbf\Sigma'_{\rm FG}
    \end{pmatrix}.
    $$

    $\mathbf T'_{\rm FG}$ is either the physical foreground operator, with prior
    $\mathbf\Sigma'_{\rm FG}=\mathbf\Sigma_q$, or its
    [reduced basis](kl-reduced-basis.md), with $\mathbf\Sigma'_{\rm FG}=\mathbf I$.
    Only the EoR block, $\mathbf\Phi$, depends on the
    [power spectrum](power-spectrum.md); the foreground prior is fixed.

=== "Code"

    Setup assembles $\mathbf T$ as `T.npy`, and the diagonal of
    $\mathbf\Psi^{-1}(\boldsymbol\varphi)$ is returned by
    [PhiInverse](api/inference.md#bayeseor_jax.inference.phi.PhiInverse). The
    foreground widths are `foreground_prior_std_mk` for the physical basis, or ones
    for a reduced basis.

## 2. The prior-predictive covariance

Integrating over $\boldsymbol b$ leaves a Gaussian likelihood for the data, whose
covariance combines the noise with the uncertainty each component induces in the
visibilities.

=== "Maths"

    $$
    \mathcal L(\boldsymbol d\mid\boldsymbol\varphi)
    =\int\mathcal L(\boldsymbol d\mid\boldsymbol b)\,
    \pi(\boldsymbol b\mid\boldsymbol\varphi)\,\mathrm d\boldsymbol b
    =\mathcal{CN}\big(\boldsymbol d;\,\boldsymbol 0,\,\mathbf C(\boldsymbol\varphi)\big),
    $$

    $$
    \mathbf C(\boldsymbol\varphi)
    =\mathbf N
    +\underbrace{\mathbf T_{21}\mathbf\Phi(\boldsymbol\varphi)\mathbf T_{21}^\dagger}_{\text{EoR}}
    +\underbrace{\mathbf T'_{\rm FG}\mathbf\Sigma'_{\rm FG}\mathbf T_{\rm FG}'^\dagger}_{\text{foregrounds, fixed}}.
    $$

    The foreground term is the same for every proposed power spectrum. Its structure
    is what the [KL reduced basis](kl-reduced-basis.md) compresses.

=== "Code"

    The $N_{\rm d}\times N_{\rm d}$ covariance is never built. The equivalent
    coefficient-space form below is used instead.

## 3. Evaluate in coefficient space

The same likelihood can be written in terms of a system the size of the
coefficient vector. Its fixed parts are computed once during
[setup](forward-model.md#saved-arrays), so each proposed power spectrum needs only
one Cholesky factorisation.

=== "Maths"

    With the fixed products

    $$
    \mathbf G=\mathbf T^\dagger\mathbf N^{-1}\mathbf T,
    \qquad
    \overline{\boldsymbol d}=\mathbf T^\dagger\mathbf N^{-1}\boldsymbol d,
    \qquad
    Q=\boldsymbol d^\dagger\mathbf N^{-1}\boldsymbol d,
    $$

    each proposal forms and solves

    $$
    \mathbf\Sigma(\boldsymbol\varphi)
    =\mathbf G+\mathbf\Psi^{-1}(\boldsymbol\varphi),
    \qquad
    \mathbf\Sigma(\boldsymbol\varphi)\,\widehat{\boldsymbol b}
    =\overline{\boldsymbol d},
    $$

    giving the log likelihood

    $$
    \ln\mathcal L
    =-\frac12\left[
    \ln\det\mathbf\Sigma+\ln\det\mathbf\Psi
    -\overline{\boldsymbol d}^{\dagger}\widehat{\boldsymbol b}\right]
    +c_{\rm fixed},
    \qquad
    c_{\rm fixed}=-\frac12\left[Q+\ln\det\mathbf N+N_{\rm d}\ln\pi\right].
    $$

    The factors of one half arise because $\boldsymbol d$ holds each visibility and
    its conjugate, so it contains only $N_{\rm d}/2$ independent measurements (see
    [data vector ordering](data.md#data-vector-ordering)). The single Cholesky
    factorisation of $\mathbf\Sigma$ supplies both the solve and the determinant,
    and is the dominant cost of each evaluation.

=== "Code"

    [MarginalisedLikelihood](api/inference.md#bayeseor_jax.inference.likelihood_marg.MarginalisedLikelihood)
    evaluates this for a vector of log amplitudes, omitting $c_{\rm fixed}$:

    ```python
    likelihood = setup.likelihood_class
    log_l = likelihood.log_likelihood(rho)
    ```

    On one CPU or GPU it uses a native JAX Cholesky factorisation; across several
    GPUs it uses JAXMg.

## Configuration

The likelihood is set by the two fixed covariances that are not sampled: the
noise, $\mathbf N$, and the foreground prior, $\mathbf\Sigma_q$.

### Noise covariance

The likelihood assumes Gaussian thermal noise with a diagonal covariance,
$\mathbf N$, whose entries are the variance $\langle|n|^2\rangle$ of each complex
visibility. It is set by one of:

`sigma`
:   The default model, with variance $N_{tfb} = \sigma^2/R_{tb}$ for a baseline of
    redundancy $R_{tb}$ (see [array layout](instrument.md#array-layout)).

    - **Units:** mK sr, the complex RMS per visibility before redundancy averaging.
    - **Required:** unless `effective_noise_path` is set, including when the data use
      `noise_data_path` or `calc_noise`.

`effective_noise_path`
:   Per-visibility noise amplitudes, replacing the `sigma` model.

    - **Format:** a `.npy` vector of length $2N_tN_\nu N_{\rm bls}$, in the
      [data vector ordering](data.md#data-vector-ordering); its squared magnitudes
      are the variances.
    - **Note:** it only changes the weights. It does not affect whether noise is
      added to the data.

A noise vector supplied with `noise_data_path`, or estimated with `calc_noise`,
does not set these weights, so `sigma` should still describe the noise level of
the data. See [noise in the data](data.md#noise).

### Foreground prior

`foreground_prior_std_mk`
:   The prior standard deviation of each foreground spectral component.

    - **Default:** `[8690.96, 2479547.30, 1406322.41]`.
    - **Units:** mK.
    - **Format:** one value for the constant term and one per entry of
      [`beta_fg`](foregrounds.md#configuration); each is applied to every pixel.
    - **Note:** not used with `foreground_kl_path`, whose coefficients have a unit
      prior.
