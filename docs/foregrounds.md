# Foreground model

Foreground brightness is parameterised directly on a spherical HEALPix grid.
Each pixel carries spectral amplitudes that are propagated through the primary
beam and full three-dimensional interferometric response, including emission
towards the horizon without a tangent-plane approximation to the foreground sky.

[![Foreground-only forward model: image-domain coefficients pass through Q_z, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/foreground-workflow.png#only-light)](assets/diagrams/foreground-workflow.png)
[![Foreground-only forward model: image-domain coefficients pass through Q_z, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/foreground-workflow-dark.png#only-dark)](assets/diagrams/foreground-workflow-dark.png)

*The component-only schematic contains no cosmological or inference branch.
A single illustrative visibility plot represents the modelled output.*

The complete mapping is

$$
\boldsymbol m_{\rm FG}
=\underbrace{\mathbf F^{-1}\mathbf P\mathbf Q_z}_{\mathbf T_{\rm FG}}
\boldsymbol q.
$$

## 1. Define the image-domain amplitudes

Each spectral component has a coefficient at every selected image pixel:

$$
\boldsymbol q_g=(q_{g,1},\ldots,q_{g,N_{\rm pix}})^{\mathsf T},
\qquad
\boldsymbol q=(\boldsymbol q_0^{\mathsf T},\ldots,
\boldsymbol q_{N_\beta}^{\mathsf T})^{\mathsf T}.
$$

The configured model has a constant component plus one component for each
power-law index, giving $N_{\rm FG,param}=(1+N_\beta)N_{\rm pix}$.
[build_image_geometry](api/geometry.md#bayeseor_jax.model.observation.image_geometry.build_image_geometry)
selects the time-union of pixels within `fov_fg` and tracks their topocentric
directions throughout the observation.

## 2. Reconstruct the spectrum at each pixel: $\mathbf Q_z$

The spectral factors turn the amplitudes into brightness at each observing channel:

$$
I_{{\rm FG},p}(\nu_f)=\sum_g Q_{fg}q_{g,p},
\qquad
Q_{f0}=1,\quad
Q_{fg}=\left(\frac{\nu_f}{\nu_0}\right)^{-\beta_g}\quad(g>0).
$$

[build_foreground_spectral_operator](api/operators.md#bayeseor_jax.model.operators.build_foreground_spectral_operator)
returns the compact frequency-by-component matrix $Q$.
The full $\mathbf Q_z$ applies these same spectral factors independently to
every pixel; it need not be allocated as a large matrix.

The high-level `beta_fg` configuration uses positive spectral slopes and
[BuildMatrices](api/construction.md#bayeseor_jax.model.construction.model_builder.BuildMatrices)
passes their negatives as the function's signed `spectral_indices`.
The builder uses the first observing channel as $\nu_0$.
The image amplitudes retain the units supplied by the caller; the operator
itself does not convert units.

## 3. Weight the brightness by the primary beam: $\mathbf P$

For each time and frequency, the primary beam weights the pixel's contribution:

$$
I_{{\rm FG},p}^{\,P}(\nu_f,t)=P_{tfp}I_{{\rm FG},p}(\nu_f).
$$

[Beam_Model](api/geometry.md#bayeseor_jax.model.instrument.beam.Beam_Model)
supports analytic and interpolated beam responses. Its returned gains are zero
at or below the horizon, so a fixed celestial pixel set can be followed through
the observation without including below-horizon emission.

## 4. Propagate into sampled visibilities: $\mathbf F^{-1}$

Apply the three-dimensional fringe response and sum over the image:

$$
m_{{\rm FG},tfb}
=\sum_p (F^{-1})_{tfbp}I_{{\rm FG},p}^{\,P}(\nu_f,t)
=\sum_p R_{tfbp}I_{{\rm FG},p}(\nu_f).
$$

The [shared response](model.md#shared-sky-to-visibility-response)
includes the pixel solid angle and the frequency-dependent baseline coordinates.
[build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator)
evaluates the combined $\mathbf R=\mathbf F^{-1}\mathbf P$ without forming
a separate dense beam matrix.

## 5. Assemble the foreground operator: $\mathbf T_{\rm FG}$

The contribution of one pixel's spectral amplitude is therefore

$$
(T_{\rm FG})_{(tfb),(gp)}=R_{tfbp}Q_{fg},
\qquad
\boldsymbol m_{\rm FG}=\mathbf T_{\rm FG}\boldsymbol q.
$$

[build_T_fg_core](api/operators.md#bayeseor_jax.model.operators.build_T_fg_core)
evaluates this product, while [T_fg](api/operators.md#bayeseor_jax.model.operators.T_fg)
also constructs $\mathbf R$ from the supplied inputs.
Columns are ordered by spectral component, then pixel.
[T_fg_build_save](api/construction.md#bayeseor_jax.model.construction.operator_construction.T_fg_build_save)
uses the same mapping to build CPU blocks and save **T_fg.npy**.

## Foreground uncertainty and the KL basis

A Gaussian prior $\boldsymbol q\sim\mathcal N(0,\mathbf\Sigma_q)$ induces
the visibility-space covariance

$$
\mathbf C_{\rm FG}
=\mathbf T_{\rm FG}\mathbf\Sigma_q\mathbf T_{\rm FG}^{\dagger}.
$$

The default setup repeats each configured component width across its pixels:

~~~yaml
fov_fg: 180.0
beta_fg: [2.63, 2.82]
foreground_prior_std_mk: [8690.96, 2479547.30, 1406322.41]
~~~

The low-level weighting functions also accept nonuniform coefficient uncertainties.
A nonzero mean, if supplied by a caller, is propagated separately from this
covariance.

The [KL foreground basis](foregrounds.md#kl-foreground-basis) represents this prior-predictive
uncertainty with fewer visibility-space modes. It is a separate construction
stage: neither the physical spectral model nor its instrumental propagation
is replaced by a phenomenological kernel.

## KL foreground basis

The physical foreground model can contain many coefficients describing
visibility responses that are degenerate or small relative to the observational
noise. The KL construction identifies the modes needed to represent their
combined prior-predictive uncertainty, with an explicit noise-relative
truncation criterion.

The workflow is

$$
\mathbf T_{\rm FG}
\ \longrightarrow\
\mathbf A=\mathbf N^{-1/2}\mathbf T_{\rm FG}\mathbf\Sigma_q^{1/2}
\ \longrightarrow\
(\mathbf U,\mathbf S)
\ \longrightarrow\
\mathbf T_{{\rm FG},r}.
$$

This stage begins with the [physical foreground operator](foregrounds.md)
and ends with a reduced operator in the same visibility space.

### 1. Weight the physical mapping by prior and noise

Let $\mathbf\Sigma_q$ describe the foreground coefficient covariance and
$\mathbf N$ the visibility-noise covariance. Define

$$
\mathbf A=\mathbf N^{-1/2}\mathbf T_{\rm FG}\mathbf\Sigma_q^{1/2},
\qquad
\mathbf C_{{\rm FG},w}=\mathbf A\mathbf A^\dagger.
$$

[weight_foreground_operator](api/operators.md#bayeseor_jax.model.operators.weight_foreground_operator)
applies these factors to an existing operator.
[T_fg_weighted](api/operators.md#bayeseor_jax.model.operators.T_fg_weighted)
constructs and weights a supplied visibility block directly.

For diagonal priors and noise, each column is multiplied by its coefficient
prior standard deviation and each row divided by its visibility-noise RMS.
[T_fg_weighted_build_save](api/construction.md#bayeseor_jax.model.construction.operator_construction.T_fg_weighted_build_save)
performs this calculation on CPU blocks and saves **A_fg.npy**, without
requiring a complete unweighted matrix to be saved first.

### 2. Decompose the weighted operator

A thin singular-value decomposition gives

$$
\mathbf A=\mathbf U\mathbf S\mathbf V^\dagger,
\qquad
\mathbf C_{{\rm FG},w}
=\mathbf U\mathbf S^2\mathbf U^\dagger.
$$

The columns of $\mathbf U$ are orthonormal visibility-space modes.
Their eigenvalues $\lambda_i=s_i^2$ measure prior-predictive variance relative
to the noise. An eigendecomposition of $\mathbf C_{{\rm FG},w}$ gives the
same eigenspaces; the saved-file construction currently uses the operator SVD.

The exact rank obeys

$$
r_{\rm FG}\leq\min(N_{\rm d},N_{\rm FG,param}).
$$

Retaining the largest singular modes minimises the discarded
noise-whitened covariance trace for a given rank. This is the sense in which
the representation is statistically optimal.

### 3. Choose the retained rank

With singular values ordered from largest to smallest, retain the smallest
$r$ satisfying

$$
\epsilon_{\rm RMS}(r)
=\left[\frac{1}{N_{\rm d}}\sum_{i>r}s_i^2\right]^{1/2}
\leq f.
$$

[select_foreground_rank](api/construction.md#bayeseor_jax.model.construction.kl_reduction.select_foreground_rank)
implements this criterion. The default $f=0.1$ limits the average discarded
prior-predictive RMS in whitened visibility space to ten per cent of the noise RMS.
This is not a bound on every individual visibility or every possible sky realisation.

### 4. Restore visibility units

Undo the noise whitening, but retain the absorbed foreground prior:

$$
\mathbf T_{{\rm FG},r}
=\mathbf N^{1/2}\mathbf U_r\mathbf S_r,
\qquad
\boldsymbol m_{\rm FG}^{(r)}=\mathbf T_{{\rm FG},r}\boldsymbol z,
\qquad
\langle\boldsymbol z\boldsymbol z^\dagger\rangle=\mathbf I_r.
$$

The reduced covariance is

$$
\mathbf C_{\rm FG}^{(r)}
=\mathbf T_{{\rm FG},r}\mathbf T_{{\rm FG},r}^{\dagger}.
$$

The reduced coefficients have a standard-normal prior by construction.
Applying the original physical prior widths again would double-count their
weighting.

### 5. Save and use the reduced operator

[T_fg_svd_build_save](api/construction.md#bayeseor_jax.model.construction.kl_reduction.T_fg_svd_build_save)
loads the weighted operator, decomposes it, selects $r$, undoes whitening and
saves the reduced operator:

~~~python
from bayeseor_jax import T_fg_svd_build_save

reduced_path = T_fg_svd_build_save(
    "arrays/A_fg.npy",
    "arrays/Ninv.npy",
    output_path="arrays/T_fg_reduced.npy",
    f=0.1,
)
~~~

The supplied inverse-noise diagonal must be the one used in constructing
$\mathbf A$. Native JAX provides the single-process CPU decomposition;
the GPU path uses JAXMG with one process per GPU and shared storage.
The saved-file routine supports diagonal noise and saves the singular values
and truncation diagnostics alongside the reduced operator.

Select it for subsequent setup with

~~~yaml
foreground_kl_path: arrays/T_fg_reduced.npy
~~~

Setup then uses its columns with unit prior widths, optionally alongside
$\mathbf T_{21}$. It does not run the decomposition again or reapply
`foreground_prior_std_mk`.

The operator must match the observation, visibility ordering and foreground
uncertainty model. Shape checks alone cannot establish that match. See the
[construction guide](forward-model.md#foreground-svd-and-truncation) for
memory requirements, distributed execution and saved metadata.
