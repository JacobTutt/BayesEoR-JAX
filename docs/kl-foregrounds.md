# KL foreground basis

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

## 1. Weight the physical mapping by prior and noise

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

## 2. Decompose the weighted operator

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

## 3. Choose the retained rank

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

## 4. Restore visibility units

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

## 5. Save and use the reduced operator

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
