# Operators

These functions evaluate the [EoR](../model/eor.md) and [foreground](../model/foregrounds.md)
operators as JAX arrays. They do not read files, choose hardware or apply priors,
so they can be used inside compiled code. For full-size operators saved to disk,
see [construction](construction.md).

$$
\mathbf T_{21}=\mathbf F^{-1}\mathbf P\mathbf F'\mathbf F_z,
\qquad
\mathbf T_{\rm FG}=\mathbf F^{-1}\mathbf P\mathbf Q_z
$$

## Main interface

`T_21` and `T_fg` build a block of each operator from the sky geometry, baselines
and beam. `T_fg_weighted` applies the prior and noise weighting used by the
[KL reduced basis](../model/kl-reduced-basis.md).

::: bayeseor_jax.model.operators
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - T_21
        - T_fg
        - T_fg_weighted

## Building blocks

Each factor of the operators can also be built separately: the fringe response,
$\mathbf R = \mathbf F^{-1}\mathbf P$; the Fourier transforms, $\mathbf F'$ and
$\mathbf F_z$; and the foreground spectral basis, $\mathbf Q_z$.

::: bayeseor_jax.model.operators
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - build_sky_to_visibility_operator
        - build_transverse_fourier_operator
        - build_spectral_fourier_operator
        - build_foreground_spectral_operator
        - build_T_21_core
        - build_T_fg_core
        - weight_foreground_operator
        - build_nudft_matrix
