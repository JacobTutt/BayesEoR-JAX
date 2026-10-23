# Construction and KL reduction

These functions build full-size operators on CPU in time-frequency blocks, writing
them to disk, and return the saved paths. Every participating process makes the
same calls; see [execution](../forward-model.md#execution).

## Operators

`BuildMatrices` prepares the physical inputs from a configuration and exposes the
same writers as methods. The writers can also be called directly.

::: bayeseor_jax.model.construction.model_builder.BuildMatrices
    options:
      heading_level: 3

::: bayeseor_jax.model.construction.operator_construction
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - T_21_build_save
        - T_fg_build_save
        - T_fg_weighted_build_save
        - assemble_joint_operator

## KL reduction

`T_fg_svd_build_save` decomposes a prior- and noise-weighted operator, truncates it
and saves the [reduced basis](../kl-reduced-basis.md).

::: bayeseor_jax.model.construction.kl_reduction.T_fg_svd_build_save
    options:
      heading_level: 3

::: bayeseor_jax.model.construction.kl_reduction.select_foreground_rank
    options:
      heading_level: 3

## Likelihood products

The fixed products of the [marginalised likelihood](../marginalisation.md),
$\mathbf T^\dagger\mathbf N^{-1}\mathbf T$, $\mathbf T^\dagger\mathbf N^{-1}\boldsymbol d$
and $\boldsymbol d^\dagger\mathbf N^{-1}\boldsymbol d$. `save_likelihood_products`
writes all three; the `build_` functions evaluate them in memory.

::: bayeseor_jax.model.construction.likelihood_products
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - save_likelihood_products
        - build_gram_matrix
        - build_data_projection
        - build_data_power
        - save_gram_matrix
        - save_data_projection
        - save_data_power
