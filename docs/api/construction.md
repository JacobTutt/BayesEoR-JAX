# Construction and KL reduction

Construction routines divide operator assembly into CPU blocks, save the results,
and prepare the reduced foreground and joint marginal-likelihood products.

::: bayeseor_jax.model.construction.model_builder.BuildMatrices

::: bayeseor_jax.model.construction.operator_construction
    options:
      show_root_heading: false
      members:
        - T_21_build_save
        - T_fg_build_save
        - T_fg_weighted_build_save
        - assemble_joint_operator

::: bayeseor_jax.model.construction.kl_reduction.select_foreground_rank

::: bayeseor_jax.model.construction.kl_reduction.T_fg_svd_build_save

::: bayeseor_jax.model.construction.likelihood_products
    options:
      show_root_heading: false
      members:
        - build_gram_matrix
        - build_data_projection
        - build_data_power
        - save_likelihood_products
        - save_gram_matrix
        - save_data_projection
        - save_data_power
