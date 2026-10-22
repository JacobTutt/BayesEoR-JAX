# Linear operators

These functions define the mappings between coefficient, image and visibility
spaces. They are independent of the disk-backed construction workflow.

::: bayeseor_jax.model.operators
    options:
      show_root_heading: false
      members:
        - build_nudft_matrix
        - build_sky_to_visibility_operator
        - build_transverse_fourier_operator
        - build_spectral_fourier_operator
        - build_T_21_core
        - T_21
        - build_foreground_spectral_operator
        - build_T_fg_core
        - T_fg
        - weight_foreground_operator
        - T_fg_weighted
