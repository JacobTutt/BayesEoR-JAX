def test_package_import_does_not_require_optional_solver_or_sampler():
    import bayeseor_jax

    assert bayeseor_jax.BuildMatrices is not None
