def test_package_import_does_not_require_optional_solver_or_sampler():
    import bayeseor_jax

    assert bayeseor_jax.BuildMatrices is not None


def test_foreground_example_uses_spherical_model_options_only():
    from bayeseor_jax.params import BayesEoRParser

    args = BayesEoRParser().parse_args(
        ["--config", "test_examples/eor_fgs/config.yaml"]
    )

    assert args.beta_fg == [2.63, 2.82]
    assert len(args.foreground_prior_std_mk) == 3
    assert not hasattr(args, "fg_model_domain")
