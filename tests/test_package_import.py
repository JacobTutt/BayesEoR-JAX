def test_package_import_does_not_require_optional_solver_or_sampler():
    import bayeseor_jax

    assert bayeseor_jax.BuildMatrices is not None


def test_full_likelihood_and_hmc_are_not_exported():
    import bayeseor_jax
    from bayeseor_jax import inference

    for module in (bayeseor_jax, inference):
        for name in ("FullLikelihood", "run_full_nuts", "run_marginalised_nuts"):
            assert name not in module.__all__
            assert not hasattr(module, name)
        assert "MarginalisedLikelihood" in module.__all__
    for name in ("run_multinest", "run_blackjaxns", "run_nestedsampling"):
        assert name in bayeseor_jax.__all__


def test_model_layout_preserves_public_exports():
    import bayeseor_jax as api
    from bayeseor_jax import model
    from bayeseor_jax.model.construction.model_builder import BuildMatrices
    from bayeseor_jax.model.cosmology.fourier_geometry import FourierGeometry
    from bayeseor_jax.model.cosmology.transforms import Cosmology
    from bayeseor_jax.model.instrument.baselines import InstrumentModel
    from bayeseor_jax.model.instrument.beam import Beam_Model
    from bayeseor_jax.model.observation.image_geometry import ImageGeometry
    from bayeseor_jax.model.operators import build_T_21_core, build_T_fg_core

    for value in (
        BuildMatrices, FourierGeometry, Cosmology, InstrumentModel,
        Beam_Model, ImageGeometry, build_T_21_core, build_T_fg_core,
    ):
        assert getattr(api, value.__name__) is value
        assert getattr(model, value.__name__) is value


def test_foreground_example_uses_spherical_model_options_only():
    from bayeseor_jax.params import BayesEoRParser

    args = BayesEoRParser().parse_args(
        ["--config", "test_examples/eor_fgs/config.yaml"]
    )

    assert args.beta_fg == [2.63, 2.82]
    assert len(args.foreground_prior_std_mk) == 3
    assert not hasattr(args, "fg_model_domain")
    assert not hasattr(args, "sky_mask_time_mode")


def test_parser_accepts_preconstructed_kl_operator():
    from bayeseor_jax.params import BayesEoRParser

    args = BayesEoRParser().parse_args([
        '--config', 'test_examples/eor_fgs/config.yaml',
        '--foreground-kl-path', 'arrays/T_fg_reduced.npy',
    ])
    assert args.foreground_kl_path == 'arrays/T_fg_reduced.npy'
