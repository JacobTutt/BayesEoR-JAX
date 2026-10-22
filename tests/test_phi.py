import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.inference.phi import PhiInverse


def _foreground_only_phi(*, n_par=6, widths=None):
    return PhiInverse(
        bin_id_per_k_mode=jnp.empty((0,), dtype=jnp.int32),
        valid_mask=jnp.empty((0,), dtype=bool),
        k_vals=jnp.empty((0,)),
        ps_box_size_ra_mpc=1.0,
        ps_box_size_dec_mpc=1.0,
        ps_box_size_para_mpc=1.0,
        inst_to_cosmo_vol=1.0,
        n_par=n_par,
        foreground_prior_std=np.repeat([1., 2., 4.], 2) if widths is None else widths,
    )


def test_foreground_prior_matches_component_major_operator_order():
    phi = _foreground_only_phi()

    result = phi.evaluate(jnp.empty((0,)))

    np.testing.assert_allclose(
        result,
        np.asarray([1.0, 1.0, 0.25, 0.25, 0.0625, 0.0625]),
    )


def test_foreground_prior_rejects_operator_length_mismatch():
    with pytest.raises(ValueError, match="foreground prior length mismatch"):
        _foreground_only_phi(n_par=7)


@pytest.mark.parametrize('rank', [0, 1, 5])
def test_standard_normal_foreground_prior_has_retained_rank(rank):
    phi = _foreground_only_phi(n_par=rank, widths=np.ones(rank))
    np.testing.assert_array_equal(jax.jit(phi.evaluate)(jnp.empty(0)), np.ones(rank))


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf, 1j])
def test_foreground_prior_rejects_invalid_widths(bad):
    with pytest.raises(ValueError, match='finite, real and positive'):
        _foreground_only_phi(n_par=1, widths=[bad])


def test_changing_foreground_representation_preserves_cosmological_prior():
    arguments = dict(
        bin_id_per_k_mode=jnp.array([0, 1, 0]), valid_mask=jnp.ones(3, dtype=bool),
        k_vals=jnp.array([.1, .2]), ps_box_size_ra_mpc=2., ps_box_size_dec_mpc=3.,
        ps_box_size_para_mpc=4., inst_to_cosmo_vol=5.,
    )
    original = PhiInverse(**arguments, n_par=9, foreground_prior_std=np.repeat([1., 2., 4.], 2))
    reduced = PhiInverse(**arguments, n_par=5, foreground_prior_std=np.ones(2))
    for powers in (jnp.array([1., 2.]), jnp.array([3., 8.])):
        original_precision = jax.jit(original.evaluate)(powers)
        reduced_precision = jax.jit(reduced.evaluate)(powers)
        np.testing.assert_array_equal(original_precision[:3], reduced_precision[:3])
        np.testing.assert_array_equal(reduced_precision[3:], [1., 1.])
