import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.inference.phi import PhiInverse


def _foreground_only_phi(*, n_par=6, n_spatial=2):
    return PhiInverse(
        bin_id_per_k_mode=jnp.empty((0,), dtype=jnp.int32),
        valid_mask=jnp.empty((0,), dtype=bool),
        k_vals=jnp.empty((0,)),
        ps_box_size_ra_mpc=1.0,
        ps_box_size_dec_mpc=1.0,
        ps_box_size_para_mpc=1.0,
        inst_to_cosmo_vol=1.0,
        n_par=n_par,
        foreground_prior_std_mk=(1.0, 2.0, 4.0),
        n_fg_spatial_modes=n_spatial,
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
