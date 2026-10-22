import jax.numpy as jnp
import numpy as np
import pytest
import warnings

from bayeseor_jax import FourierGeometry, build_fourier_geometry
from bayeseor_jax.model.cosmology.power_spectrum import generate_k_modes_in_physical_coordinates
from bayeseor_jax.model.cosmology.fourier_geometry import _resolve_uv_grid_from_extents


@pytest.mark.parametrize("requested", [None, 3, 15])
def test_fourier_resolution_infers_axes_independently_and_warns_if_small(requested):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = _resolve_uv_grid_from_extents(
            extent_u_eor=.2, extent_v_eor=.1,
            freqs_hz=np.array([100e6, 299792458.]),
            uvws_m=np.array([[[10., 20., 0.]], [[30., 10., 0.]]]),
            beam_type="uniform", fwhm_deg=None, diam=None,
            n_u_eor=requested, n_v_eor=requested, du_eor=None, dv_eor=None,
        )
    assert grid["n_u_eor_required"] == 13
    assert grid["n_v_eor_required"] == 5
    assert grid["n_u_eor"] == (13 if requested is None else requested)
    assert grid["n_v_eor"] == (5 if requested is None else requested)
    messages = [str(w.message) for w in caught]
    if requested == 3:
        assert any("n_u_eor=3 is smaller" in msg for msg in messages)
        assert any("n_v_eor=3 is smaller" in msg for msg in messages)
    else:
        assert messages == []


def test_fourier_geometry_defines_retained_uv_eta_grid():
    geometry = build_fourier_geometry(
        n_u=3,
        n_v=3,
        n_eta=3,
        du=0.5,
        dv=0.25,
        d_eta=0.1,
    )

    assert isinstance(geometry, FourierGeometry)
    assert geometry.uv.shape == (2, 8)
    assert geometry.eta.shape == (2,)
    assert (geometry.du, geometry.dv, geometry.d_eta) == (0.5, 0.25, 0.1)
    np.testing.assert_allclose(
        geometry.uv,
        np.asarray(
            [
                [-0.5, 0.0, 0.5, -0.5, 0.5, -0.5, 0.0, 0.5],
                [-0.25, -0.25, -0.25, 0.0, 0.0, 0.25, 0.25, 0.25],
            ]
        ),
    )
    np.testing.assert_allclose(geometry.eta, np.asarray([-0.1, 0.1]))


def test_k_modes_follow_the_retained_fourier_coefficient_order():
    uv = jnp.asarray([[1.0, 2.0], [0.0, 0.0]])
    eta = jnp.asarray([-1.0, 1.0])

    mod_k = generate_k_modes_in_physical_coordinates(
        uv,
        eta,
        transverse_distance_mpc=2.0 * np.pi,
        line_of_sight_distance_mpc_per_hz=2.0 * np.pi,
    )

    np.testing.assert_allclose(
        mod_k,
        np.asarray([np.sqrt(2.0), np.sqrt(2.0), np.sqrt(5.0), np.sqrt(5.0)]),
    )
