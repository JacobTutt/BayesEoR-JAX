import jax.numpy as jnp
import numpy as np

from bayeseor_jax import FourierGeometry, build_fourier_geometry
from bayeseor_jax.model.cosmology.power_spectrum import generate_k_modes_in_physical_coordinates


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
