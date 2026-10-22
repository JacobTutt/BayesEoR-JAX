from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from bayeseor_jax import FourierGeometry, build_fourier_geometry
from bayeseor_jax.model.k_cube import generate_k_modes_in_physical_coordinates


def test_fourier_geometry_defines_uv_eta_grid_and_bases_together():
    image_geometry = SimpleNamespace(
        lmn_fg=jnp.asarray([[[0.0, 0.5], [0.0, 0.0], [1.0, np.sqrt(0.75)]]]),
        time_ind_center=0,
        eor_fg_shared_axis=jnp.asarray([True, True]),
    )

    geometry = build_fourier_geometry(
        image_geometry=image_geometry,
        n_u=3,
        n_v=3,
        n_eta=3,
        du=0.5,
        dv=0.25,
        d_eta=0.1,
        n_frequency=3,
    )

    assert isinstance(geometry, FourierGeometry)
    assert geometry.uv.shape == (2, 8)
    assert geometry.eta.shape == (2,)
    assert geometry.transverse_basis.shape == (2, 8)
    assert geometry.spectral_basis.shape == (3, 2)
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

    frequency_indices = np.asarray([-1.0, 0.0, 1.0])
    eta_indices = np.asarray([-1.0, 1.0])
    expected_spectral_basis = (
        np.exp(-2j * np.pi * np.outer(frequency_indices, eta_indices) / 3.0)
        * 0.1
    )
    np.testing.assert_allclose(geometry.spectral_basis, expected_spectral_basis)


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
