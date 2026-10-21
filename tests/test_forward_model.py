import jax.numpy as jnp
import numpy as np

from bayeseor_jax import (
    build_T_eor_core,
    build_T_fg_core,
    build_foreground_spectral_basis,
    build_sky_to_visibility_core,
)


def test_sky_to_visibility_matches_explicit_measurement_equation():
    lmn = jnp.asarray([[[0.0, 0.5], [0.0, 0.0], [1.0, np.sqrt(0.75)]]])
    uvw = jnp.asarray([[[1.0], [0.0], [0.0]]])
    beam = jnp.asarray([[[1.0, 0.25]]])

    response = build_sky_to_visibility_core(lmn, uvw, beam, 0.1)

    expected = 0.1 * np.asarray(
        [[[[1.0, 0.25 * np.exp(2j * np.pi * 0.5)]]]]
    )
    np.testing.assert_allclose(response, expected, rtol=1e-13, atol=1e-13)


def test_sky_to_visibility_masks_pixels_below_horizon():
    lmn = jnp.asarray([[[0.0], [0.0], [-1.0]]])
    uvw = jnp.zeros((1, 3, 1))
    beam = jnp.ones((1, 1, 1))

    response = build_sky_to_visibility_core(lmn, uvw, beam, 1.0)

    np.testing.assert_array_equal(response, np.zeros((1, 1, 1, 1)))


def test_T_eor_core_is_R_Fprime_Fz():
    response = jnp.asarray(
        [
            [
                [[1.0 + 1.0j, 2.0 - 1.0j]],
                [[0.5 + 0.0j, -1.0 + 2.0j]],
            ]
        ]
    )
    transverse_basis = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])
    spectral_basis = jnp.asarray([[1.0, 2.0j], [0.5, -1.0j]])

    result = build_T_eor_core(response, transverse_basis, spectral_basis)

    expected_rows = []
    for frequency in range(2):
        spatial = np.asarray(response[0, frequency, 0]) @ np.asarray(
            transverse_basis
        )
        expected_rows.append(
            (spatial[:, None] * np.asarray(spectral_basis[frequency])[None, :])
            .reshape(-1)
        )
    np.testing.assert_allclose(result, np.asarray(expected_rows))


def test_T_fg_core_uses_component_major_column_order():
    response = jnp.asarray([[[[2.0, 3.0]]]])
    spectral_basis = jnp.asarray([[1.0, 10.0, 100.0]])

    result = build_T_fg_core(response, spectral_basis)

    np.testing.assert_array_equal(
        result,
        np.asarray([[2.0, 3.0, 20.0, 30.0, 200.0, 300.0]]),
    )


def test_forward_operator_rows_are_time_frequency_baseline_ordered():
    response = jnp.arange(8).reshape(2, 2, 2, 1)

    result = build_T_fg_core(response, jnp.ones((2, 1)))

    np.testing.assert_array_equal(result[:, 0], np.arange(8))


def test_foreground_spectral_basis_matches_physical_model():
    frequencies = jnp.asarray([100.0, 200.0])

    result = build_foreground_spectral_basis(
        frequencies,
        reference_frequency_mhz=100.0,
        spectral_indices=(-2.0, -3.0),
    )

    np.testing.assert_allclose(
        result,
        np.asarray([[1.0, 1.0, 1.0], [1.0, 0.25, 0.125]]),
    )
