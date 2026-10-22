import jax.numpy as jnp
import numpy as np

from bayeseor_jax import (
    Beam_Model,
    build_T_21_core,
    build_T_fg_core,
    build_foreground_spectral_operator,
    build_fourier_geometry,
    build_sky_to_visibility_operator,
    build_spectral_fourier_operator,
    build_transverse_fourier_operator,
)


def test_forward_operators_preserve_real_sky_conjugacy():
    geometry = build_fourier_geometry(n_u=3, n_v=3, n_eta=3, du=.5, dv=.4, d_eta=.1)
    lmn = jnp.array([[[0., .3], [0., .4], [1., np.sqrt(.75)]]])
    uvw = jnp.array([[[1., -1.], [.2, -.2], [.1, -.1]]])
    uvw = uvw * jnp.array([.9, 1., 1.1])[:, None, None]
    response = build_sky_to_visibility_operator(lmn, uvw, jnp.ones((1, 3, 2)), .1)
    transverse = build_transverse_fourier_operator(geometry.uv, lmn[0, :2], uv_cell_area=.2)
    spectral = build_spectral_fourier_operator(
        eta=geometry.eta, frequency_offsets_hz=jnp.array([-1., 0., 1.]), d_eta=.1,
    )
    t21 = np.asarray(build_T_21_core(response, transverse, spectral)).reshape(3, 2, -1)
    tfg = np.asarray(build_T_fg_core(response, jnp.array([[1., 2.], [1., 3.], [1., 4.]]))).reshape(3, 2, -1)
    # Reversing a baseline conjugates real-pixel responses and swaps Fourier partners.
    np.testing.assert_allclose(tfg[:, 1], tfg[:, 0].conj(), atol=1e-13)
    np.testing.assert_allclose(t21[:, 1], t21[:, 0, ::-1].conj(), atol=1e-13)


def test_sky_to_visibility_matches_explicit_measurement_equation():
    lmn = jnp.asarray([[[0.0, 0.5], [0.0, 0.0], [1.0, np.sqrt(0.75)]]])
    uvw = jnp.asarray([[[1.0], [0.0], [0.0]]])
    beam = jnp.asarray([[[1.0, 0.25]]])

    response = build_sky_to_visibility_operator(lmn, uvw, beam, 0.1)

    expected = 0.1 * np.asarray(
        [[[[1.0, 0.25 * np.exp(2j * np.pi * 0.5)]]]]
    )
    np.testing.assert_allclose(response, expected, rtol=1e-13, atol=1e-13)


def test_beam_has_zero_gain_at_and_below_horizon():
    coords = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.0, jnp.pi / 2, 3 * jnp.pi / 4]]]
    )

    beam = Beam_Model("uniform").get_beam_vals(
        coords,
        jnp.asarray([100.0e6]),
    )

    np.testing.assert_array_equal(beam, np.asarray([[[1.0, 0.0, 0.0]]]))


def test_T_21_core_is_R_Fprime_Fz():
    response = jnp.asarray(
        [
            [
                [[1.0 + 1.0j, 2.0 - 1.0j]],
                [[0.5 + 0.0j, -1.0 + 2.0j]],
            ]
        ]
    )
    transverse_operator = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])
    spectral_operator = jnp.asarray([[1.0, 2.0j], [0.5, -1.0j]])

    result = build_T_21_core(response, transverse_operator, spectral_operator)

    expected_rows = []
    for frequency in range(2):
        spatial = np.asarray(response[0, frequency, 0]) @ np.asarray(
            transverse_operator
        )
        expected_rows.append(
            (spatial[:, None] * np.asarray(spectral_operator[frequency])[None, :])
            .reshape(-1)
        )
    np.testing.assert_allclose(result, np.asarray(expected_rows))


def test_T_fg_core_uses_component_major_column_order():
    response = jnp.asarray([[[[2.0, 3.0]]]])
    spectral_operator = jnp.asarray([[1.0, 10.0, 100.0]])

    result = build_T_fg_core(response, spectral_operator)

    np.testing.assert_array_equal(
        result,
        np.asarray([[2.0, 3.0, 20.0, 30.0, 200.0, 300.0]]),
    )


def test_forward_operator_rows_are_time_frequency_baseline_ordered():
    response = jnp.arange(8).reshape(2, 2, 2, 1)

    result = build_T_fg_core(response, jnp.ones((2, 1)))

    np.testing.assert_array_equal(result[:, 0], np.arange(8))


def test_foreground_spectral_operator_matches_physical_model():
    frequencies = jnp.asarray([100.0, 200.0])

    result = build_foreground_spectral_operator(
        frequencies,
        reference_frequency_mhz=100.0,
        spectral_indices=(-2.0, -3.0),
    )

    np.testing.assert_allclose(
        result,
        np.asarray([[1.0, 1.0, 1.0], [1.0, 0.25, 0.125]]),
    )


def test_transverse_fourier_operator_maps_uv_modes_to_image_pixels():
    uv = jnp.asarray([[0.0, 0.5], [0.0, 0.0]])
    lm = jnp.asarray([[0.0, 0.5], [0.0, 0.0]])

    result = build_transverse_fourier_operator(uv, lm, uv_cell_area=0.25)

    expected = 0.25 * np.exp(
        -2j * np.pi * np.asarray([[0.0, 0.0], [0.0, 0.25]])
    )
    np.testing.assert_allclose(result, expected)


def test_spectral_fourier_operator_preserves_delay_transform_convention():
    frequency_indices = np.asarray([-1.0, 0.0, 1.0])
    eta_indices = np.asarray([-1.0, 1.0])
    d_eta = 0.1

    result = build_spectral_fourier_operator(
        eta=jnp.asarray(eta_indices * d_eta),
        frequency_offsets_hz=jnp.asarray(frequency_indices / (3.0 * d_eta)),
        d_eta=d_eta,
    )

    expected = (
        np.exp(-2j * np.pi * np.outer(frequency_indices, eta_indices) / 3.0)
        * d_eta
    )
    np.testing.assert_allclose(result, expected)
