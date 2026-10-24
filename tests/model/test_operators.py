"""Tests for ``bayeseor_jax.model.operators``."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bayeseor_jax.model.operators import (
    T_21,
    T_fg,
    T_fg_weighted,
    build_foreground_spectral_operator,
    build_nudft_matrix,
    build_sky_to_visibility_operator,
    build_spectral_fourier_operator,
    build_T_21_core,
    build_T_fg_core,
    build_transverse_fourier_operator,
    weight_foreground_operator,
)

N_TIME, N_FREQ, N_BL, N_PIX = 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Shared random inputs
# ---------------------------------------------------------------------------

@pytest.fixture
def lmn(rng):
    """Unit direction cosines, shape (n_time, 3, n_pixel), above the horizon."""
    vec = rng.normal(size=(N_TIME, 3, N_PIX))
    vec[:, 2] = np.abs(vec[:, 2]) + 0.5
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


@pytest.fixture
def uvw_lambda(rng):
    """Baselines in wavelengths, shape (n_frequency, 3, n_baseline)."""
    return rng.normal(scale=5.0, size=(N_FREQ, 3, N_BL))


@pytest.fixture
def beam(rng):
    """Positive beam power gains, shape (n_time, n_frequency, n_pixel)."""
    return rng.uniform(0.1, 1.0, size=(N_TIME, N_FREQ, N_PIX))


def brute_force_response(lmn, uvw_lambda, beam, pixel_area):
    """Evaluate R[t,f,b,p] = ΔΩ P[t,f,p] exp(+2πi u_b(ν_f)·s_p(t)) with explicit loops."""
    out = np.zeros((N_TIME, N_FREQ, N_BL, N_PIX), dtype=complex)
    for t in range(N_TIME):
        for f in range(N_FREQ):
            for b in range(N_BL):
                for p in range(N_PIX):
                    phase = np.dot(uvw_lambda[f, :, b], lmn[t, :, p])
                    out[t, f, b, p] = pixel_area * beam[t, f, p] * np.exp(2j * np.pi * phase)
    return out


# ---------------------------------------------------------------------------
# build_nudft_matrix
# ---------------------------------------------------------------------------

class TestBuildNudftMatrix:
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_matches_explicit_kernel(self, rng, sign):
        """Each entry equals exp(sign 2πi y_j·x_i) for every input/output pair."""
        x = rng.normal(size=(3, 4))
        y = rng.normal(size=(3, 6))
        expected = np.array(
            [[np.exp(sign * 2j * np.pi * np.dot(y[:, j], x[:, i])) for i in range(4)] for j in range(6)]
        )
        np.testing.assert_allclose(build_nudft_matrix(x, y, sign=sign), expected, rtol=1e-12)

    def test_default_sign_is_negative(self, rng):
        """The default kernel uses the negative exponent."""
        x, y = rng.normal(size=(2, 3)), rng.normal(size=(2, 2))
        np.testing.assert_allclose(build_nudft_matrix(x, y), build_nudft_matrix(x, y, sign=-1))

    def test_opposite_signs_are_conjugate(self, rng):
        """Reversing the sign gives the complex-conjugate kernel."""
        x, y = rng.normal(size=(2, 3)), rng.normal(size=(2, 2))
        np.testing.assert_allclose(
            build_nudft_matrix(x, y, sign=1), np.conj(build_nudft_matrix(x, y, sign=-1))
        )

    def test_one_dimensional_coordinates(self):
        """One-dimensional inputs are treated as a single coordinate axis."""
        x, y = np.array([0.0, 0.25, 0.5]), np.array([1.0, 2.0])
        expected = np.exp(-2j * np.pi * np.outer(y, x))
        result = build_nudft_matrix(x, y)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_returns_complex128(self, rng):
        """The kernel is returned in double-precision complex arithmetic."""
        assert build_nudft_matrix(rng.normal(size=(2, 3)), rng.normal(size=(2, 2))).dtype == np.complex128


# ---------------------------------------------------------------------------
# build_sky_to_visibility_operator
# ---------------------------------------------------------------------------

class TestBuildSkyToVisibilityOperator:
    def test_matches_brute_force_sum(self, lmn, uvw_lambda, beam):
        """R equals ΔΩ·P·exp(+2πi u·s) evaluated entry by entry."""
        pixel_area = 0.013
        result = build_sky_to_visibility_operator(lmn, uvw_lambda, beam, pixel_area)
        np.testing.assert_allclose(
            result, brute_force_response(lmn, uvw_lambda, beam, pixel_area), rtol=1e-11, atol=1e-14
        )

    def test_output_shape(self, lmn, uvw_lambda, beam):
        """The response has shape (n_time, n_frequency, n_baseline, n_pixel)."""
        assert build_sky_to_visibility_operator(lmn, uvw_lambda, beam, 1.0).shape == (
            N_TIME, N_FREQ, N_BL, N_PIX
        )

    def test_zero_baseline_gives_beam_weighted_area(self, lmn, beam):
        """A zero-length baseline returns ΔΩ·P with no phase."""
        uvw = np.zeros((N_FREQ, 3, 1))
        result = build_sky_to_visibility_operator(lmn, uvw, beam, 0.5)
        np.testing.assert_allclose(result[:, :, 0, :], 0.5 * beam, rtol=1e-14)

    def test_conjugate_for_reversed_baseline(self, lmn, uvw_lambda, beam):
        """Reversing a baseline conjugates its response, for a real beam."""
        forward = build_sky_to_visibility_operator(lmn, uvw_lambda, beam, 1.0)
        reverse = build_sky_to_visibility_operator(lmn, -uvw_lambda, beam, 1.0)
        np.testing.assert_allclose(reverse, np.conj(forward), rtol=1e-12)

    def test_w_term_contributes(self, lmn, uvw_lambda, beam):
        """The up component of the baseline changes the phase, so the full 3D fringe is kept."""
        no_w = uvw_lambda.copy()
        no_w[:, 2] = 0.0
        assert not np.allclose(
            build_sky_to_visibility_operator(lmn, uvw_lambda, beam, 1.0),
            build_sky_to_visibility_operator(lmn, no_w, beam, 1.0),
        )

    @pytest.mark.parametrize(
        "which, bad_shape",
        [
            ("lmn", (N_TIME, 2, N_PIX)),
            ("lmn", (3, N_PIX)),
            ("uvw", (N_FREQ, 2, N_BL)),
            ("uvw", (3, N_BL)),
            ("beam", (N_TIME, N_FREQ, N_PIX + 1)),
            ("beam", (N_FREQ, N_TIME, N_PIX)),
        ],
    )
    def test_rejects_bad_shapes(self, lmn, uvw_lambda, beam, which, bad_shape):
        """Inputs with incompatible shapes raise ValueError."""
        args = {"lmn": lmn, "uvw": uvw_lambda, "beam": beam}
        args[which] = np.ones(bad_shape)
        with pytest.raises(ValueError):
            build_sky_to_visibility_operator(args["lmn"], args["uvw"], args["beam"], 1.0)


# ---------------------------------------------------------------------------
# build_transverse_fourier_operator
# ---------------------------------------------------------------------------

class TestBuildTransverseFourierOperator:
    def test_matches_explicit_formula(self, rng):
        """F'_pu = Δu Δv exp(-2πi (u l + v m)) for every pixel and mode."""
        uv = rng.normal(scale=3.0, size=(2, 4))
        lm = rng.uniform(-0.3, 0.3, size=(2, 6))
        area = 0.07
        expected = area * np.exp(-2j * np.pi * (np.outer(lm[0], uv[0]) + np.outer(lm[1], uv[1])))
        result = build_transverse_fourier_operator(uv, lm, area)
        assert result.shape == (6, 4)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("area", [0.0, -1.0])
    def test_rejects_non_positive_cell_area(self, rng, area):
        """A non-positive Fourier-cell area raises ValueError."""
        with pytest.raises(ValueError, match="uv_cell_area"):
            build_transverse_fourier_operator(rng.normal(size=(2, 2)), rng.normal(size=(2, 2)), area)


# ---------------------------------------------------------------------------
# build_spectral_fourier_operator
# ---------------------------------------------------------------------------

class TestBuildSpectralFourierOperator:
    def test_matches_explicit_formula(self):
        """(F_z)_fe = Δη exp(-2πi Δν_f η_e)."""
        eta = np.array([-2e-6, -1e-6, 1e-6])
        offsets = np.array([-1e5, 0.0, 1e5, 2e5])
        d_eta = 1e-6
        expected = d_eta * np.exp(-2j * np.pi * np.outer(offsets, eta))
        result = build_spectral_fourier_operator(eta, offsets, d_eta)
        assert result.shape == (4, 3)
        assert result.dtype == np.complex128
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_reference_channel_is_real(self):
        """At zero frequency offset every entry is the real weight Δη."""
        result = build_spectral_fourier_operator(np.array([1e-6, 2e-6]), np.array([0.0]), 0.5)
        np.testing.assert_allclose(result, [[0.5, 0.5]])

    @pytest.mark.parametrize("d_eta", [0.0, -1e-6])
    def test_rejects_non_positive_spacing(self, d_eta):
        """A non-positive η spacing raises ValueError."""
        with pytest.raises(ValueError, match="d_eta"):
            build_spectral_fourier_operator(np.ones(2), np.ones(2), d_eta)

    @pytest.mark.parametrize(
        "eta, offsets", [(np.ones((2, 2)), np.ones(2)), (np.ones(2), np.ones((2, 2)))]
    )
    def test_rejects_multidimensional_inputs(self, eta, offsets):
        """η and frequency offsets must be one-dimensional."""
        with pytest.raises(ValueError, match="one-dimensional"):
            build_spectral_fourier_operator(eta, offsets, 1.0)


# ---------------------------------------------------------------------------
# build_T_21_core and T_21
# ---------------------------------------------------------------------------

@pytest.fixture
def cosmological_factors(rng):
    """Random transverse (n_pixel, 3) and spectral (n_frequency, 2) operators."""
    transverse = rng.normal(size=(N_PIX, 3)) + 1j * rng.normal(size=(N_PIX, 3))
    spectral = rng.normal(size=(N_FREQ, 2)) + 1j * rng.normal(size=(N_FREQ, 2))
    return transverse, spectral


class TestBuildT21Core:
    def test_matches_explicit_product(self, lmn, uvw_lambda, beam, cosmological_factors):
        """Each entry equals [Σ_p R_tfbp F'_pu] (F_z)_fe in the documented order."""
        transverse, spectral = cosmological_factors
        response = brute_force_response(lmn, uvw_lambda, beam, 1.0)
        expected = np.zeros((N_TIME * N_FREQ * N_BL, 3 * 2), dtype=complex)
        for t in range(N_TIME):
            for f in range(N_FREQ):
                for b in range(N_BL):
                    row = (t * N_FREQ + f) * N_BL + b
                    for u in range(3):
                        for e in range(2):
                            expected[row, u * 2 + e] = (response[t, f, b] @ transverse[:, u]) * spectral[f, e]
        result = build_T_21_core(response, transverse, spectral)
        np.testing.assert_allclose(result, expected, rtol=1e-11)

    def test_applies_to_coefficients_like_steps(self, lmn, uvw_lambda, beam, cosmological_factors, rng):
        """T_21 a equals reconstructing brightness then summing beam-weighted pixels."""
        transverse, spectral = cosmological_factors
        response = brute_force_response(lmn, uvw_lambda, beam, 1.0)
        a = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        brightness = transverse @ (spectral @ a.T).T  # (n_pixel, n_frequency)
        by_steps = np.einsum("tfbp,pf->tfb", response, brightness).reshape(-1)
        np.testing.assert_allclose(build_T_21_core(response, transverse, spectral) @ a.reshape(-1), by_steps, rtol=1e-11)

    @pytest.mark.parametrize(
        "response_shape, transverse_shape, spectral_shape",
        [
            ((N_TIME, N_FREQ, N_PIX), (N_PIX, 3), (N_FREQ, 2)),
            ((N_TIME, N_FREQ, N_BL, N_PIX), (N_PIX + 1, 3), (N_FREQ, 2)),
            ((N_TIME, N_FREQ, N_BL, N_PIX), (N_PIX,), (N_FREQ, 2)),
            ((N_TIME, N_FREQ, N_BL, N_PIX), (N_PIX, 3), (N_FREQ + 1, 2)),
        ],
    )
    def test_rejects_bad_shapes(self, response_shape, transverse_shape, spectral_shape):
        """Mismatched pixel or frequency axes raise ValueError."""
        with pytest.raises(ValueError):
            build_T_21_core(np.ones(response_shape), np.ones(transverse_shape), np.ones(spectral_shape))


class TestT21:
    def test_equals_core_of_response(self, lmn, uvw_lambda, beam, cosmological_factors):
        """T_21 composes the sky-to-visibility response with build_T_21_core."""
        transverse, spectral = cosmological_factors
        response = build_sky_to_visibility_operator(lmn, uvw_lambda, beam, 0.2)
        np.testing.assert_allclose(
            T_21(lmn, uvw_lambda, beam, 0.2, transverse, spectral),
            build_T_21_core(response, transverse, spectral),
            rtol=1e-13,
        )

    def test_jit_matches_eager(self, lmn, uvw_lambda, beam, cosmological_factors):
        """T_21 gives the same result under jax.jit."""
        transverse, spectral = cosmological_factors
        eager = T_21(lmn, uvw_lambda, beam, 0.2, transverse, spectral)
        np.testing.assert_allclose(jax.jit(T_21)(lmn, uvw_lambda, beam, 0.2, transverse, spectral), eager, rtol=1e-12)


# ---------------------------------------------------------------------------
# build_foreground_spectral_operator
# ---------------------------------------------------------------------------

class TestBuildForegroundSpectralOperator:
    def test_constant_then_power_laws(self):
        """Columns are a constant followed by (ν/ν0)^β for each signed index."""
        freqs = np.array([140.0, 150.0, 160.0])
        result = build_foreground_spectral_operator(freqs, 150.0, (-2.63, -2.82))
        expected = np.column_stack([np.ones(3), (freqs / 150.0) ** -2.63, (freqs / 150.0) ** -2.82])
        assert result.dtype == np.complex128
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_unity_at_reference_frequency(self):
        """Every component equals one at the reference frequency."""
        result = build_foreground_spectral_operator(np.array([120.0]), 120.0, (-1.0, 2.0, 3.5))
        np.testing.assert_allclose(result, np.ones((1, 4)))

    def test_default_indices(self):
        """The default spectral indices are -2.63 and -2.82."""
        np.testing.assert_allclose(
            build_foreground_spectral_operator(np.array([100.0, 200.0]), 150.0),
            build_foreground_spectral_operator(np.array([100.0, 200.0]), 150.0, (-2.63, -2.82)),
        )

    def test_no_power_laws_gives_constant_only(self):
        """An empty index list leaves only the constant component."""
        result = build_foreground_spectral_operator(np.array([100.0, 200.0]), 150.0, ())
        np.testing.assert_allclose(result, np.ones((2, 1)))

    def test_rejects_multidimensional_frequencies(self):
        """Frequencies must be one-dimensional."""
        with pytest.raises(ValueError, match="one-dimensional"):
            build_foreground_spectral_operator(np.ones((2, 2)), 150.0)

    @pytest.mark.parametrize("reference", [0.0, -150.0])
    def test_rejects_non_positive_reference(self, reference):
        """A non-positive reference frequency raises ValueError."""
        with pytest.raises(ValueError, match="reference_frequency_mhz"):
            build_foreground_spectral_operator(np.array([150.0]), reference)


# ---------------------------------------------------------------------------
# build_T_fg_core and T_fg
# ---------------------------------------------------------------------------

class TestBuildTfgCore:
    def test_matches_explicit_entries(self, lmn, uvw_lambda, beam, rng):
        """(T_FG)_(tfb),(sp) = R_tfbp Q_fs, with component-major columns."""
        response = brute_force_response(lmn, uvw_lambda, beam, 1.0)
        q_spec = rng.normal(size=(N_FREQ, 3)) + 0j
        expected = np.zeros((N_TIME * N_FREQ * N_BL, 3 * N_PIX), dtype=complex)
        for t in range(N_TIME):
            for f in range(N_FREQ):
                for b in range(N_BL):
                    row = (t * N_FREQ + f) * N_BL + b
                    for s in range(3):
                        for p in range(N_PIX):
                            expected[row, s * N_PIX + p] = response[t, f, b, p] * q_spec[f, s]
        np.testing.assert_allclose(build_T_fg_core(response, q_spec), expected, rtol=1e-12)

    def test_applies_to_pixel_spectra(self, lmn, uvw_lambda, beam, rng):
        """T_FG q equals Σ_p R_tfbp s_p(ν_f) with s_p = Q q_p."""
        response = brute_force_response(lmn, uvw_lambda, beam, 1.0)
        q_spec = rng.normal(size=(N_FREQ, 3)) + 0j
        q = rng.normal(size=(3, N_PIX))
        spectra = q_spec @ q  # (n_frequency, n_pixel)
        expected = np.einsum("tfbp,fp->tfb", response, spectra).reshape(-1)
        np.testing.assert_allclose(build_T_fg_core(response, q_spec) @ q.reshape(-1), expected, rtol=1e-11)

    @pytest.mark.parametrize(
        "response_shape, spectral_shape",
        [((N_TIME, N_FREQ, N_PIX), (N_FREQ, 3)), ((N_TIME, N_FREQ, N_BL, N_PIX), (N_FREQ + 1, 3)), ((N_TIME, N_FREQ, N_BL, N_PIX), (N_FREQ,))],
    )
    def test_rejects_bad_shapes(self, response_shape, spectral_shape):
        """Mismatched dimensions raise ValueError."""
        with pytest.raises(ValueError):
            build_T_fg_core(np.ones(response_shape), np.ones(spectral_shape))


class TestTfg:
    def test_equals_core_of_response(self, lmn, uvw_lambda, beam):
        """T_fg composes the sky-to-visibility response with build_T_fg_core."""
        q_spec = build_foreground_spectral_operator(np.array([150.0, 151.0, 152.0]), 150.0)
        response = build_sky_to_visibility_operator(lmn, uvw_lambda, beam, 0.3)
        np.testing.assert_allclose(T_fg(lmn, uvw_lambda, beam, 0.3, q_spec), build_T_fg_core(response, q_spec), rtol=1e-13)

    def test_jit_matches_eager(self, lmn, uvw_lambda, beam):
        """T_fg gives the same result under jax.jit."""
        q_spec = build_foreground_spectral_operator(np.array([150.0, 151.0, 152.0]), 150.0)
        np.testing.assert_allclose(
            jax.jit(T_fg)(lmn, uvw_lambda, beam, 0.3, q_spec), T_fg(lmn, uvw_lambda, beam, 0.3, q_spec), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# weight_foreground_operator and T_fg_weighted
# ---------------------------------------------------------------------------

class TestWeightForegroundOperator:
    @pytest.fixture
    def operator(self, rng):
        """Random complex (n_visibility=6, n_coefficient=4) operator."""
        return rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))

    def test_diagonal_factors_scale_rows_and_columns(self, operator, rng):
        """Vector factors give diag(W) T diag(L)."""
        prior = rng.uniform(1, 2, size=4)
        whitener = rng.uniform(1, 2, size=6)
        np.testing.assert_allclose(
            weight_foreground_operator(operator, prior, whitener), np.diag(whitener) @ operator @ np.diag(prior), rtol=1e-13
        )

    def test_dense_factors_multiply(self, operator, rng):
        """Square factors give W @ T @ L."""
        prior = rng.normal(size=(4, 4))
        whitener = rng.normal(size=(6, 6))
        np.testing.assert_allclose(
            weight_foreground_operator(operator, prior, whitener), whitener @ operator @ prior, rtol=1e-12
        )

    def test_vector_and_diagonal_matrix_agree(self, operator, rng):
        """A diagonal factor supplied as a vector equals the same diagonal matrix."""
        prior, whitener = rng.uniform(1, 2, size=4), rng.uniform(1, 2, size=6)
        np.testing.assert_allclose(
            weight_foreground_operator(operator, prior, whitener),
            weight_foreground_operator(operator, np.diag(prior), np.diag(whitener)),
            rtol=1e-13,
        )

    def test_gram_is_whitened_prior_predictive_covariance(self, operator, rng):
        """A Aᴴ equals W T Σ_q Tᴴ Wᴴ for Σ_q = L Lᴴ."""
        prior = rng.uniform(1, 2, size=4)
        whitener = rng.uniform(1, 2, size=6)
        a = np.asarray(weight_foreground_operator(operator, prior, whitener))
        w = np.diag(whitener)
        expected = w @ operator @ np.diag(prior**2) @ operator.conj().T @ w.conj().T
        np.testing.assert_allclose(a @ a.conj().T, expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "operator_shape, prior_shape, whitener_shape",
        [((6,), (4,), (6,)), ((6, 4), (3,), (6,)), ((6, 4), (4, 3), (6,)), ((6, 4), (4,), (5,)), ((6, 4), (4,), (6, 5))],
    )
    def test_rejects_bad_shapes(self, operator_shape, prior_shape, whitener_shape):
        """Factors that do not match the operator axes raise ValueError."""
        with pytest.raises(ValueError):
            weight_foreground_operator(np.ones(operator_shape), np.ones(prior_shape), np.ones(whitener_shape))


class TestTfgWeighted:
    def test_equals_weighting_t_fg(self, lmn, uvw_lambda, beam, rng):
        """T_fg_weighted equals weight_foreground_operator applied to T_fg."""
        q_spec = build_foreground_spectral_operator(np.array([150.0, 151.0, 152.0]), 150.0)
        prior = rng.uniform(1, 2, size=3 * N_PIX)
        whitener = rng.uniform(1, 2, size=N_TIME * N_FREQ * N_BL)
        expected = weight_foreground_operator(T_fg(lmn, uvw_lambda, beam, 0.3, q_spec), prior, whitener)
        result = T_fg_weighted(lmn, uvw_lambda, beam, 0.3, q_spec, prior_sqrt=prior, noise_whitener=whitener)
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_jit_matches_eager(self, lmn, uvw_lambda, beam, rng):
        """T_fg_weighted gives the same result under jax.jit."""
        q_spec = build_foreground_spectral_operator(np.array([150.0, 151.0, 152.0]), 150.0)
        prior = rng.uniform(1, 2, size=3 * N_PIX)
        whitener = rng.uniform(1, 2, size=N_TIME * N_FREQ * N_BL)
        kwargs = dict(prior_sqrt=prior, noise_whitener=whitener)
        np.testing.assert_allclose(
            jax.jit(T_fg_weighted)(lmn, uvw_lambda, beam, 0.3, q_spec, **kwargs),
            T_fg_weighted(lmn, uvw_lambda, beam, 0.3, q_spec, **kwargs),
            rtol=1e-12,
        )
