"""Tests for ``bayeseor_jax.model.cosmology.power_spectrum``."""

from __future__ import annotations

import numpy as np
import pytest

from bayeseor_jax.model.cosmology.power_spectrum import (
    _spherical_bin_edges,
    calc_mean_binned_k_vals,
    compress_to_nonempty_k_bins,
    generate_k_cube_model_cylindrical_binning,
    generate_k_cube_model_spherical_binning,
    generate_k_modes_in_physical_coordinates,
)

L_PARA = 100.0  # Mpc
DK_PARA = 2.0 * np.pi / L_PARA


def reference_bin(k, low, high):
    """Return the (low, high] bin of k by direct comparison, or -1."""
    for i, (lo, hi) in enumerate(zip(low, high)):
        if lo < k <= hi:
            return i
    return -1


class TestGenerateKModesInPhysicalCoordinates:
    def test_matches_explicit_formula(self, rng):
        """|k| = 2π sqrt((u² + v²)/D_M² + η²/|dχ/dν|²), with η varying fastest."""
        uv = rng.normal(size=(2, 4))
        eta = rng.normal(size=3) * 1e-6
        d_m, d_los = 9000.0, 1e-5
        expected = [
            2 * np.pi * np.sqrt((uv[0, i] / d_m) ** 2 + (uv[1, i] / d_m) ** 2 + (eta[e] / d_los) ** 2)
            for i in range(4)
            for e in range(3)
        ]
        np.testing.assert_allclose(generate_k_modes_in_physical_coordinates(uv, eta, d_m, d_los), expected, rtol=1e-13)

    @pytest.mark.parametrize("uv_shape, eta_shape", [((3, 4), (2,)), ((4,), (2,)), ((2, 4), (2, 1))])
    def test_rejects_bad_shapes(self, uv_shape, eta_shape):
        """uv must be (2, n) and η one-dimensional."""
        with pytest.raises(ValueError):
            generate_k_modes_in_physical_coordinates(np.ones(uv_shape), np.ones(eta_shape), 1.0, 1.0)


class TestSphericalBinEdges:
    @pytest.mark.parametrize("num_k_bins, growth", [(1, 1.35), (5, 1.35), (8, 2.0)])
    def test_geometric_edges(self, num_k_bins, growth):
        """Upper edges are 2Δk∥ gᶦ and each lower edge is the previous upper edge."""
        low, high = _spherical_bin_edges(L_PARA, num_k_bins, growth)
        expected_high = 2 * DK_PARA * growth ** np.arange(num_k_bins)
        np.testing.assert_allclose(high, expected_high, rtol=1e-13)
        np.testing.assert_allclose(low, np.concatenate([[0.0], expected_high[:-1]]), rtol=1e-13)

    def test_defaults(self):
        """The defaults are 50 bins with ratio 1.35."""
        _, high = _spherical_bin_edges(L_PARA)
        assert high.shape == (50,)
        assert high[1] / high[0] == pytest.approx(1.35)

    def test_rejects_non_positive_bin_count(self):
        """A bin count below one raises ValueError."""
        with pytest.raises(ValueError, match="num_k_bins"):
            _spherical_bin_edges(L_PARA, 0)

    @pytest.mark.parametrize("growth", [1.0, 0.5])
    def test_rejects_non_growing_edges(self, growth):
        """A growth factor at or below one raises ValueError."""
        with pytest.raises(ValueError, match="growth_factor"):
            _spherical_bin_edges(L_PARA, 5, growth)


class TestGenerateKCubeModelSphericalBinning:
    def test_matches_direct_interval_assignment(self, rng):
        """Each mode lands in the (low, high] bin found by direct comparison."""
        _, high = _spherical_bin_edges(L_PARA, 6)
        k = rng.uniform(0.0, high[-1] * 1.3, size=200)
        bid, valid, low, high, counts = generate_k_cube_model_spherical_binning(k, L_PARA, 6)
        expected = np.array([reference_bin(x, np.asarray(low), np.asarray(high)) for x in k])
        np.testing.assert_array_equal(bid, expected)
        np.testing.assert_array_equal(valid, expected >= 0)
        np.testing.assert_array_equal(counts, [np.sum(expected == i) for i in range(6)])

    def test_upper_edges_are_closed(self):
        """A mode exactly on an upper edge belongs to that bin, not the next."""
        _, high = _spherical_bin_edges(L_PARA, 4)
        bid, valid, *_ = generate_k_cube_model_spherical_binning(np.asarray(high), L_PARA, 4)
        np.testing.assert_array_equal(bid, [0, 1, 2, 3])
        assert np.all(valid)

    def test_zero_and_overflow_are_invalid(self):
        """k = 0 and k above the last edge are invalid, with id -1."""
        _, high = _spherical_bin_edges(L_PARA, 3)
        bid, valid, *_, counts = generate_k_cube_model_spherical_binning(np.array([0.0, 2 * high[-1]]), L_PARA, 3)
        np.testing.assert_array_equal(bid, [-1, -1])
        assert not np.any(valid)
        np.testing.assert_array_equal(counts, [0, 0, 0])

    def test_output_dtypes(self):
        """Bin ids and counts are int32 and the mask is boolean."""
        bid, valid, _, _, counts = generate_k_cube_model_spherical_binning(np.array([0.1]), L_PARA, 3)
        assert bid.dtype == np.int32 and counts.dtype == np.int32 and valid.dtype == bool


class TestCalcMeanBinnedKVals:
    def test_matches_numpy_means(self, rng):
        """Each bin's value is the mean |k| of its valid modes; empty bins are NaN."""
        k = rng.uniform(0.0, 1.0, size=40)
        bid, valid, *_ = generate_k_cube_model_spherical_binning(k, L_PARA, 8)
        result = np.asarray(calc_mean_binned_k_vals(k, bid, valid, 8))
        bid, valid = np.asarray(bid), np.asarray(valid)
        for i in range(8):
            members = k[valid & (bid == i)]
            if members.size:
                assert result[i] == pytest.approx(members.mean())
            else:
                assert np.isnan(result[i])

    def test_ignores_invalid_modes(self):
        """Modes flagged invalid do not contribute, even if their id is set."""
        k = np.array([1.0, 3.0, 100.0])
        result = calc_mean_binned_k_vals(k, np.array([0, 0, 0]), np.array([True, True, False]), 2)
        assert result[0] == pytest.approx(2.0)
        assert np.isnan(result[1])


class TestCompressToNonemptyKBins:
    def test_drops_empty_bins_and_remaps(self):
        """Empty bins are removed and the remaining ids become contiguous."""
        k_full = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
        counts = np.array([2, 0, 1, 0, 1])
        bid = np.array([0, 2, 0, -1, 4])
        valid = np.array([True, True, True, False, True])
        k_vals, compact, compact_valid = compress_to_nonempty_k_bins(k_full, counts, bid, valid)
        np.testing.assert_allclose(k_vals, [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(compact, [0, 1, 0, -1, 2])
        np.testing.assert_array_equal(compact_valid, [True, True, True, False, True])

    def test_mode_in_empty_bin_becomes_invalid(self):
        """A mode whose bin has zero count is remapped to -1."""
        _, compact, compact_valid = compress_to_nonempty_k_bins(
            np.array([0.1, 0.2]), np.array([1, 0]), np.array([0, 1]), np.array([True, True])
        )
        np.testing.assert_array_equal(compact, [0, -1])
        np.testing.assert_array_equal(compact_valid, [True, False])

    def test_round_trip_with_binning(self, rng):
        """Compacted ids index the non-empty k values consistently with the original bins."""
        k = rng.uniform(0.01, 1.0, size=60)
        bid, valid, _, _, counts = generate_k_cube_model_spherical_binning(k, L_PARA, 10)
        k_full = calc_mean_binned_k_vals(k, bid, valid, 10)
        k_vals, compact, compact_valid = compress_to_nonempty_k_bins(k_full, counts, bid, valid)
        np.testing.assert_allclose(np.asarray(k_vals)[np.asarray(compact)[np.asarray(compact_valid)]],
                                   np.asarray(k_full)[np.asarray(bid)[np.asarray(valid)]])


class TestGenerateKCubeModelCylindricalBinning:
    @pytest.fixture
    def modes(self, rng):
        """Random Cartesian k components, including some with k_z = 0."""
        k_x = rng.normal(scale=0.1, size=80)
        k_y = rng.normal(scale=0.1, size=80)
        k_z = rng.normal(scale=0.3, size=80)
        k_z[:5] = 0.0
        return np.sqrt(k_x**2 + k_y**2 + k_z**2), k_z, k_y, k_x

    def test_matches_direct_assignment(self, modes):
        """Each mode lands in the (k∥, k⊥) bin found by direct comparison, flattened row-major."""
        mod_k, k_z, k_y, k_x = modes
        n_perp_edges, n_para = 5, 6
        bid, valid, kp_low, kp_high, kt_low, kt_high, counts = generate_k_cube_model_cylindrical_binning(
            mod_k, k_z, k_y, k_x, n_perp_edges, L_PARA, n_para, 1.5
        )
        k_par, k_perp = np.abs(k_z), np.hypot(k_x, k_y)
        expected = []
        for kp, kt, kz in zip(k_par, k_perp, k_z):
            i = reference_bin(kp, np.asarray(kp_low), np.asarray(kp_high))
            j = reference_bin(kt, np.asarray(kt_low), np.asarray(kt_high))
            expected.append(i * (n_perp_edges - 1) + j if (i >= 0 and j >= 0 and kz != 0) else -1)
        expected = np.array(expected)
        np.testing.assert_array_equal(bid, expected)
        np.testing.assert_array_equal(valid, expected >= 0)
        assert counts.shape == (n_para, n_perp_edges - 1)
        np.testing.assert_array_equal(np.asarray(counts).reshape(-1), np.bincount(expected[expected >= 0], minlength=n_para * (n_perp_edges - 1)))

    def test_zero_kz_excluded(self, modes):
        """Modes on the k_z = 0 plane are never valid."""
        mod_k, k_z, k_y, k_x = modes
        _, valid, *_ = generate_k_cube_model_cylindrical_binning(mod_k, k_z, k_y, k_x, 5, L_PARA)
        assert not np.any(np.asarray(valid)[:5])

    def test_k_perp_edges_span_padded_range(self, modes):
        """k⊥ edges run linearly from 0.99 min to 1.01 max of the data."""
        mod_k, k_z, k_y, k_x = modes
        *_, kt_low, kt_high, _ = generate_k_cube_model_cylindrical_binning(mod_k, k_z, k_y, k_x, 5, L_PARA)
        k_perp = np.hypot(k_x, k_y)
        edges = np.linspace(0.99 * k_perp.min(), 1.01 * k_perp.max(), 5)
        np.testing.assert_allclose(kt_low, edges[:-1])
        np.testing.assert_allclose(kt_high, edges[1:])

    def test_k_parallel_edges_use_growth_factor(self, modes):
        """k∥ edges are the geometric spherical edges with the parallel growth factor."""
        mod_k, k_z, k_y, k_x = modes
        _, _, kp_low, kp_high, *_ = generate_k_cube_model_cylindrical_binning(mod_k, k_z, k_y, k_x, 5, L_PARA, 4, 1.7)
        low, high = _spherical_bin_edges(L_PARA, 4, 1.7)
        np.testing.assert_allclose(kp_low, low)
        np.testing.assert_allclose(kp_high, high)

    @pytest.mark.parametrize("n_perp", [0, 1])
    def test_rejects_too_few_perp_edges(self, n_perp):
        """Fewer than two k⊥ edge points raises ValueError."""
        with pytest.raises(ValueError, match="n_k_perp_bins"):
            generate_k_cube_model_cylindrical_binning(np.ones(2), np.ones(2), np.ones(2), np.ones(2), n_perp, L_PARA)
