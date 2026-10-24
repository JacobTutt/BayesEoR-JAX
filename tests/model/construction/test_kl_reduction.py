"""Tests for ``bayeseor_jax.model.construction.kl_reduction``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bayeseor_jax.model.construction.kl_reduction import (
    T_fg_svd_build_save,
    _raise_if_any,
    _svd_mesh,
    _svd_padded_shape,
    select_foreground_rank,
)


def _reference_rank(values, n_vis, f):
    """Smallest r with sqrt(sum(s[r:]**2) / n_vis) <= f, by direct search."""
    for rank in range(len(values) + 1):
        if np.sqrt(np.sum(values[rank:] ** 2) / n_vis) <= f:
            return rank
    raise AssertionError("criterion never met")


class TestSelectForegroundRank:
    @pytest.mark.parametrize("f", [0.0, 0.05, 0.1, 0.5, 1.0, 100.0])
    def test_matches_direct_criterion(self, rng, f):
        """The rank equals a direct search of the discarded-RMS criterion."""
        values = np.sort(rng.uniform(0.0, 3.0, size=12))[::-1]
        rank, _ = select_foreground_rank(values, 20, f)
        assert rank == _reference_rank(values, 20, f)

    def test_tail_rms_values(self, rng):
        """The tail RMS lists the discarded RMS at every rank, ending at zero."""
        values = np.sort(rng.uniform(0.0, 3.0, size=6))[::-1]
        _, tail = select_foreground_rank(values, 9, 0.1)
        expected = [np.sqrt(np.sum(values[r:] ** 2) / 9) for r in range(7)]
        np.testing.assert_allclose(tail, expected, rtol=1e-12, atol=1e-15)

    def test_zero_threshold_keeps_nonzero_modes(self):
        """``f=0`` retains every mode with a nonzero singular value."""
        rank, _ = select_foreground_rank(np.array([3.0, 2.0, 1.0, 0.0, 0.0]), 10, 0.0)
        assert rank == 3

    def test_large_threshold_discards_everything(self):
        """A threshold above the total RMS retains no modes."""
        rank, _ = select_foreground_rank(np.array([0.3, 0.2, 0.1]), 100, 1.0)
        assert rank == 0

    def test_empty_spectrum(self):
        """An empty spectrum gives rank zero."""
        rank, tail = select_foreground_rank(np.array([]), 5, 0.1)
        assert rank == 0
        np.testing.assert_array_equal(tail, [0.0])

    def test_no_overflow_for_huge_values(self):
        """Very large singular values do not overflow the accumulated tail."""
        rank, tail = select_foreground_rank(np.array([1e200, 1e200]), 1, 0.1)
        assert rank == 2 and np.all(np.isfinite(tail))

    @pytest.mark.parametrize(
        ("values", "n_vis", "f", "match"),
        [
            (np.ones((2, 2)), 4, 0.1, "real vector"),
            (np.array([1.0 + 0j]), 4, 0.1, "real vector"),
            (np.array([1.0]), 0, 0.1, "n_vis"),
            (np.array([1.0]), 4, -0.1, "f finite"),
            (np.array([1.0]), 4, np.nan, "f finite"),
            (np.array([1.0, -1.0]), 4, 0.1, "non-negative"),
            (np.array([np.inf]), 4, 0.1, "finite"),
            (np.array([1.0, 2.0]), 4, 0.1, "largest to smallest"),
        ],
    )
    def test_rejects_invalid_input(self, values, n_vis, f, match):
        """Invalid spectra, counts or thresholds raise ValueError."""
        with pytest.raises(ValueError, match=match):
            select_foreground_rank(values, n_vis, f)

    def test_rejects_non_integer_count(self):
        """A non-integer visibility count raises TypeError."""
        with pytest.raises(TypeError):
            select_foreground_rank(np.array([1.0]), 4.5, 0.1)


class TestSvdMesh:
    def test_cpu_uses_single_device(self):
        """On CPU the mesh is a single device on axes ``svd_rows`` and ``svd_cols``."""
        mesh = _svd_mesh()
        assert mesh.devices.shape == (1, 1)
        assert mesh.axis_names == ("svd_rows", "svd_cols")


class TestSvdPaddedShape:
    @pytest.mark.parametrize(
        ("shape", "grid", "tile", "expected"),
        [
            ((10, 7), (1, 1), 4, (12, 8)),
            ((10, 7), (2, 3), 2, (12, 12)),
            ((16, 8), (1, 2), 4, (16, 8)),
            ((12, 8), (1, 2), 4, (16, 8)),
            ((1, 1), (2, 2), 128, (256, 256)),
        ],
    )
    def test_rounds_up_to_mesh_and_tile(self, shape, grid, tile, expected):
        """Each dimension is rounded up to a multiple of lcm(grid) times the tile."""
        assert _svd_padded_shape(shape, grid, tile) == expected


class TestRaiseIfAny:
    def test_passes_without_failure(self):
        """No failure on any process returns normally."""
        _raise_if_any(False, "unused")

    def test_raises_with_failing_ranks(self):
        """A local failure raises RuntimeError naming the failing process."""
        with pytest.raises(RuntimeError, match=r"broken; failing process indices: \[0\]"):
            _raise_if_any(True, "broken")


N_VIS, N_PAR = 12, 5


@pytest.fixture
def weighted(tmp_path, rng):
    """A random weighted operator ``A`` and inverse noise, saved to disk."""
    scales = np.array([5.0, 2.0, 1.0, 0.1, 0.01])
    a = (rng.normal(size=(N_VIS, N_PAR)) + 1j * rng.normal(size=(N_VIS, N_PAR))) * scales
    inverse_noise = rng.uniform(0.5, 2.0, size=N_VIS)
    np.save(tmp_path / "A_fg.npy", a)
    np.save(tmp_path / "Ninv.npy", inverse_noise)
    return a, inverse_noise


def _reference_covariance(a, inverse_noise, rank):
    """``T_r T_r^H`` from a NumPy SVD, which is independent of column phases."""
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    reduced = (1.0 / np.sqrt(inverse_noise))[:, None] * u[:, :rank] * s[:rank]
    return reduced @ reduced.conj().T


class TestTfgSvdBuildSave:
    @pytest.mark.parametrize("f", [0.0, 0.1, 0.5])
    def test_matches_numpy_svd(self, tmp_path, weighted, f):
        """``T_r T_r^H`` equals ``N^1/2 U_r S_r^2 U_r^H N^1/2`` from a NumPy SVD."""
        a, inverse_noise = weighted
        path = T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                   output_path=tmp_path / "T_fg_reduced.npy", f=f)
        reduced = np.load(path)
        rank = select_foreground_rank(np.linalg.svd(a, compute_uv=False), N_VIS, f)[0]
        assert reduced.shape == (N_VIS, rank) and reduced.dtype == np.complex128
        np.testing.assert_allclose(reduced @ reduced.conj().T,
                                   _reference_covariance(a, inverse_noise, rank), rtol=1e-10, atol=1e-12)

    def test_full_rank_reproduces_prior_covariance(self, tmp_path, weighted):
        """With every mode kept, ``T_r T_r^H`` equals the unwhitened ``A A^H``."""
        a, inverse_noise = weighted
        path = T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                   output_path=tmp_path / "T_fg_reduced.npy", f=0.0)
        reduced = np.load(path)
        unwhitened = (1.0 / np.sqrt(inverse_noise))[:, None] * a
        np.testing.assert_allclose(reduced @ reduced.conj().T, unwhitened @ unwhitened.conj().T,
                                   rtol=1e-10, atol=1e-12)

    def test_wide_operator(self, tmp_path, rng):
        """An operator with more columns than rows keeps at most ``n_vis`` modes."""
        a = rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))
        np.save(tmp_path / "A_fg.npy", a)
        np.save(tmp_path / "Ninv.npy", np.ones(4))
        path = T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                   output_path=tmp_path / "T_fg_reduced.npy", f=0.0)
        reduced = np.load(path)
        assert reduced.shape == (4, 4)
        np.testing.assert_allclose(reduced @ reduced.conj().T, a @ a.conj().T, rtol=1e-10)

    def test_discards_everything_below_threshold(self, tmp_path, weighted):
        """A very large threshold produces a zero-column operator."""
        path = T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                   output_path=tmp_path / "T_fg_reduced.npy", f=1e6)
        assert np.load(path).shape == (N_VIS, 0)

    def test_saves_singular_values(self, tmp_path, weighted):
        """The singular-value file matches a NumPy SVD of ``A``."""
        a, _ = weighted
        T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                            output_path=tmp_path / "T_fg_reduced.npy")
        np.testing.assert_allclose(np.load(tmp_path / "T_fg_reduced.singular_values.npy"),
                                   np.linalg.svd(a, compute_uv=False), rtol=1e-10)

    def test_saves_metadata(self, tmp_path, weighted):
        """The ``.svd.json`` file records the shape, backend, rank and discarded RMS."""
        a, _ = weighted
        path = T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                   output_path=tmp_path / "T_fg_reduced.npy", f=0.1)
        metadata = json.loads((tmp_path / "T_fg_reduced.svd.json").read_text())
        rank, tail = select_foreground_rank(np.linalg.svd(a, compute_uv=False), N_VIS, 0.1)
        assert metadata["original_shape"] == [N_VIS, N_PAR]
        assert metadata["padded_shape"] == [N_VIS, N_PAR]
        assert metadata["backend"] == "jax_cpu" and metadata["tile_size"] is None
        assert metadata["retained_rank"] == rank == np.load(path).shape[1]
        assert metadata["threshold_over_noise_rms"] == 0.1
        assert metadata["discarded_rms_over_noise_rms"] == pytest.approx(tail[rank], rel=1e-8)
        assert metadata["discarded_rms_over_noise_rms"] <= 0.1
        assert metadata["coefficient_prior"] == "standard-normal"
        assert metadata["reduced_operator_path"] == str(path.resolve())

    def test_leaves_only_products(self, tmp_path, weighted):
        """Only the inputs and the three products remain afterwards."""
        T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                            output_path=tmp_path / "T_fg_reduced.npy")
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "A_fg.npy", "Ninv.npy", "T_fg_reduced.npy",
            "T_fg_reduced.singular_values.npy", "T_fg_reduced.svd.json",
        ]

    def test_rejects_nonpositive_tile(self, tmp_path, weighted):
        """A non-positive tile size raises ValueError."""
        with pytest.raises(ValueError, match="tile_size"):
            T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                output_path=tmp_path / "out.npy", tile_size=0)

    @pytest.mark.parametrize(
        "matrix", [np.ones((4, 3)), np.ones((0, 3), dtype=np.complex128), np.ones(4, dtype=np.complex128)],
    )
    def test_rejects_invalid_operator(self, tmp_path, matrix):
        """Non-complex128, empty or non-matrix operators raise ValueError."""
        np.save(tmp_path / "A_fg.npy", matrix)
        np.save(tmp_path / "Ninv.npy", np.ones(4))
        with pytest.raises(ValueError, match="nonempty complex128 matrix"):
            T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy", output_path=tmp_path / "out.npy")

    @pytest.mark.parametrize(
        ("noise", "match"),
        [(np.ones(N_VIS + 1), "match"), (np.zeros(N_VIS), "positive"), (np.full(N_VIS, np.nan), "finite")],
    )
    def test_rejects_invalid_noise(self, tmp_path, weighted, noise, match):
        """Mismatched, non-positive or nonfinite inverse noise raises ValueError."""
        np.save(tmp_path / "Ninv.npy", noise)
        with pytest.raises(ValueError, match=match):
            T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy", output_path=tmp_path / "out.npy")

    def test_rejects_negative_threshold(self, tmp_path, weighted):
        """A negative threshold raises ValueError before decomposition."""
        with pytest.raises(ValueError, match="f finite"):
            T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy",
                                output_path=tmp_path / "out.npy", f=-1.0)

    def test_rejects_overwriting_input(self, tmp_path, weighted):
        """The output may not overwrite the weighted operator."""
        with pytest.raises(ValueError, match="overwrite"):
            T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy", output_path=tmp_path / "A_fg.npy")

    def test_rejects_nonfinite_operator(self, tmp_path, weighted):
        """Nonfinite operator entries raise RuntimeError."""
        a, _ = weighted
        a = a.copy()
        a[0, 0] = np.nan
        np.save(tmp_path / "A_fg.npy", a)
        with pytest.raises(RuntimeError, match="nonfinite"):
            T_fg_svd_build_save(tmp_path / "A_fg.npy", tmp_path / "Ninv.npy", output_path=tmp_path / "out.npy")
