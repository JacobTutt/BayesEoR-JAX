"""Tests for ``bayeseor_jax.model.construction.operator_construction``."""

from __future__ import annotations

import numpy as np
import pytest

from bayeseor_jax.model.construction.operator_construction import (
    T_21_build_save,
    T_fg_build_save,
    T_fg_weighted_build_save,
    _build_save,
    assemble_joint_operator,
)
from bayeseor_jax.model.operators import T_21, T_fg, T_fg_weighted

N_TIME, N_FREQ, N_BL, N_PIX = 2, 3, 4, 5
N_UV, N_ETA, N_COMP = 3, 2, 2
PIXEL_AREA = 0.01


@pytest.fixture
def inputs(rng):
    """Small random sky directions, baselines, beam and spectral factors."""
    lmn = rng.normal(size=(N_TIME, 3, N_PIX))
    lmn /= np.linalg.norm(lmn, axis=1, keepdims=True)
    uvw = rng.normal(scale=5.0, size=(N_FREQ, 3, N_BL))
    beam = rng.uniform(0.1, 1.0, size=(N_TIME, N_FREQ, N_PIX))
    f_prime = rng.normal(size=(N_PIX, N_UV)) + 1j * rng.normal(size=(N_PIX, N_UV))
    f_z = rng.normal(size=(N_FREQ, N_ETA)) + 1j * rng.normal(size=(N_FREQ, N_ETA))
    q = rng.uniform(0.5, 1.5, size=(N_FREQ, N_COMP)).astype(complex)
    prior = rng.uniform(0.5, 2.0, size=N_COMP * N_PIX)
    whitener = rng.uniform(0.5, 2.0, size=N_TIME * N_FREQ * N_BL)
    return dict(lmn=lmn, uvw=uvw, beam=beam, f_prime=f_prime, f_z=f_z, q=q,
                prior=prior, whitener=whitener)


class TestBuildSave:
    def test_matches_in_memory_operator(self, tmp_path, inputs):
        """Block-wise construction equals the full in-memory operator."""
        path = _build_save(T_fg, inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                           (inputs["q"],), tmp_path / "T.npy")
        expected = T_fg(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"])
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)

    def test_leaves_only_final_file(self, tmp_path, inputs):
        """No partial or ready-marker files remain after publication."""
        _build_save(T_fg, inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                    (inputs["q"],), tmp_path / "T.npy")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["T.npy"]

    def test_saves_complex128(self, tmp_path, inputs):
        """The saved operator is complex128."""
        path = _build_save(T_fg, inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                           (inputs["q"],), tmp_path / "T.npy")
        assert np.load(path, mmap_mode="r").dtype == np.complex128

    def test_replaces_existing_output(self, tmp_path, inputs):
        """An existing file at the output path is replaced."""
        np.save(tmp_path / "T.npy", np.zeros(3))
        path = _build_save(T_fg, inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                           (inputs["q"],), tmp_path / "T.npy")
        assert np.load(path).shape == (N_TIME * N_FREQ * N_BL, N_COMP * N_PIX)

    @pytest.mark.parametrize("area", [0.0, -1.0])
    def test_rejects_nonpositive_pixel_area(self, tmp_path, inputs, area):
        """A non-positive pixel solid angle raises ValueError."""
        with pytest.raises(ValueError, match="pixel_area_sr"):
            _build_save(T_fg, inputs["lmn"], inputs["uvw"], inputs["beam"], area,
                        (inputs["q"],), tmp_path / "T.npy")

    def test_rejects_empty_baseline_axis(self, tmp_path, inputs):
        """An observation without baselines raises ValueError."""
        with pytest.raises(ValueError, match="nonempty"):
            _build_save(T_fg, inputs["lmn"], inputs["uvw"][:, :, :0], inputs["beam"], PIXEL_AREA,
                        (inputs["q"],), tmp_path / "T.npy")


class TestT21BuildSave:
    def test_matches_in_memory_operator(self, tmp_path, inputs):
        """The saved cosmological operator equals ``T_21`` evaluated in memory."""
        path = T_21_build_save(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                               inputs["f_prime"], inputs["f_z"], output_path=tmp_path / "T_21.npy")
        expected = T_21(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                        inputs["f_prime"], inputs["f_z"])
        assert np.load(path).shape == (N_TIME * N_FREQ * N_BL, N_UV * N_ETA)
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)


class TestTfgBuildSave:
    def test_matches_in_memory_operator(self, tmp_path, inputs):
        """The saved foreground operator equals ``T_fg`` evaluated in memory."""
        path = T_fg_build_save(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA,
                               inputs["q"], output_path=tmp_path / "T_fg.npy")
        expected = T_fg(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"])
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)


class TestTfgWeightedBuildSave:
    def test_matches_in_memory_operator(self, tmp_path, inputs):
        """The saved weighted operator equals ``T_fg_weighted`` evaluated in memory."""
        path = T_fg_weighted_build_save(
            inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"],
            prior_sqrt=inputs["prior"], noise_whitener=inputs["whitener"],
            output_path=tmp_path / "A_fg.npy",
        )
        expected = T_fg_weighted(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"],
                                 prior_sqrt=inputs["prior"], noise_whitener=inputs["whitener"])
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)

    def test_equals_diagonally_weighted_physical_operator(self, tmp_path, inputs):
        """The weighted operator equals ``diag(w) T_fg diag(s)`` built independently."""
        path = T_fg_weighted_build_save(
            inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"],
            prior_sqrt=inputs["prior"], noise_whitener=inputs["whitener"],
            output_path=tmp_path / "A_fg.npy",
        )
        physical = np.asarray(T_fg(inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"]))
        expected = inputs["whitener"][:, None] * physical * inputs["prior"][None, :]
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize(
        ("prior_change", "whitener_change", "match"),
        [
            (lambda p: -p, None, "nonnegative"),
            (None, lambda w: 0.0 * w, "positive"),
            (lambda p: p[:, None], None, "diagonal weight vector"),
            (lambda p: p.astype(complex), None, "diagonal weight vector"),
            (None, lambda w: np.where(np.arange(w.size) == 0, np.nan, w), "diagonal weight vector"),
        ],
    )
    def test_rejects_invalid_weights(self, tmp_path, inputs, prior_change, whitener_change, match):
        """Negative, zero, non-vector, complex or nonfinite weights raise ValueError."""
        prior = inputs["prior"] if prior_change is None else prior_change(inputs["prior"])
        whitener = inputs["whitener"] if whitener_change is None else whitener_change(inputs["whitener"])
        with pytest.raises(ValueError, match=match):
            T_fg_weighted_build_save(
                inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"],
                prior_sqrt=prior, noise_whitener=whitener, output_path=tmp_path / "A_fg.npy",
            )

    def test_zero_prior_width_zeros_column(self, tmp_path, inputs):
        """A zero prior width produces an all-zero column."""
        prior = inputs["prior"].copy()
        prior[3] = 0.0
        path = T_fg_weighted_build_save(
            inputs["lmn"], inputs["uvw"], inputs["beam"], PIXEL_AREA, inputs["q"],
            prior_sqrt=prior, noise_whitener=inputs["whitener"], output_path=tmp_path / "A_fg.npy",
        )
        np.testing.assert_array_equal(np.load(path)[:, 3], 0.0)


@pytest.fixture
def components(tmp_path, rng):
    """Two saved complex128 component matrices with matching rows."""
    t21 = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    tfg = rng.normal(size=(7, 2)) + 1j * rng.normal(size=(7, 2))
    np.save(tmp_path / "T_21.npy", t21)
    np.save(tmp_path / "T_fg.npy", tfg)
    return t21, tfg


class TestAssembleJointOperator:
    @pytest.mark.parametrize("block_rows", [1, 3, 7, 1024])
    def test_concatenates_columns(self, tmp_path, components, block_rows):
        """The joint operator is ``[T_21, T_fg]`` for any copy-block size."""
        path = assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "T_fg.npy",
                                       output_path=tmp_path / "T.npy", block_rows=block_rows)
        np.testing.assert_array_equal(np.load(path), np.hstack(components))

    def test_omits_foreground(self, tmp_path, components):
        """Passing ``None`` for the foreground copies only ``T_21``."""
        path = assemble_joint_operator(tmp_path / "T_21.npy", None, output_path=tmp_path / "T.npy")
        np.testing.assert_array_equal(np.load(path), components[0])

    def test_omits_cosmology(self, tmp_path, components):
        """Passing ``None`` for ``T_21`` copies only the foreground operator."""
        path = assemble_joint_operator(None, tmp_path / "T_fg.npy", output_path=tmp_path / "T.npy")
        np.testing.assert_array_equal(np.load(path), components[1])

    def test_empty_foreground_block(self, tmp_path, components):
        """A reduced foreground with zero retained columns leaves ``T_21`` unchanged."""
        np.save(tmp_path / "T_fg_reduced.npy", np.zeros((7, 0), dtype=np.complex128))
        path = assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "T_fg_reduced.npy",
                                       output_path=tmp_path / "T.npy")
        np.testing.assert_array_equal(np.load(path), components[0])

    def test_inputs_unchanged(self, tmp_path, components):
        """Assembly does not modify either input file."""
        assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "T_fg.npy", output_path=tmp_path / "T.npy")
        np.testing.assert_array_equal(np.load(tmp_path / "T_21.npy"), components[0])
        np.testing.assert_array_equal(np.load(tmp_path / "T_fg.npy"), components[1])

    def test_requires_a_component(self, tmp_path):
        """At least one input operator is required."""
        with pytest.raises(ValueError, match="At least one"):
            assemble_joint_operator(None, None, output_path=tmp_path / "T.npy")

    def test_rejects_overwriting_input(self, tmp_path, components):
        """The output may not overwrite a component input."""
        with pytest.raises(ValueError, match="overwrite"):
            assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "T_fg.npy",
                                    output_path=tmp_path / "T_21.npy")

    def test_rejects_temporary_overwriting_input(self, tmp_path, components):
        """The temporary output may not overwrite a component input."""
        np.save(tmp_path / "T.partial.npy", components[0])
        with pytest.raises(ValueError, match="temporary"):
            assemble_joint_operator(tmp_path / "T.partial.npy", None, output_path=tmp_path / "T.npy")

    def test_rejects_nonpositive_block_rows(self, tmp_path, components):
        """A non-positive copy-block size raises ValueError."""
        with pytest.raises(ValueError, match="block_rows"):
            assemble_joint_operator(tmp_path / "T_21.npy", None, output_path=tmp_path / "T.npy", block_rows=0)

    def test_rejects_row_mismatch(self, tmp_path, components):
        """Components with different row counts raise ValueError."""
        np.save(tmp_path / "short.npy", np.zeros((5, 2), dtype=np.complex128))
        with pytest.raises(ValueError, match="same row count"):
            assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "short.npy",
                                    output_path=tmp_path / "T.npy")

    def test_rejects_non_matrix(self, tmp_path, components):
        """A component that is not two-dimensional raises ValueError."""
        np.save(tmp_path / "vector.npy", np.zeros(7, dtype=np.complex128))
        with pytest.raises(ValueError, match="matrices"):
            assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "vector.npy",
                                    output_path=tmp_path / "T.npy")

    def test_rejects_non_complex128(self, tmp_path, components):
        """Components stored at other precisions raise ValueError."""
        np.save(tmp_path / "single.npy", np.zeros((7, 2), dtype=np.complex64))
        with pytest.raises(ValueError, match="complex128"):
            assemble_joint_operator(tmp_path / "T_21.npy", tmp_path / "single.npy",
                                    output_path=tmp_path / "T.npy")
