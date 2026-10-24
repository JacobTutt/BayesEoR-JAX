"""Tests for ``bayeseor_jax.model.instrument.baselines``: loading the instrument model."""

from __future__ import annotations

import numpy as np
import pytest

from bayeseor_jax.io import load_numpy_dict
from bayeseor_jax.model.instrument.baselines import InstrumentModel


@pytest.fixture
def arrays(rng):
    """Random baselines (2 times, 4 baselines), redundancies and antenna pairs."""
    uvw = rng.normal(size=(2, 4, 3))
    red = rng.integers(1, 4, size=(2, 4)).astype(float)
    antpairs = [(0, 1), (0, 2), (1, 0), (2, 0)]
    return uvw, red, antpairs


def _write_separate(directory, uvw, red, antpairs=None):
    """Write the separate-file layout into ``directory``."""
    directory.mkdir(exist_ok=True)
    np.save(directory / "uvw_model.npy", uvw)
    np.save(directory / "redundancy_model.npy", red)
    if antpairs is not None:
        np.save(directory / "antpairs.npy", np.asarray(antpairs))
    return directory


class TestInstrumentModelLoading:
    """Reading the supported file layouts."""

    def test_separate_files(self, tmp_path, arrays):
        """Plain arrays in separate files are loaded unchanged."""
        uvw, red, antpairs = arrays
        model = InstrumentModel(_write_separate(tmp_path / "inst", uvw, red, antpairs))
        np.testing.assert_allclose(model.uvw_array_m, uvw)
        np.testing.assert_allclose(model.bl_red_array, red)
        assert [tuple(ap) for ap in model.antpairs] == antpairs

    def test_antpairs_are_optional(self, tmp_path, arrays):
        """Without ``antpairs.npy`` the labels are None."""
        uvw, red, _ = arrays
        assert InstrumentModel(_write_separate(tmp_path / "inst", uvw, red)).antpairs is None

    def test_antpairs_are_returned_as_a_list(self, tmp_path, arrays):
        """Array antenna pairs are converted to a Python list."""
        uvw, red, antpairs = arrays
        model = InstrumentModel(_write_separate(tmp_path / "inst", uvw, red, antpairs))
        assert isinstance(model.antpairs, list)

    def test_dictionary_payload_files(self, tiny_npy_dir):
        """BayesEoR dictionary payloads, as written by ``save_model``, are read."""
        model = InstrumentModel(tiny_npy_dir)
        np.testing.assert_allclose(model.uvw_array_m, load_numpy_dict(tiny_npy_dir / "uvw_model.npy"))
        assert model.bl_red_array.shape == (4, 12)
        assert len(model.antpairs) == 12

    def test_combined_file_takes_precedence(self, tmp_path, arrays):
        """``instrument_model.npy`` is used in preference to separate files."""
        uvw, red, antpairs = arrays
        directory = _write_separate(tmp_path / "inst", uvw, red)
        np.save(directory / "instrument_model.npy",
                {"uvw_model": 2 * uvw, "redundancy_model": red, "antpairs": antpairs})
        model = InstrumentModel(directory)
        np.testing.assert_allclose(model.uvw_array_m, 2 * uvw)
        assert model.antpairs == antpairs

    def test_combined_file_without_antpairs(self, tmp_path, arrays):
        """A combined dictionary may omit antenna pairs."""
        uvw, red, _ = arrays
        (tmp_path / "inst").mkdir()
        np.save(tmp_path / "inst" / "instrument_model.npy", {"uvw_model": uvw, "redundancy_model": red})
        assert InstrumentModel(tmp_path / "inst").antpairs is None

    def test_custom_file_names(self, tmp_path, arrays):
        """Alternative file names can be supplied."""
        uvw, red, _ = arrays
        directory = tmp_path / "inst"
        directory.mkdir()
        np.save(directory / "u.npy", uvw)
        np.save(directory / "r.npy", red)
        model = InstrumentModel(str(directory), uvw_file="u.npy", red_file="r.npy")
        assert (model.n_t, model.n_bls) == (2, 4)

    def test_trailing_singleton_redundancy_is_squeezed(self, tmp_path, arrays):
        """Redundancy shaped (n_t, n_bls, 1) is reduced to (n_t, n_bls)."""
        uvw, red, _ = arrays
        model = InstrumentModel(_write_separate(tmp_path / "inst", uvw, red[..., None]))
        assert model.bl_red_array.shape == (2, 4)

    def test_dimension_properties(self, tmp_path, arrays):
        """``n_t`` and ``n_bls`` follow the baseline array shape."""
        uvw, red, _ = arrays
        model = InstrumentModel(_write_separate(tmp_path / "inst", uvw, red))
        assert (model.n_t, model.n_bls) == (2, 4)


class TestInstrumentModelErrors:
    """Validation of paths and array shapes."""

    def test_missing_directory_raises(self, tmp_path):
        """A directory that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            InstrumentModel(tmp_path / "missing")

    def test_file_instead_of_directory_raises(self, tmp_path):
        """A file path instead of a directory raises ValueError."""
        path = tmp_path / "file.npy"
        np.save(path, np.zeros(1))
        with pytest.raises(ValueError, match="not a directory"):
            InstrumentModel(path)

    def test_missing_files_raise(self, tmp_path, arrays):
        """A directory missing the redundancy file raises FileNotFoundError."""
        uvw, _, _ = arrays
        (tmp_path / "inst").mkdir()
        np.save(tmp_path / "inst" / "uvw_model.npy", uvw)
        with pytest.raises(FileNotFoundError, match="must contain"):
            InstrumentModel(tmp_path / "inst")

    def test_combined_file_must_be_a_dictionary(self, tmp_path):
        """A combined file holding a plain array raises ValueError."""
        (tmp_path / "inst").mkdir()
        np.save(tmp_path / "inst" / "instrument_model.npy", np.zeros(3))
        with pytest.raises(ValueError, match="dictionary"):
            InstrumentModel(tmp_path / "inst")

    @pytest.mark.parametrize("shape", [(4, 3), (2, 4, 2), (1, 2, 4, 3)])
    def test_invalid_baseline_shape_raises(self, tmp_path, shape):
        """Baselines not shaped (n_t, n_bls, 3) raise ValueError."""
        with pytest.raises(ValueError, match="uvw_array_m"):
            InstrumentModel(_write_separate(tmp_path / "inst", np.zeros(shape), np.ones((2, 4))))

    @pytest.mark.parametrize("shape", [(4,), (2, 4, 2), (2, 3), (3, 4)])
    def test_invalid_redundancy_shape_raises(self, tmp_path, arrays, shape):
        """Redundancy inconsistent with the baseline axes raises ValueError."""
        uvw, _, _ = arrays
        with pytest.raises(ValueError, match="bl_red_array"):
            InstrumentModel(_write_separate(tmp_path / "inst", uvw, np.ones(shape)))
