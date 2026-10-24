"""Tests for ``bayeseor_jax.io``."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from bayeseor_jax.configuration import RunConfig
from bayeseor_jax.io import (
    _coerce_args_dict,
    load_numpy_dict,
    load_sharded_npy,
    save_numpy_dict,
    write_analysis_metadata,
    write_log_files,
)


class TestCoerceArgsDict:
    """Converting run settings or metadata into a dictionary."""

    def test_run_config(self):
        """A ``RunConfig`` becomes a dictionary of all its fields."""
        result = _coerce_args_dict(RunConfig(sigma=2.0))
        assert result["sigma"] == 2.0
        assert "foreground_prior_std_mk" in result

    def test_dict_is_copied(self):
        """A dictionary is returned as a copy, not the same object."""
        original = {"a": 1}
        result = _coerce_args_dict(original)
        assert result == original and result is not original

    @pytest.mark.parametrize("bad", [None, 3, ["a"], "text"])
    def test_rejects_other_types(self, bad):
        """Anything other than a ``RunConfig`` or dictionary raises ``TypeError``."""
        with pytest.raises(TypeError):
            _coerce_args_dict(bad)


class TestWriteLogFiles:
    """Writing ``args.json`` and ``version.txt``."""

    def test_writes_args_json(self, tmp_path):
        """The settings are written as JSON to ``args.json``, creating the directory."""
        out = tmp_path / "new" / "dir"
        write_log_files({"sigma": 1.5, "path": tmp_path}, out_dir=out)
        written = json.loads((out / "args.json").read_text())
        assert written == {"sigma": 1.5, "path": str(tmp_path)}

    def test_accepts_string_directory(self, tmp_path):
        """The output directory may be given as a string."""
        write_log_files(RunConfig(), out_dir=str(tmp_path))
        assert (tmp_path / "args.json").exists()

    def test_writes_version_only_when_given(self, tmp_path):
        """``version.txt`` is written only when a version is supplied."""
        write_log_files({}, out_dir=tmp_path / "a")
        write_log_files({}, out_dir=tmp_path / "b", version="1.2.3")
        assert not (tmp_path / "a" / "version.txt").exists()
        assert (tmp_path / "b" / "version.txt").read_text() == "1.2.3\n"

    def test_existing_files_are_preserved(self, tmp_path):
        """Existing ``args.json`` and ``version.txt`` are never overwritten."""
        write_log_files({"run": 1}, out_dir=tmp_path, version="old")
        write_log_files({"run": 2}, out_dir=tmp_path, version="new")
        assert json.loads((tmp_path / "args.json").read_text()) == {"run": 1}
        assert (tmp_path / "version.txt").read_text() == "old\n"


class TestWriteAnalysisMetadata:
    """Writing the spherical k-bin metadata files."""

    @pytest.fixture
    def binning(self):
        """Six modes in two compact bins, with one invalid mode."""
        return {
            "k_vals": np.array([0.15, 0.35]),
            "mod_k_vo": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 9.0]),
            "bin_id_per_k_mode": np.array([0, 0, 1, 1, 1, 0]),
            "valid_mask": np.array([True, True, True, True, True, False]),
        }

    def test_writes_k_files(self, tmp_path, binning):
        """Representative k, bin edges and per-bin counts are written from the modes."""
        write_analysis_metadata(output_dir=tmp_path, args={}, **binning)
        np.testing.assert_allclose(np.loadtxt(tmp_path / "k-vals.txt"), [0.15, 0.35])
        # Edges are each bin's minimum valid |k| and the final bin's maximum.
        np.testing.assert_allclose(np.loadtxt(tmp_path / "k-vals-bins.txt"), [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(np.loadtxt(tmp_path / "k-vals-nsamples.txt"), [2, 3])
        assert (tmp_path / "args.json").exists()

    def test_skips_existing_without_clobber(self, tmp_path, binning):
        """Existing k-bin files are kept unless ``clobber`` is set."""
        write_analysis_metadata(output_dir=tmp_path, args={}, **binning)
        changed = dict(binning, k_vals=np.array([0.16, 0.36]))
        write_analysis_metadata(output_dir=tmp_path, args={}, **changed)
        np.testing.assert_allclose(np.loadtxt(tmp_path / "k-vals.txt"), [0.15, 0.35])
        write_analysis_metadata(output_dir=tmp_path, args={}, clobber=True, **changed)
        np.testing.assert_allclose(np.loadtxt(tmp_path / "k-vals.txt"), [0.16, 0.36])

    def test_rejects_no_bins(self, tmp_path, binning):
        """An empty list of bins raises ``ValueError``."""
        with pytest.raises(ValueError, match="zero non-empty bins"):
            write_analysis_metadata(output_dir=tmp_path, args={}, **dict(binning, k_vals=np.array([])))

    def test_rejects_empty_bin(self, tmp_path, binning):
        """A compact bin with no assigned modes raises ``ValueError``."""
        with pytest.raises(ValueError, match="no assigned modes"):
            write_analysis_metadata(
                output_dir=tmp_path, args={}, **dict(binning, k_vals=np.array([0.1, 0.2, 0.3])),
            )


class TestSaveNumpyDict:
    """Saving arrays with metadata as a dictionary payload."""

    def test_payload_contents(self, tmp_path):
        """The data, settings, version and extra metadata are stored under their keys."""
        path = tmp_path / "sub" / "array.npy"
        save_numpy_dict(path, np.arange(3), {"a": 1}, version="v1", extra={"b": 2})
        payload = np.load(path, allow_pickle=True).item()
        np.testing.assert_array_equal(payload["data"], np.arange(3))
        assert payload["args"] == {"a": 1}
        assert payload["version"] == "v1"
        assert payload["extra"] == {"b": 2}

    def test_optional_keys_omitted(self, tmp_path):
        """``version`` and ``extra`` are omitted when not supplied."""
        path = tmp_path / "array.npy"
        save_numpy_dict(path, np.ones(2), RunConfig())
        assert set(np.load(path, allow_pickle=True).item()) == {"data", "args"}

    def test_refuses_overwrite_without_clobber(self, tmp_path):
        """An existing file raises unless ``clobber`` is set."""
        path = tmp_path / "array.npy"
        save_numpy_dict(path, np.ones(2), {})
        with pytest.raises(ValueError, match="clobber"):
            save_numpy_dict(path, np.zeros(2), {})
        save_numpy_dict(str(path), np.zeros(2), {}, clobber=True)
        np.testing.assert_array_equal(load_numpy_dict(path), np.zeros(2))

    def test_rejects_non_dict_extra(self, tmp_path):
        """``extra`` must be a dictionary."""
        with pytest.raises(ValueError, match="extra"):
            save_numpy_dict(tmp_path / "a.npy", np.ones(2), {}, extra=[1, 2])


class TestLoadNumpyDict:
    """Loading plain arrays or dictionary payloads."""

    def test_round_trip_dict_payload(self, tmp_path, rng):
        """The ``data`` entry of a saved payload is returned unchanged."""
        data = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
        save_numpy_dict(tmp_path / "a.npy", data, {})
        np.testing.assert_array_equal(load_numpy_dict(tmp_path / "a.npy"), data)

    def test_plain_array(self, tmp_path):
        """A plain ``.npy`` array is returned as is."""
        np.save(tmp_path / "plain.npy", np.arange(5))
        np.testing.assert_array_equal(load_numpy_dict(str(tmp_path / "plain.npy")), np.arange(5))

    def test_dict_without_data_key(self, tmp_path):
        """A pickled dictionary without ``data`` is returned whole."""
        np.save(tmp_path / "d.npy", {"x": 1})
        assert load_numpy_dict(tmp_path / "d.npy") == {"x": 1}

    def test_pickled_non_dict_object(self, tmp_path):
        """A pickled zero-dimensional object that is not a dictionary is unwrapped."""
        np.save(tmp_path / "list.npy", np.array([(0, 1), (1, 2)], dtype=object), allow_pickle=True)
        loaded = load_numpy_dict(tmp_path / "list.npy")
        assert [tuple(row) for row in loaded] == [(0, 1), (1, 2)]

    def test_as_jax_converts_numeric(self, tmp_path):
        """``as_jax=True`` returns a JAX array for numeric payloads."""
        np.save(tmp_path / "a.npy", np.arange(4.0))
        result = load_numpy_dict(tmp_path / "a.npy", as_jax=True)
        assert isinstance(result, jax.Array)
        np.testing.assert_array_equal(np.asarray(result), np.arange(4.0))

    def test_as_jax_leaves_non_numeric(self, tmp_path):
        """``as_jax=True`` returns payloads that cannot be converted unchanged."""
        save_numpy_dict(tmp_path / "s.npy", {"name": "text"}, {})
        assert load_numpy_dict(tmp_path / "s.npy", as_jax=True) == {"name": "text"}

    def test_missing_file(self, tmp_path):
        """A missing file raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            load_numpy_dict(tmp_path / "missing.npy")


class TestLoadShardedNpy:
    """Loading ``.npy`` files directly into sharded JAX arrays."""

    @pytest.fixture
    def sharding(self):
        """Row sharding over the available CPU devices."""
        mesh = Mesh(np.array(jax.devices()), ("rows",))
        return NamedSharding(mesh, PartitionSpec("rows"))

    def test_values_and_sharding(self, tmp_path, rng, sharding):
        """The loaded array equals the file and carries the requested sharding."""
        data = rng.normal(size=(len(jax.devices()) * 3, 2))
        np.save(tmp_path / "a.npy", data)
        result = load_sharded_npy(tmp_path / "a.npy", sharding)
        np.testing.assert_array_equal(np.asarray(result), data)
        assert result.sharding == sharding

    def test_dtype_override(self, tmp_path, sharding):
        """``dtype`` converts the values while loading."""
        np.save(tmp_path / "a.npy", np.arange(len(jax.devices()) * 2, dtype=np.float32))
        result = load_sharded_npy(str(tmp_path / "a.npy"), sharding, dtype=np.complex128)
        assert result.dtype == jnp.complex128

    def test_missing_file(self, tmp_path, sharding):
        """A missing file raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            load_sharded_npy(tmp_path / "missing.npy", sharding)
