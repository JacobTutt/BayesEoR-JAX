"""Tests for ``bayeseor_jax.model.construction.storage``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bayeseor_jax.model.construction import storage
from bayeseor_jax.model.construction.storage import (
    _check_output_path,
    _file_signature,
    _prepare_memmap_output,
    _slice_bounds,
    _wait_for_file,
    ensure_cached,
    publish_output,
    save_array,
)


class TestCheckOutputPath:
    def test_returns_final_and_partial_paths(self, tmp_path):
        """The temporary path inserts ``.partial`` before the suffix."""
        path, temporary = _check_output_path(tmp_path / "T.npy")
        assert path == tmp_path / "T.npy"
        assert temporary == tmp_path / "T.partial.npy"

    def test_creates_parent_directory(self, tmp_path):
        """Missing parent directories are created."""
        _check_output_path(tmp_path / "a" / "b" / "T.npy")
        assert (tmp_path / "a" / "b").is_dir()

    def test_rejects_output_equal_to_input(self, tmp_path):
        """An output that would overwrite an input is refused."""
        source = tmp_path / "T.npy"
        with pytest.raises(ValueError, match="overwrite an input"):
            _check_output_path(source, source)

    def test_rejects_temporary_equal_to_input(self, tmp_path):
        """An output whose temporary file would overwrite an input is refused."""
        with pytest.raises(ValueError, match="overwrite an input"):
            _check_output_path(tmp_path / "T.npy", tmp_path / "T.partial.npy")

    def test_unrelated_inputs_are_allowed(self, tmp_path):
        """Inputs at other paths do not block the output."""
        path, _ = _check_output_path(tmp_path / "G.npy", tmp_path / "T.npy", tmp_path / "Ninv.npy")
        assert path.name == "G.npy"


class TestWaitForFile:
    def test_returns_when_file_exists(self, tmp_path):
        """An existing file returns immediately."""
        path = tmp_path / "ready"
        path.write_text("x")
        _wait_for_file(path, timeout_s=0.0)

    def test_times_out_for_missing_file(self, tmp_path):
        """A file that never appears raises TimeoutError."""
        with pytest.raises(TimeoutError, match="Timed out"):
            _wait_for_file(tmp_path / "missing", timeout_s=0.0)


class TestPrepareMemmapOutput:
    def test_allocates_writable_array(self, tmp_path):
        """The memmap has the requested shape and dtype and accepts writes."""
        path = tmp_path / "out.npy"
        output = _prepare_memmap_output(path, shape=(3, 2), dtype=np.dtype(np.complex128), sync_name="t")
        assert output.shape == (3, 2) and output.dtype == np.complex128
        output[:] = 1 + 2j
        output.flush()
        del output
        np.testing.assert_array_equal(np.load(path), np.full((3, 2), 1 + 2j))

    def test_writes_ready_token(self, tmp_path):
        """A ``.ready`` token is left beside the allocated file."""
        path = tmp_path / "out.npy"
        _prepare_memmap_output(path, shape=(1,), dtype=np.dtype(np.float64), sync_name="t")
        assert (tmp_path / "out.npy.ready").read_text()

    def test_overwrites_existing_file(self, tmp_path):
        """An existing file at the path is replaced by a fresh allocation."""
        path = tmp_path / "out.npy"
        np.save(path, np.ones(7))
        output = _prepare_memmap_output(path, shape=(2, 2), dtype=np.dtype(np.float64), sync_name="t")
        assert output.shape == (2, 2)
        np.testing.assert_array_equal(output, np.zeros((2, 2)))

    def test_replaces_stale_ready_token(self, tmp_path):
        """A stale ready token from an earlier allocation is replaced."""
        path = tmp_path / "out.npy"
        (tmp_path / "out.npy.ready").write_text("stale")
        _prepare_memmap_output(path, shape=(1,), dtype=np.dtype(np.float64), sync_name="t")
        assert (tmp_path / "out.npy.ready").read_text() != "stale"


class TestSliceBounds:
    @pytest.mark.parametrize(
        ("sl", "extent", "expected"),
        [
            (slice(None, None), 5, (0, 5)),
            (slice(2, 4), 5, (2, 4)),
            (slice(None, 3), 5, (0, 3)),
            (slice(1, None), 5, (1, 5)),
            (slice(0, 0), 0, (0, 0)),
        ],
    )
    def test_resolves_missing_bounds(self, sl, extent, expected):
        """Missing bounds become zero and the full axis length."""
        assert _slice_bounds(sl, extent) == expected


class TestPublishOutput:
    def test_moves_partial_to_final(self, tmp_path):
        """The partial file is renamed to the final path and its marker removed."""
        partial = tmp_path / "T.partial.npy"
        np.save(partial, np.arange(3))
        (tmp_path / "T.partial.npy.ready").write_text("token")
        publish_output(partial, tmp_path / "T.npy")
        assert not partial.exists()
        assert not (tmp_path / "T.partial.npy.ready").exists()
        np.testing.assert_array_equal(np.load(tmp_path / "T.npy"), np.arange(3))

    def test_replaces_existing_final(self, tmp_path):
        """An existing final file is replaced on publication."""
        final = tmp_path / "T.npy"
        np.save(final, np.zeros(2))
        partial = tmp_path / "T.partial.npy"
        np.save(partial, np.ones(2))
        publish_output(partial, final)
        np.testing.assert_array_equal(np.load(final), np.ones(2))


class TestSaveArray:
    @pytest.mark.parametrize("value", [np.arange(6.0).reshape(2, 3), np.array(3.5), np.array([1 + 1j])])
    def test_round_trip(self, tmp_path, value):
        """Saved arrays and scalars load back unchanged."""
        path = save_array(tmp_path / "x.npy", value)
        np.testing.assert_array_equal(np.load(path), value)

    def test_leaves_no_partial_file(self, tmp_path):
        """Only the final file remains after saving."""
        save_array(tmp_path / "x.npy", np.ones(2))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["x.npy"]

    def test_accepts_replicated_jax_array(self, tmp_path):
        """A single-device JAX array is fully replicated and can be saved."""
        import jax.numpy as jnp

        path = save_array(tmp_path / "x.npy", jnp.arange(4.0))
        np.testing.assert_array_equal(np.load(path), np.arange(4.0))

    def test_failed_write_keeps_existing_output(self, tmp_path, monkeypatch):
        """A failure while writing leaves a previously completed file intact."""
        path = tmp_path / "x.npy"
        np.save(path, np.zeros(3))

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(storage.np, "save", fail)
        with pytest.raises(OSError, match="disk full"):
            save_array(path, np.ones(3))
        monkeypatch.undo()
        np.testing.assert_array_equal(np.load(path), np.zeros(3))


class TestFileSignature:
    def test_reports_path_size_and_mtime(self, tmp_path):
        """The signature records the resolved path, size and modification time."""
        path = tmp_path / "a.npy"
        np.save(path, np.ones(4))
        signature = _file_signature(path)
        assert signature["path"] == str(path.resolve())
        assert signature["size"] == path.stat().st_size
        assert signature["mtime_ns"] == path.stat().st_mtime_ns

    def test_changes_when_file_rewritten(self, tmp_path):
        """Rewriting a file with different content changes its signature."""
        path = tmp_path / "a.npy"
        np.save(path, np.ones(4))
        before = _file_signature(path)
        np.save(path, np.ones(9))
        assert _file_signature(path) != before


class _Builder:
    """Callable that publishes an array and counts its calls."""

    def __init__(self, path, value=1.0):
        self.path, self.value, self.calls = path, value, 0

    def __call__(self):
        self.calls += 1
        save_array(self.path, np.full(2, self.value + self.calls))


class TestEnsureCached:
    def test_builds_missing_output(self, tmp_path):
        """A missing product is built once and its cache metadata written."""
        path = tmp_path / "G.npy"
        build = _Builder(path)
        assert ensure_cached(path, build) == path
        assert build.calls == 1
        metadata = json.loads((tmp_path / "G.npy.cache.json").read_text())
        assert metadata["output"] == _file_signature(path)
        assert metadata["inputs"] == []

    def test_reuses_existing_output_without_dependencies(self, tmp_path):
        """An existing product without dependencies is reused by filename."""
        path = tmp_path / "G.npy"
        build = _Builder(path)
        ensure_cached(path, build)
        ensure_cached(path, build)
        assert build.calls == 1

    def test_clobber_forces_rebuild(self, tmp_path):
        """``clobber=True`` rebuilds an existing product."""
        path = tmp_path / "G.npy"
        build = _Builder(path)
        ensure_cached(path, build)
        ensure_cached(path, build, clobber=True)
        assert build.calls == 2
        np.testing.assert_array_equal(np.load(path), np.full(2, 3.0))

    def test_reuses_when_dependencies_unchanged(self, tmp_path):
        """Unchanged dependency files allow reuse."""
        dependency = tmp_path / "T.npy"
        np.save(dependency, np.ones(3))
        path = tmp_path / "G.npy"
        build = _Builder(path)
        ensure_cached(path, build, dependencies=(dependency,))
        ensure_cached(path, build, dependencies=(dependency,))
        assert build.calls == 1

    def test_rebuilds_when_dependency_changes(self, tmp_path):
        """Rewriting a dependency triggers a rebuild."""
        dependency = tmp_path / "T.npy"
        np.save(dependency, np.ones(3))
        path = tmp_path / "G.npy"
        build = _Builder(path)
        ensure_cached(path, build, dependencies=(dependency,))
        np.save(dependency, np.ones(5))
        ensure_cached(path, build, dependencies=(dependency,))
        assert build.calls == 2

    def test_rebuilds_when_dependency_set_changes(self, tmp_path):
        """Selecting a different dependency file triggers a rebuild."""
        first, second = tmp_path / "T_fg.npy", tmp_path / "T_fg_reduced.npy"
        np.save(first, np.ones(3))
        np.save(second, np.ones(3))
        path = tmp_path / "T.npy"
        build = _Builder(path)
        ensure_cached(path, build, dependencies=(first,))
        ensure_cached(path, build, dependencies=(second,))
        assert build.calls == 2

    def test_rebuilds_when_metadata_corrupt(self, tmp_path):
        """Unreadable cache metadata is treated as a cache miss."""
        dependency = tmp_path / "T.npy"
        np.save(dependency, np.ones(3))
        path = tmp_path / "G.npy"
        build = _Builder(path)
        ensure_cached(path, build, dependencies=(dependency,))
        (tmp_path / "G.npy.cache.json").write_text("{not json")
        ensure_cached(path, build, dependencies=(dependency,))
        assert build.calls == 2

    def test_rebuilds_when_output_modified(self, tmp_path):
        """A product changed after caching no longer matches its metadata."""
        dependency = tmp_path / "T.npy"
        np.save(dependency, np.ones(3))
        path = tmp_path / "G.npy"
        build = _Builder(path)
        ensure_cached(path, build, dependencies=(dependency,))
        np.save(path, np.ones(10))
        ensure_cached(path, build, dependencies=(dependency,))
        assert build.calls == 2

    def test_raises_if_builder_does_not_publish(self, tmp_path):
        """A builder that writes nothing raises RuntimeError."""
        with pytest.raises(RuntimeError, match="did not publish"):
            ensure_cached(tmp_path / "G.npy", lambda: None)
