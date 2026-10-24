"""Tests for ``bayeseor_jax.inference.multinest_paths``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bayeseor_jax.inference.multinest_paths import (
    MultiNestPathManager,
    _cleanup_multinest_path,
    _prepare_multinest_path,
)


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path, monkeypatch):
    """Run every test from ``tmp_path``, where the relative short links are created."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def long_dir(tmp_path) -> Path:
    """An output directory whose MultiNest filenames would exceed 100 characters."""
    return tmp_path / ("very_long_directory_name_" * 4) / "nested_sampling"


class TestMultiNestPathManagerInit:
    def test_short_path_used_directly(self):
        """Short outputs are passed to MultiNest unchanged."""
        manager = MultiNestPathManager("out", file_root="data-")
        assert not manager.using_short_path
        assert manager.outputfiles_basename == "out/data-"

    def test_long_path_uses_short_link(self, long_dir):
        """Outputs whose longest filename exceeds 100 characters use a hashed link."""
        manager = MultiNestPathManager(long_dir)
        assert manager.using_short_path
        assert manager.outputfiles_basename.startswith(".mn_chains_symlinks/mn_")
        assert manager.outputfiles_basename.endswith("/data-")

    def test_threshold_is_100_characters(self):
        """A basename giving exactly 100 characters stays direct, and 101 is shortened."""
        suffix = "data-post_equal_weights.dat"
        for length, expected in [(100, False), (101, True)]:
            directory = "d" * (length - len(suffix) - 1)
            assert MultiNestPathManager(directory).using_short_path is expected

    def test_link_name_is_deterministic(self, long_dir):
        """The same output directory always maps to the same link."""
        assert MultiNestPathManager(long_dir).short_link == MultiNestPathManager(long_dir).short_link
        assert MultiNestPathManager(long_dir).short_link != MultiNestPathManager(long_dir / "other").short_link

    def test_logical_basename_is_resolved(self, long_dir):
        """The logical basename points at the real, resolved output directory."""
        assert MultiNestPathManager(long_dir).logical_outputfiles_basename == (long_dir.resolve() / "data-").as_posix()

    @pytest.mark.parametrize("file_root", ["", "sub/data-"])
    def test_rejects_invalid_file_root(self, file_root):
        """The file root must be a non-empty filename prefix."""
        with pytest.raises(ValueError, match="file_root"):
            MultiNestPathManager("out", file_root=file_root)


class TestPrepare:
    def test_direct_path_creates_directory(self, tmp_path, monkeypatch):
        """Preparing a short path creates the output directory and returns its basename."""
        monkeypatch.chdir(tmp_path)  # a relative path stays short under any tmp_path
        manager = MultiNestPathManager("out")
        assert manager.prepare() == "out/data-"
        assert (tmp_path / "out").is_dir()

    def test_creates_symlink_to_output(self, long_dir):
        """A long path is reached through a symlink to the real output directory."""
        manager = MultiNestPathManager(long_dir)
        basename = manager.prepare()
        assert manager.short_link.is_symlink()
        assert manager.short_link.resolve() == long_dir.resolve()
        assert len(f"{basename}post_equal_weights.dat") <= 100

    def test_files_written_through_link_land_in_output(self, long_dir):
        """Files written via the short basename appear in the real directory."""
        basename = MultiNestPathManager(long_dir).prepare()
        Path(f"{basename}test.txt").write_text("x")
        assert (long_dir / "data-test.txt").read_text() == "x"

    def test_reuses_existing_link(self, long_dir):
        """Preparing twice keeps the same valid link."""
        MultiNestPathManager(long_dir).prepare()
        manager = MultiNestPathManager(long_dir)
        manager.prepare()
        assert manager.short_link.resolve() == long_dir.resolve()

    def test_repairs_stale_link(self, long_dir, tmp_path):
        """A link pointing elsewhere is replaced by one to the requested output."""
        manager = MultiNestPathManager(long_dir)
        manager.short_link.parent.mkdir(parents=True)
        (tmp_path / "elsewhere").mkdir()
        manager.short_link.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
        manager.prepare()
        assert manager.short_link.resolve() == long_dir.resolve()

    def test_refuses_non_symlink_collision(self, long_dir):
        """A real directory at the link path is never deleted."""
        manager = MultiNestPathManager(long_dir)
        manager.short_link.mkdir(parents=True)
        with pytest.raises(FileExistsError, match="non-symlink"):
            manager.prepare()
        assert manager.short_link.is_dir()


class TestCleanup:
    def test_removes_link_but_keeps_output(self, long_dir):
        """Cleanup removes only the link, leaving the output files."""
        manager = MultiNestPathManager(long_dir)
        Path(f"{manager.prepare()}keep.txt").write_text("x")
        manager.cleanup()
        assert not os.path.lexists(manager.short_link)
        assert (long_dir / "data-keep.txt").exists()

    def test_noop_when_not_prepared(self, long_dir):
        """Cleanup before ``prepare`` does nothing."""
        MultiNestPathManager(long_dir).cleanup()

    def test_noop_for_direct_path(self, tmp_path):
        """Direct paths have no link to remove."""
        manager = MultiNestPathManager(tmp_path / "out")
        manager.prepare()
        manager.cleanup()
        assert (tmp_path / "out").is_dir()

    def test_tolerates_missing_link(self, long_dir):
        """A link already removed by someone else is not an error."""
        manager = MultiNestPathManager(long_dir)
        manager.prepare()
        manager.short_link.unlink()
        manager.cleanup()

    def test_refuses_changed_target(self, long_dir, tmp_path):
        """A link retargeted after ``prepare`` is left in place and reported."""
        manager = MultiNestPathManager(long_dir)
        manager.prepare()
        manager.short_link.unlink()
        (tmp_path / "other").mkdir()
        manager.short_link.symlink_to(tmp_path / "other", target_is_directory=True)
        with pytest.raises(RuntimeError, match="target changed"):
            manager.cleanup()
        assert manager.short_link.is_symlink()

    def test_refuses_non_symlink(self, long_dir):
        """A real directory replacing the link is never removed."""
        manager = MultiNestPathManager(long_dir)
        manager.prepare()
        manager.short_link.unlink()
        manager.short_link.mkdir()
        with pytest.raises(RuntimeError, match="non-symlink"):
            manager.cleanup()


class TestProcessHelpers:
    def test_prepare_helper_matches_prepare(self, long_dir):
        """In a single process the helper simply prepares the path."""
        manager = MultiNestPathManager(long_dir)
        assert _prepare_multinest_path(manager) == manager.outputfiles_basename
        assert manager.short_link.is_symlink()

    def test_cleanup_helper_removes_link(self, long_dir):
        """In a single process the helper removes the prepared link."""
        manager = MultiNestPathManager(long_dir)
        _prepare_multinest_path(manager)
        _cleanup_multinest_path(manager)
        assert not os.path.lexists(manager.short_link)

    def test_cleanup_helper_ignores_direct_paths(self, tmp_path):
        """Direct paths need no cleanup."""
        manager = MultiNestPathManager(tmp_path / "out")
        _prepare_multinest_path(manager)
        _cleanup_multinest_path(manager)
        assert (tmp_path / "out").is_dir()


@pytest.fixture
def fake_distributed(monkeypatch):
    """Pretend distributed JAX is initialised, with this process as rank 0."""
    import jax
    from bayeseor_jax.inference import multinest_paths

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(multinest_paths.jax, "process_index", lambda: 0)
    monkeypatch.setattr(multinest_paths.multihost_utils, "broadcast_one_to_all", lambda x, is_source: x)
    monkeypatch.setattr(multinest_paths.multihost_utils, "sync_global_devices", lambda name: None)


class TestDistributedHelpers:
    def test_prepare_on_rank_zero_returns_basename(self, long_dir, fake_distributed):
        """In a distributed run, rank 0 prepares the link and every rank gets its basename."""
        manager = MultiNestPathManager(long_dir)
        assert _prepare_multinest_path(manager) == manager.outputfiles_basename
        assert manager.short_link.is_symlink()

    def test_prepare_failure_is_raised(self, long_dir, fake_distributed, monkeypatch):
        """A failure on rank 0 is broadcast and re-raised."""
        manager = MultiNestPathManager(long_dir)
        monkeypatch.setattr(manager, "prepare", lambda: (_ for _ in ()).throw(OSError("no link")))
        with pytest.raises(OSError, match="no link"):
            _prepare_multinest_path(manager)

    def test_cleanup_removes_link(self, long_dir, fake_distributed):
        """In a distributed run, rank 0 removes the short link after sampling."""
        manager = MultiNestPathManager(long_dir)
        _prepare_multinest_path(manager)
        _cleanup_multinest_path(manager)
        assert not manager.short_link.exists()

    def test_cleanup_failure_warns(self, long_dir, fake_distributed, monkeypatch):
        """A cleanup failure on rank 0 warns rather than failing the completed run."""
        manager = MultiNestPathManager(long_dir)
        _prepare_multinest_path(manager)
        monkeypatch.setattr(manager, "cleanup", lambda: (_ for _ in ()).throw(OSError("busy")))
        with pytest.warns(RuntimeWarning, match="could not remove"):
            _cleanup_multinest_path(manager)
