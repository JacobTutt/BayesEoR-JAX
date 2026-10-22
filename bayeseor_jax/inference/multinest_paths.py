"""Safe short output paths for MultiNest, including shared-filesystem coordination."""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils


class MultiNestPathManager:
    """Hide long output directories from MultiNest behind a short symlink.

    MultiNest output filenames must not exceed 100 characters. Its longest
    filename in this pipeline is ``<file_root>post_equal_weights.dat``. If the
    requested path would exceed that limit, this class passes MultiNest a
    short path of the form ``.mn_chains_symlinks/mn_<hash>/<file_root>``.
    The symlink targets the requested output directory, so files are written
    directly to their final location without copying or staging.

    The hash is derived from the resolved output directory. Repeating a run,
    including a resumed run, therefore recreates the same link. The relative
    link directory must be on a filesystem shared by all JAX processes and
    the working directory must remain unchanged while MultiNest is running.

    Parameters
    ----------
    output_dir : pathlib.Path or str
        Real directory in which MultiNest products should remain.
    file_root : str, optional
        MultiNest filename prefix. It must be a filename rather than a path.

    Notes
    -----
    ``prepare`` must be called before using ``outputfiles_basename`` and
    ``cleanup`` after MultiNest has finished reading its products. Cleanup
    removes only a verified symlink and never removes the real output data.
    """

    def __init__(
        self,
        output_dir: Path | str,
        file_root: str = "data-",
    ) -> None:
        if not file_root:
            raise ValueError("MultiNest file_root must not be empty.")
        if Path(file_root).name != file_root:
            raise ValueError(
                "MultiNest file_root must be a filename prefix, not a path: "
                f"{file_root!r}"
            )

        self.output_dir = Path(output_dir)
        self.real_output_dir = self.output_dir.expanduser().resolve()
        self.file_root = file_root
        self.logical_outputfiles_basename = (
            self.real_output_dir / self.file_root
        ).as_posix()

        direct_basename = (self.output_dir / self.file_root).as_posix()
        self.using_short_path = (
            len(f"{direct_basename}post_equal_weights.dat") > 100
        )

        digest = hashlib.sha256(os.fsencode(self.real_output_dir)).hexdigest()[:16]
        self.short_link = Path(".mn_chains_symlinks") / f"mn_{digest}"
        basename_directory = (
            self.short_link if self.using_short_path else self.output_dir
        )
        self.outputfiles_basename = (
            basename_directory / self.file_root
        ).as_posix()
        self._prepared = False

    def _resolved_link_target(self) -> Path:
        """Resolve the current symlink target for safety comparisons."""
        target = Path(os.readlink(self.short_link))
        if not target.is_absolute():
            target = self.short_link.parent / target
        return target.resolve()

    def prepare(self) -> str:
        """Prepare the output path and return the basename for MultiNest.

        Short paths are returned unchanged. For long paths, the method creates
        or reuses the deterministic symlink. A stale symlink is repaired, but
        a non-symlink collision is never deleted. The final length check is a
        guard against future changes to the fixed short-link layout.

        Returns
        -------
        str
            The ``outputfiles_basename`` to pass to ``pymultinest.solve``.
        """
        self.real_output_dir.mkdir(parents=True, exist_ok=True)
        if not self.using_short_path:
            self._prepared = True
            return self.outputfiles_basename

        self.short_link.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(self.short_link):
            if not self.short_link.is_symlink():
                raise FileExistsError(
                    "Refusing to replace non-symlink MultiNest short path: "
                    f"{self.short_link}"
                )
            if self._resolved_link_target() != self.real_output_dir:
                self.short_link.unlink()

        if not os.path.lexists(self.short_link):
            self.short_link.symlink_to(
                self.real_output_dir,
                target_is_directory=True,
            )

        longest_path = f"{self.outputfiles_basename}post_equal_weights.dat"
        if len(longest_path) > 100:
            raise ValueError(
                "The short MultiNest path still exceeds 100 characters: "
                f"{longest_path}"
            )

        self._prepared = True
        return self.outputfiles_basename

    def cleanup(self) -> None:
        """Remove the temporary link without touching the output directory.

        The link is removed only when this object prepared a shortened path,
        the path is still a symlink, and it still targets the expected real
        output directory. Any unexpected filesystem object is left intact and
        reported as an error.
        """
        if not self._prepared or not self.using_short_path:
            return
        if not os.path.lexists(self.short_link):
            self._prepared = False
            return
        if not self.short_link.is_symlink():
            raise RuntimeError(
                "Refusing to remove non-symlink MultiNest short path: "
                f"{self.short_link}"
            )
        if self._resolved_link_target() != self.real_output_dir:
            raise RuntimeError(
                "Refusing to remove a MultiNest short link whose target changed: "
                f"{self.short_link}"
            )
        self.short_link.unlink()
        self._prepared = False


def _prepare_multinest_path(path_manager: MultiNestPathManager) -> str:
    """Prepare a MultiNest path safely across JAX processes.

    In a normal single-process run this simply calls ``path_manager.prepare``.
    In a distributed run, process zero performs the filesystem mutation and
    broadcasts whether it succeeded before any process raises an exception.
    This ordering prevents other processes hanging at a barrier when link
    creation fails. All processes then wait until the shared link is visible
    and use the same deterministic basename.
    """
    distributed = bool(
        getattr(jax, "distributed", None) is not None
        and jax.distributed.is_initialized()
    )
    if not distributed:
        return path_manager.prepare()

    process_index = int(jax.process_index())
    source_error: Exception | None = None
    if process_index == 0:
        try:
            path_manager.prepare()
        except Exception as exc:  # broadcast failure before any rank raises
            source_error = exc

    source_ok = np.asarray(1 if source_error is None else 0, dtype=np.int32)
    source_ok = multihost_utils.broadcast_one_to_all(
        source_ok,
        is_source=process_index == 0,
    )
    if not bool(np.asarray(source_ok).item()):
        if source_error is not None:
            raise source_error
        raise RuntimeError(
            "JAX process 0 could not prepare the shared MultiNest short path."
        )

    multihost_utils.sync_global_devices("multinest-short-path-ready")
    return path_manager.outputfiles_basename


def _cleanup_multinest_path(path_manager: MultiNestPathManager) -> None:
    """Remove a short link after every sampler process has finished.

    Direct output paths require no work. In distributed runs, all processes
    first confirm that MultiNest has returned; process zero then removes the
    verified symlink and broadcasts the cleanup result. Cleanup failure emits
    a warning rather than invalidating an otherwise completed inference run.
    """
    if not path_manager.using_short_path:
        return

    distributed = bool(
        getattr(jax, "distributed", None) is not None
        and jax.distributed.is_initialized()
    )
    if not distributed:
        path_manager.cleanup()
        return

    multihost_utils.sync_global_devices("multinest-short-path-finished")
    process_index = int(jax.process_index())
    source_error: Exception | None = None
    if process_index == 0:
        try:
            path_manager.cleanup()
        except Exception as exc:  # cleanup must not invalidate completed samples
            source_error = exc

    source_ok = np.asarray(1 if source_error is None else 0, dtype=np.int32)
    source_ok = multihost_utils.broadcast_one_to_all(
        source_ok,
        is_source=process_index == 0,
    )
    if not bool(np.asarray(source_ok).item()):
        warnings.warn(
            "MultiNest completed, but process 0 could not remove its short "
            f"symlink: {source_error or path_manager.short_link}",
            RuntimeWarning,
            stacklevel=2,
        )
    multihost_utils.sync_global_devices("multinest-short-path-cleaned")
