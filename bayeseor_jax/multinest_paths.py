"""Short, resume-safe filesystem paths for MultiNest output."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


MULTINEST_MAX_PATH_LENGTH = 100
MULTINEST_LONGEST_SUFFIX = "post_equal_weights.dat"
DEFAULT_SHORT_PATH_DIRECTORY = Path(".mn_chains_symlinks")


def validate_multinest_output_path(
    out_dir: Path,
    file_root: str,
    longest_suffix: str = MULTINEST_LONGEST_SUFFIX,
    max_path_length: int = MULTINEST_MAX_PATH_LENGTH,
) -> str:
    """Return a MultiNest basename after validating its longest output path."""
    outputfiles_basename = (Path(out_dir) / file_root).as_posix()
    longest_path = f"{outputfiles_basename}{longest_suffix}"
    if len(longest_path) > int(max_path_length):
        raise ValueError(
            "MultiNest requires output file paths of length "
            f"<= {max_path_length} characters. The derived path "
            f"'{longest_path}' has length {len(longest_path)}."
        )
    return outputfiles_basename


class MultiNestPathManager:
    """Provide a short symlink basename when a real output path is too long.

    MultiNest opens files through ``outputfiles_basename``. When the real path
    exceeds its path-length limit, this manager creates a deterministic short
    symlink to the real output directory and returns a basename below that
    link. Files therefore land directly in the requested directory without a
    copy or staging step.

    The short-link directory must be visible from every process participating
    in a distributed run. The default is relative to the shared working
    directory so that the string passed to MultiNest stays short.
    """

    def __init__(
        self,
        output_dir: Path | str,
        file_root: str = "data-",
        *,
        short_path_directory: Path | str = DEFAULT_SHORT_PATH_DIRECTORY,
        longest_suffix: str = MULTINEST_LONGEST_SUFFIX,
        max_path_length: int = MULTINEST_MAX_PATH_LENGTH,
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
        self.short_path_directory = Path(short_path_directory)
        self.longest_suffix = longest_suffix
        self.max_path_length = int(max_path_length)

        digest = hashlib.sha256(
            os.fsencode(self.real_output_dir.as_posix())
        ).hexdigest()[:16]
        self.short_link = self.short_path_directory / f"mn_{digest}"
        self._prepared = False

    @property
    def logical_outputfiles_basename(self) -> str:
        """Return the basename in the requested real output directory."""
        return (self.real_output_dir / self.file_root).as_posix()

    @property
    def direct_outputfiles_basename(self) -> str:
        """Return the basename without applying symlink indirection."""
        return (self.output_dir / self.file_root).as_posix()

    @property
    def direct_longest_path(self) -> str:
        """Return the longest expected path without symlink indirection."""
        return f"{self.direct_outputfiles_basename}{self.longest_suffix}"

    @property
    def using_short_path(self) -> bool:
        """Whether the direct path exceeds the configured MultiNest limit."""
        return len(self.direct_longest_path) > self.max_path_length

    @property
    def outputfiles_basename(self) -> str:
        """Return the basename that should be passed to MultiNest."""
        parent = self.short_link if self.using_short_path else self.output_dir
        return (parent / self.file_root).as_posix()

    def _resolved_link_target(self) -> Path:
        target = Path(os.readlink(self.short_link))
        if not target.is_absolute():
            target = self.short_link.parent / target
        return target.resolve()

    def _ensure_short_link(self) -> None:
        self.short_path_directory.mkdir(parents=True, exist_ok=True)

        if os.path.lexists(self.short_link):
            if not self.short_link.is_symlink():
                raise FileExistsError(
                    "Refusing to replace non-symlink MultiNest short path: "
                    f"{self.short_link}"
                )
            if self._resolved_link_target() == self.real_output_dir:
                return
            self.short_link.unlink()

        self.short_link.symlink_to(self.real_output_dir, target_is_directory=True)

    def prepare(self) -> str:
        """Create or verify the short link and return MultiNest's basename."""
        self.real_output_dir.mkdir(parents=True, exist_ok=True)
        if self.using_short_path:
            self._ensure_short_link()

        parent = self.short_link if self.using_short_path else self.output_dir
        outputfiles_basename = validate_multinest_output_path(
            parent,
            self.file_root,
            longest_suffix=self.longest_suffix,
            max_path_length=self.max_path_length,
        )
        self._prepared = True
        return outputfiles_basename

    def mark_prepared(self) -> str:
        """Mark a link prepared by another distributed process as usable."""
        self._prepared = True
        return self.outputfiles_basename

    def cleanup(self) -> None:
        """Remove only this manager's verified symlink, never real output."""
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
