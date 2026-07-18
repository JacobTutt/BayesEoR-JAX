from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "bayeseor_jax"
    / "multinest_paths.py"
)
SPEC = importlib.util.spec_from_file_location("multinest_paths", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
multinest_paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multinest_paths)

MultiNestPathManager = multinest_paths.MultiNestPathManager


class MultiNestPathManagerTests(unittest.TestCase):
    def test_short_output_path_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "chains"
            manager = MultiNestPathManager(output_dir)

            basename = manager.prepare()

            self.assertFalse(manager.using_short_path)
            self.assertEqual(basename, (output_dir / "data-").as_posix())
            manager.cleanup()
            self.assertTrue(output_dir.is_dir())

    def test_long_output_path_writes_directly_into_real_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            output_dir = working_dir / ("long-output-component-" * 5)
            previous_cwd = Path.cwd()
            os.chdir(working_dir)
            try:
                manager = MultiNestPathManager(output_dir)
                basename = manager.prepare()
                checkpoint = Path(f"{basename}resume.dat")
                checkpoint.write_text("checkpoint", encoding="utf-8")

                self.assertTrue(manager.using_short_path)
                self.assertLessEqual(
                    len(f"{basename}{manager.longest_suffix}"),
                    manager.max_path_length,
                )
                self.assertEqual(
                    (output_dir / "data-resume.dat").read_text(encoding="utf-8"),
                    "checkpoint",
                )

                short_link = manager.short_link
                manager.cleanup()
                self.assertFalse(os.path.lexists(short_link))
                self.assertTrue((output_dir / "data-resume.dat").is_file())
            finally:
                os.chdir(previous_cwd)

    def test_link_name_is_deterministic_and_resume_file_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            output_dir = working_dir / ("resume-output-component-" * 5)
            previous_cwd = Path.cwd()
            os.chdir(working_dir)
            try:
                first = MultiNestPathManager(output_dir)
                first_basename = first.prepare()
                Path(f"{first_basename}resume.dat").write_text(
                    "resume", encoding="utf-8"
                )
                first_link = first.short_link
                first.cleanup()

                second = MultiNestPathManager(output_dir)
                second_basename = second.prepare()
                self.assertEqual(first_link, second.short_link)
                self.assertEqual(first_basename, second_basename)
                self.assertTrue(Path(f"{second_basename}resume.dat").is_file())
                second.cleanup()
            finally:
                os.chdir(previous_cwd)

    def test_wrong_symlink_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            output_dir = working_dir / ("target-component-" * 6)
            wrong_dir = working_dir / "wrong"
            wrong_dir.mkdir()
            previous_cwd = Path.cwd()
            os.chdir(working_dir)
            try:
                manager = MultiNestPathManager(output_dir)
                manager.short_link.parent.mkdir(parents=True)
                manager.short_link.symlink_to(wrong_dir, target_is_directory=True)

                manager.prepare()

                self.assertEqual(manager.short_link.resolve(), output_dir.resolve())
                manager.cleanup()
            finally:
                os.chdir(previous_cwd)

    def test_non_symlink_collision_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            output_dir = working_dir / ("collision-component-" * 6)
            previous_cwd = Path.cwd()
            os.chdir(working_dir)
            try:
                manager = MultiNestPathManager(output_dir)
                manager.short_link.parent.mkdir(parents=True)
                manager.short_link.write_text("keep", encoding="utf-8")

                with self.assertRaises(FileExistsError):
                    manager.prepare()

                self.assertEqual(
                    manager.short_link.read_text(encoding="utf-8"), "keep"
                )
            finally:
                os.chdir(previous_cwd)

    def test_cleanup_refuses_to_remove_changed_link(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            output_dir = working_dir / ("cleanup-component-" * 6)
            replacement_target = working_dir / "replacement"
            replacement_target.mkdir()
            previous_cwd = Path.cwd()
            os.chdir(working_dir)
            try:
                manager = MultiNestPathManager(output_dir)
                manager.prepare()
                manager.short_link.unlink()
                manager.short_link.symlink_to(
                    replacement_target, target_is_directory=True
                )

                with self.assertRaises(RuntimeError):
                    manager.cleanup()

                self.assertEqual(
                    manager.short_link.resolve(), replacement_target.resolve()
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
