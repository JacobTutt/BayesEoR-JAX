"""Tests for ``bayeseor_jax.model.construction.parallel_layout``."""

from __future__ import annotations

import dataclasses
import os

import jax
import pytest

from bayeseor_jax.model.construction.parallel_layout import (
    CPUResources,
    build_solver_mesh,
    discover_cpu_resources,
    padded_visibility_partition,
    partition_blocks,
)


class TestCPUResources:
    def test_is_frozen(self):
        """Resource records cannot be modified after creation."""
        resources = CPUResources(0, 1, jax.devices("cpu")[0], 4)
        with pytest.raises(dataclasses.FrozenInstanceError):
            resources.cpu_count = 8


class TestDiscoverCpuResources:
    def test_single_process_cpu_device(self, monkeypatch):
        """A single-process job reports rank zero and a local CPU device."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        resources = discover_cpu_resources()
        assert resources.process_index == 0
        assert resources.process_count == 1
        assert resources.device.platform == "cpu"

    @pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires CPU affinity")
    def test_uses_cpu_affinity(self, monkeypatch):
        """Without Slurm, the CPU count equals the process affinity set."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        assert discover_cpu_resources().cpu_count == len(os.sched_getaffinity(0))

    def test_slurm_allocation_limits_count(self, monkeypatch):
        """``SLURM_CPUS_PER_TASK`` caps the reported CPU count."""
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
        assert discover_cpu_resources().cpu_count == 1

    def test_large_slurm_allocation_keeps_available_count(self, monkeypatch):
        """A Slurm allocation larger than the visible CPUs does not raise the count."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        available = discover_cpu_resources().cpu_count
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(available + 100))
        assert discover_cpu_resources().cpu_count == available

    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_rejects_nonpositive_slurm_allocation(self, monkeypatch, value):
        """A non-positive ``SLURM_CPUS_PER_TASK`` raises ValueError."""
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", value)
        with pytest.raises(ValueError, match="positive"):
            discover_cpu_resources()

    def test_falls_back_without_affinity(self, monkeypatch):
        """Without ``sched_getaffinity`` the process CPU count is used."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
        assert discover_cpu_resources().cpu_count == 3

    def test_unknown_cpu_count_defaults_to_one(self, monkeypatch):
        """An undeterminable CPU count is treated as one CPU."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
        assert discover_cpu_resources().cpu_count == 1


class TestPartitionBlocks:
    @pytest.mark.parametrize("n_blocks", [0, 1, 5, 12, 13])
    @pytest.mark.parametrize("process_count", [1, 2, 3, 7])
    def test_contiguous_disjoint_cover(self, n_blocks, process_count):
        """Ranges are contiguous, disjoint, cover every block and differ by at most one."""
        ranges = [partition_blocks(n_blocks, process_count, rank) for rank in range(process_count)]
        assert ranges[0][0] == 0 and ranges[-1][1] == n_blocks
        for (_, stop), (start, _) in zip(ranges[:-1], ranges[1:]):
            assert stop == start
        sizes = [stop - start for start, stop in ranges]
        assert min(sizes) >= 0 and max(sizes) - min(sizes) <= 1

    def test_remainder_goes_to_first_ranks(self):
        """Extra blocks are assigned to the lowest ranks."""
        assert [partition_blocks(5, 3, rank) for rank in range(3)] == [(0, 2), (2, 4), (4, 5)]

    @pytest.mark.parametrize(
        ("n_blocks", "process_count", "process_index"),
        [(-1, 1, 0), (4, 0, 0), (4, 2, 2), (4, 2, -1)],
    )
    def test_rejects_invalid_arguments(self, n_blocks, process_count, process_index):
        """Negative counts, empty jobs and out-of-range ranks raise ValueError."""
        with pytest.raises(ValueError):
            partition_blocks(n_blocks, process_count, process_index)


class TestBuildSolverMesh:
    def test_single_axis_over_all_devices(self):
        """The mesh has one axis ``x`` containing every visible device."""
        mesh = build_solver_mesh()
        assert mesh.axis_names == ("x",)
        assert mesh.devices.size == len(jax.devices())


class TestPaddedVisibilityPartition:
    @pytest.mark.parametrize("n_vis", [1, 7, 16, 25])
    @pytest.mark.parametrize("process_count", [1, 2, 3])
    @pytest.mark.parametrize("local_device_count", [1, 2, 4])
    def test_stripes_cover_rows_with_even_padding(self, n_vis, process_count, local_device_count):
        """Stripes tile the rows in order, and the padded length divides the local devices."""
        covered = []
        for rank in range(process_count):
            start, stop, padded = padded_visibility_partition(n_vis, process_count, local_device_count, rank)
            assert padded % local_device_count == 0
            assert padded * process_count >= n_vis
            covered.extend(range(start, max(start, stop)))
        assert covered == list(range(n_vis))

    def test_exact_division_needs_no_padding(self):
        """Rows that divide the layout exactly receive no padding."""
        assert padded_visibility_partition(16, 2, 4, 1) == (8, 16, 8)

    def test_rank_beyond_data_is_empty(self):
        """A rank past the end of the data receives an empty stripe."""
        start, stop, padded = padded_visibility_partition(3, 4, 2, 3)
        assert stop <= start and padded == 2

    def test_clamps_nonpositive_counts(self):
        """Non-positive process and device counts are treated as one."""
        assert padded_visibility_partition(5, 0, 0, 0) == (0, 5, 5)
