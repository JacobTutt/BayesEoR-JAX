"""Exercise the boundary between pure JAX operators and CPU/file orchestration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax import (
    T_21, T_fg, T_21_build_save, T_fg_build_save,
    T_fg_weighted_build_save, assemble_joint_operator,
)
from bayeseor_jax.model.construction import parallel_layout, operator_construction
from bayeseor_jax.model.operators import (
    build_sky_to_visibility_operator, build_T_21_core, build_T_fg_core,
)


@pytest.fixture
def inputs():
    rng = np.random.default_rng(4)
    lmn = rng.normal(size=(2, 3, 3))
    lmn /= np.linalg.norm(lmn, axis=1, keepdims=True)
    uvw = rng.normal(size=(3, 3, 2))
    beam = rng.uniform(size=(2, 3, 3))
    transverse = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    spectral = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    foreground = np.column_stack([np.ones(3), [1, .8, .6], [1, .5, .25]])
    return lmn, uvw, beam, .2, transverse, spectral, foreground


def test_pure_compositions_work_under_jit_and_differentiation(inputs):
    lmn, uvw, beam, area, transverse, spectral, foreground = inputs
    response = build_sky_to_visibility_operator(lmn, uvw, beam, area)
    np.testing.assert_allclose(
        jax.jit(T_21)(lmn, uvw, beam, area, transverse, spectral),
        build_T_21_core(response, transverse, spectral), atol=1e-13,
    )
    compiled = jax.jit(T_fg)
    expected = build_T_fg_core(response, foreground)
    np.testing.assert_allclose(compiled(lmn, uvw, beam, area, foreground), expected, atol=1e-13)
    np.testing.assert_allclose(compiled(lmn, uvw, 2 * beam, area, foreground), 2 * expected, atol=1e-13)

    # A dynamic solid-angle argument must remain differentiable, not a static host value.
    def power(pixel_area):
        matrix = T_21(lmn, uvw, beam, pixel_area, transverse, spectral)
        return jnp.real(jnp.vdot(matrix, matrix))

    np.testing.assert_allclose(jax.jit(jax.grad(power))(area), 2 * power(area) / area)


def test_selected_inputs_reproduce_corresponding_rows_and_columns(inputs):
    lmn, uvw, beam, area, transverse, spectral, foreground = inputs
    full_21 = np.asarray(T_21(lmn, uvw, beam, area, transverse, spectral)).reshape(2, 3, 2, 4)
    selected_21 = jax.jit(T_21)(
        lmn[1:2], uvw[[0, 2], :, 1:2], beam[1:2, [0, 2]], area,
        transverse[:, 1:2], spectral[[0, 2], 1:2],
    )
    np.testing.assert_allclose(selected_21, full_21[1, [0, 2], 1, 3:4], atol=1e-13)

    full_fg = np.asarray(T_fg(lmn, uvw, beam, area, foreground)).reshape(2, 3, 2, 9)
    selected_fg = jax.jit(T_fg)(
        lmn[1:2, :, [0, 2]], uvw[[0, 2], :, 1:2],
        beam[1:2, [0, 2]][:, :, [0, 2]], area, foreground[np.ix_([0, 2], [0, 2])],
    )
    expected = full_fg[1, [0, 2], 1][:, [0, 2, 6, 8]]
    np.testing.assert_allclose(selected_fg, expected, atol=1e-13)


def test_saved_components_and_assembly_match_pure_operators(tmp_path, inputs):
    lmn, uvw, beam, area, transverse, spectral, foreground = inputs
    path_21 = T_21_build_save(lmn, uvw, beam, area, transverse, spectral, output_path=tmp_path / 'T_21.npy')
    path_fg = T_fg_build_save(lmn, uvw, beam, area, foreground, output_path=tmp_path / 'T_fg.npy')
    path = assemble_joint_operator(path_21, path_fg, output_path=tmp_path / 'T.npy', block_rows=5)
    expected_21 = np.asarray(T_21(lmn, uvw, beam, area, transverse, spectral))
    expected_fg = np.asarray(T_fg(lmn, uvw, beam, area, foreground))
    np.testing.assert_allclose(np.load(path_21), expected_21, atol=1e-13)
    np.testing.assert_allclose(np.load(path_fg), expected_fg, atol=1e-13)
    np.testing.assert_allclose(np.load(path), np.concatenate([expected_21, expected_fg], axis=1), atol=1e-13)
    assert not list(tmp_path.glob('*.partial*'))
    with pytest.raises(ValueError, match='overwrite'):
        assemble_joint_operator(path_21, path_fg, output_path=path_fg)
    np.testing.assert_allclose(np.load(path_fg), expected_fg, atol=1e-13)


def test_empty_cosmological_coefficient_axis_can_be_saved(tmp_path, inputs):
    lmn, uvw, beam, area, transverse, spectral, _ = inputs
    path = T_21_build_save(lmn, uvw, beam, area, transverse[:, :0], spectral,
                           output_path=tmp_path / 'empty.npy')
    assert np.load(path).shape == (12, 0)


@pytest.mark.parametrize("component", ["eor", "fg"])
def test_single_component_assembly_preserves_input(tmp_path, component):
    source = tmp_path / "component.npy"
    matrix = np.arange(12).reshape(4, 3).astype(np.complex128)
    np.save(source, matrix)
    paths = (source, None) if component == "eor" else (None, source)
    output = assemble_joint_operator(*paths, output_path=tmp_path / "T.npy", block_rows=3)
    np.testing.assert_array_equal(np.load(output), matrix)
    np.testing.assert_array_equal(np.load(source), matrix)
    with pytest.raises(ValueError, match="overwrite"):
        assemble_joint_operator(*paths, output_path=source)


def test_assembly_rejects_missing_components(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        assemble_joint_operator(None, None, output_path=tmp_path / "T.npy")


@pytest.mark.parametrize('uniform', [False, True])
def test_weighted_writer_matches_full_operator_without_unweighted_file(tmp_path, inputs, uniform):
    lmn, uvw, beam, area, _, _, foreground = inputs
    prior = np.ones(9) if uniform else np.linspace(0, 2, 9)
    noise = np.ones(12) if uniform else np.linspace(.2, 3, 12)
    path = T_fg_weighted_build_save(
        lmn, uvw, beam, area, foreground,
        prior_sqrt=prior, noise_whitener=noise, output_path=tmp_path / 'A_fg.npy',
    )
    expected = np.asarray(T_fg(lmn, uvw, beam, area, foreground)) * prior[None, :] * noise[:, None]
    saved = np.load(path)
    assert saved.dtype == np.complex128
    np.testing.assert_allclose(saved, expected, atol=1e-13)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('name,value', [
    ('prior_sqrt', np.eye(9)),
    ('noise_whitener', np.eye(12)),
    ('prior_sqrt', np.ones(8)),
    ('noise_whitener', np.ones(11)),
    ('prior_sqrt', -np.ones(9)),
    ('noise_whitener', np.zeros(12)),
    ('noise_whitener', -np.ones(12)),
    ('prior_sqrt', np.full(9, np.nan)),
    ('noise_whitener', np.full(12, np.inf)),
    ('prior_sqrt', np.ones(9) * 1j),
])
def test_invalid_weights_preserve_existing_output(tmp_path, inputs, name, value):
    lmn, uvw, beam, area, _, _, foreground = inputs
    path = tmp_path / 'A_fg.npy'
    np.save(path, np.array([[3j]]))
    weights = dict(prior_sqrt=np.ones(9), noise_whitener=np.ones(12))
    weights[name] = value
    with pytest.raises(ValueError, match=name):
        T_fg_weighted_build_save(lmn, uvw, beam, area, foreground, **weights, output_path=path)
    np.testing.assert_array_equal(np.load(path), [[3j]])
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('process_count', [4, 8])
def test_weighted_writer_partitions_rows_with_global_noise_indices(
    tmp_path, inputs, monkeypatch, process_count,
):
    lmn, uvw, beam, area, _, _, foreground = inputs
    prior, noise = np.arange(1., 10.), np.arange(1., 13.)
    path = tmp_path / 'A_fg.npy'
    partial = tmp_path / 'A_fg.partial.npy'
    shared = np.lib.format.open_memmap(partial, mode='w+', shape=(12, 9), dtype=np.complex128)
    shared[:] = np.nan
    shared.flush()
    del shared

    # Emulate ranks sequentially while keeping the real partitioning and CPU block evaluation.
    monkeypatch.setattr(operator_construction, '_prepare_memmap_output', lambda path, **kwargs: np.load(path, mmap_mode='r+'))
    monkeypatch.setattr(operator_construction, 'publish_output', lambda *args: None)
    for rank in range(process_count):
        resources = parallel_layout.CPUResources(rank, process_count, jax.local_devices(backend='cpu')[0], 1)
        monkeypatch.setattr(operator_construction, 'discover_cpu_resources', lambda: resources)
        before = np.load(partial).copy()
        T_fg_weighted_build_save(
            lmn, uvw, beam, area, foreground,
            prior_sqrt=prior, noise_whitener=noise, output_path=path,
        )
        start, stop = parallel_layout.partition_blocks(6, process_count, rank)
        outside = np.ones(12, dtype=bool)
        outside[start * 2:stop * 2] = False
        np.testing.assert_array_equal(np.load(partial)[outside], before[outside])
    expected = np.asarray(T_fg(lmn, uvw, beam, area, foreground)) * prior[None, :] * noise[:, None]
    np.testing.assert_allclose(np.load(partial), expected, atol=1e-13)


def test_invalid_inputs_do_not_replace_existing_operator(tmp_path, inputs):
    lmn, uvw, beam, area, _, _, foreground = inputs
    path = tmp_path / 'T_fg.npy'
    np.save(path, np.array([[3j]]))
    with pytest.raises(ValueError, match='spectral_operator'):
        T_fg_build_save(lmn, uvw, beam, area, foreground[:1], output_path=path)
    np.testing.assert_array_equal(np.load(path), [[3j]])
    assert not list(tmp_path.glob('*.partial*'))


@pytest.mark.parametrize('n_blocks,n_processes', [(6, 4), (2, 5), (0, 3), (9, 3)])
def test_time_frequency_partition_has_no_overlap_or_missing_rows(n_blocks, n_processes):
    ranges = [parallel_layout.partition_blocks(n_blocks, n_processes, rank) for rank in range(n_processes)]
    assert [i for start, stop in ranges for i in range(start, stop)] == list(range(n_blocks))
    counts = [stop - start for start, stop in ranges]
    assert max(counts) - min(counts) <= 1


def test_cpu_resource_discovery_respects_affinity_and_slurm(monkeypatch):
    monkeypatch.setattr(parallel_layout.os, 'sched_getaffinity', lambda _: {2, 3, 4, 5}, raising=False)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    resources = parallel_layout.discover_cpu_resources()
    assert resources.cpu_count == 2
    assert resources.device.platform == 'cpu'
    assert resources.process_count == jax.process_count()
    assert resources.process_index == jax.process_index()
