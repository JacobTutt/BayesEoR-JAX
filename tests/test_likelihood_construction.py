"""Compare staged likelihood preparation with direct complex linear algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.model.construction.operator_construction import assemble_joint_operator
from bayeseor_jax.model.construction import likelihood_products as products
from bayeseor_jax.model.construction.likelihood_products import (
    build_gram_matrix, build_data_projection, build_data_power,
    save_gram_matrix, save_data_projection, save_data_power,
    save_likelihood_products,
)
from bayeseor_jax.model.construction.storage import ensure_cached, save_array


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(7, 5)) + 1j * rng.normal(size=(7, 5))
    weights = np.linspace(.5, 2., 7)
    data = rng.normal(size=7) + 1j * rng.normal(size=7)
    np.save(tmp_path / "T.npy", matrix)
    np.save(tmp_path / "Ninv.npy", weights)
    return matrix, weights, data


@pytest.mark.parametrize("block_cols,right_batch", [(1, 1), (2, 2), (3, 4), (16, 1)])
def test_saved_products_match_direct_contractions(tmp_path, inputs, block_cols, right_batch):
    matrix, weights, data = inputs
    t_path, n_path = tmp_path / "T.npy", tmp_path / "Ninv.npy"
    g_path = save_gram_matrix(t_path, n_path, output_path=tmp_path / "G.npy",
                              block_cols=block_cols, right_block_batch=right_batch)
    d_path = save_data_projection(t_path, n_path, data, output_path=tmp_path / "dbar.npy")
    p_path = save_data_power(n_path, data, output_path=tmp_path / "power.npy")
    np.testing.assert_allclose(np.load(g_path), matrix.conj().T @ (weights[:, None] * matrix), atol=1e-12)
    np.testing.assert_allclose(np.load(d_path), matrix.conj().T @ (weights * data), atol=1e-12)
    np.testing.assert_allclose(np.load(p_path), np.vdot(data, weights * data).real, atol=1e-12)
    assert not list(tmp_path.glob("*.partial.npy"))
    assert not list(tmp_path.glob("*.ready"))


def test_pure_products_are_jittable(inputs):
    matrix, weights, data = map(jnp.asarray, inputs)
    np.testing.assert_allclose(jax.jit(build_gram_matrix)(matrix, weights),
                               np.asarray(matrix).conj().T @ (np.asarray(weights)[:, None] * np.asarray(matrix)))
    np.testing.assert_allclose(jax.jit(build_data_projection)(matrix, weights, data),
                               np.asarray(matrix).conj().T @ (np.asarray(weights) * np.asarray(data)))
    np.testing.assert_allclose(jax.jit(build_data_power)(weights, data), np.vdot(data, weights * data).real)


@pytest.mark.parametrize("n_parameter", [0, 5])
def test_shared_products_load_once_and_reuse_gram(tmp_path, inputs, monkeypatch, n_parameter):
    matrix, weights, data = inputs
    matrix = matrix[:, :n_parameter]
    t_path, n_path = tmp_path / "T.npy", tmp_path / "Ninv.npy"
    np.save(t_path, matrix)
    loads, noise_loads, builds = [], [], []
    load_operator = products._load_sharded_operator
    load_noise = products._load_inverse_noise
    write_gram = products._save_gram_matrix

    def track_load(*args):
        loaded = load_operator(*args)
        assert loaded[0].size == jax.device_count()
        assert len(loaded[1].addressable_shards) == jax.local_device_count()
        loads.append(True)
        return loaded

    def track_noise(*args):
        noise_loads.append(True)
        return load_noise(*args)

    def track_build(*args):
        builds.append(True)
        return write_gram(*args)

    monkeypatch.setattr(products, "_load_sharded_operator", track_load)
    monkeypatch.setattr(products, "_load_inverse_noise", track_noise)
    monkeypatch.setattr(products, "_save_gram_matrix", track_build)

    def prepare(observations, **kwargs):
        paths = save_likelihood_products(t_path, n_path, observations, output_directory=tmp_path,
                                         block_cols=2, right_block_batch=2, **kwargs)
        np.testing.assert_allclose(np.load(paths[0]), matrix.conj().T @ (weights[:, None] * matrix), atol=1e-12)
        np.testing.assert_allclose(np.load(paths[1]), matrix.conj().T @ (weights * observations), atol=1e-12)
        np.testing.assert_allclose(np.load(paths[2]), np.vdot(observations, weights * observations).real)
        return paths

    paths = prepare(data)
    assert len(loads) == len(noise_loads) == len(builds) == 1
    stamp = paths[0].stat().st_mtime_ns
    prepare(2 * data)
    assert len(loads) == len(noise_loads) == 2
    assert len(builds) == 1
    assert paths[0].stat().st_mtime_ns == stamp
    weights = weights * 2
    np.save(n_path, weights)
    prepare(data)
    matrix = matrix * 3
    np.save(t_path, matrix)
    prepare(data)
    prepare(data, clobber=True)
    assert len(loads) == len(noise_loads) == 5
    assert len(builds) == 4


@pytest.mark.parametrize("bad_data", [[1.], [np.nan] * 7])
def test_shared_products_validate_before_writing(tmp_path, inputs, bad_data):
    output = tmp_path / "T_Ninv_T.npy"
    np.save(output, [[42.]])
    with pytest.raises(ValueError, match="data must"):
        save_likelihood_products(tmp_path / "T.npy", tmp_path / "Ninv.npy", bad_data,
                                 output_directory=tmp_path)
    np.testing.assert_array_equal(np.load(output), [[42.]])
    assert not (tmp_path / "dbar.npy").exists()


@pytest.mark.parametrize("input_name", ["T_Ninv_T.npy", "dbar.npy", "d_ninv_d.npy"])
def test_shared_outputs_cannot_overwrite_operator(tmp_path, inputs, input_name):
    matrix, _, data = inputs
    t_path = tmp_path / input_name
    np.save(t_path, matrix)
    with pytest.raises(ValueError, match="overwrite"):
        save_likelihood_products(t_path, tmp_path / "Ninv.npy", data, output_directory=tmp_path)
    np.testing.assert_array_equal(np.load(t_path), matrix)


@pytest.mark.parametrize("bad_noise", [[1.], [0.] * 7, [np.nan] * 7, [1j] * 7])
def test_bad_inputs_preserve_completed_outputs(tmp_path, inputs, bad_noise):
    output = tmp_path / "G.npy"
    np.save(output, np.array([[42.]]))
    np.save(tmp_path / "Ninv.npy", bad_noise)
    with pytest.raises(ValueError, match="Inverse noise"):
        save_gram_matrix(tmp_path / "T.npy", tmp_path / "Ninv.npy", output_path=output)
    np.testing.assert_array_equal(np.load(output), [[42.]])


def test_outputs_cannot_overwrite_inputs(tmp_path, inputs):
    matrix, weights, data = inputs
    t_path, n_path = tmp_path / "T.npy", tmp_path / "Ninv.npy"
    for output in (t_path, n_path):
        with pytest.raises(ValueError, match="overwrite"):
            save_gram_matrix(t_path, n_path, output_path=output)
        with pytest.raises(ValueError, match="overwrite"):
            save_data_projection(t_path, n_path, data, output_path=output)
    with pytest.raises(ValueError, match="overwrite"):
        save_data_power(n_path, data, output_path=n_path)
    np.testing.assert_array_equal(np.load(t_path), matrix)
    np.testing.assert_array_equal(np.load(n_path), weights)


def test_empty_coefficient_space_and_changed_data(tmp_path, inputs):
    _, weights, data = inputs
    t_path, n_path = tmp_path / "T.npy", tmp_path / "Ninv.npy"
    np.save(t_path, np.empty((7, 0), dtype=np.complex128))
    result = save_gram_matrix(t_path, n_path, output_path=tmp_path / "G.npy")
    assert np.load(result).shape == (0, 0)
    projected = save_data_projection(t_path, n_path, data, output_path=tmp_path / "dbar.npy")
    assert np.load(projected).shape == (0,)
    power = save_data_power(n_path, data, output_path=tmp_path / "power.npy")
    original = np.load(power)
    save_data_power(n_path, 2 * data, output_path=power)
    np.testing.assert_allclose(np.load(power), 4 * original)


def test_cache_tracks_selected_foreground_and_changed_files(tmp_path, inputs):
    matrix, weights, _ = inputs
    cosmological = tmp_path / "T_21.npy"
    original = tmp_path / "T_fg.npy"
    reduced = tmp_path / "T_fg_reduced.npy"
    np.save(cosmological, matrix[:, :2])
    np.save(original, matrix[:, 2:])
    np.save(reduced, matrix[:, 2:3])
    joint, gram = tmp_path / "joint.npy", tmp_path / "G.npy"
    n_path = tmp_path / "Ninv.npy"

    def prepare(foreground):
        ensure_cached(joint, lambda: assemble_joint_operator(cosmological, foreground, output_path=joint),
                      dependencies=(cosmological, foreground))
        ensure_cached(gram, lambda: save_gram_matrix(joint, n_path, output_path=gram),
                      dependencies=(joint, n_path))

    prepare(original)
    stamp = gram.stat().st_mtime_ns
    prepare(original)
    assert gram.stat().st_mtime_ns == stamp
    prepare(reduced)
    assert np.load(joint).shape == (7, 3)
    # The same component filename can also be rebuilt with changed responses.
    np.save(reduced, 2 * matrix[:, 2:3])
    prepare(reduced)
    expected = np.concatenate([matrix[:, :2], 2 * matrix[:, 2:3]], axis=1)
    np.testing.assert_allclose(np.load(gram), expected.conj().T @ (weights[:, None] * expected), atol=1e-12)
    # Returning to the original basis must not reuse the reduced joint model.
    prepare(original)
    np.testing.assert_allclose(np.load(joint), matrix)


def test_failed_cached_build_does_not_remove_previous_output(tmp_path):
    output = tmp_path / "value.npy"
    save_array(output, np.array([1.]))

    def fail():
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        ensure_cached(output, fail, clobber=True)
    np.testing.assert_array_equal(np.load(output), [1.])
