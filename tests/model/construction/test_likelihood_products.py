"""Tests for ``bayeseor_jax.model.construction.likelihood_products``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.model.construction.likelihood_products import (
    _gram_block,
    _load_inverse_noise,
    _load_sharded_operator,
    _shard_data,
    build_data_power,
    build_data_projection,
    build_gram_matrix,
    save_data_power,
    save_data_projection,
    save_gram_matrix,
    save_likelihood_products,
)

N_VIS, N_PAR = 11, 5


@pytest.fixture
def problem(rng):
    """Random complex operator, positive inverse noise and complex data."""
    operator = rng.normal(size=(N_VIS, N_PAR)) + 1j * rng.normal(size=(N_VIS, N_PAR))
    inverse_noise = rng.uniform(0.5, 2.0, size=N_VIS)
    data = rng.normal(size=N_VIS) + 1j * rng.normal(size=N_VIS)
    return operator, inverse_noise, data


@pytest.fixture
def saved(tmp_path, problem):
    """The random problem saved as ``T.npy`` and ``Ninv.npy``."""
    operator, inverse_noise, data = problem
    np.save(tmp_path / "T.npy", operator)
    np.save(tmp_path / "Ninv.npy", inverse_noise)
    return tmp_path / "T.npy", tmp_path / "Ninv.npy", data


def _dense(problem):
    """Reference products computed with explicit dense NumPy matrices."""
    operator, inverse_noise, data = problem
    n_inv = np.diag(inverse_noise)
    return (operator.conj().T @ n_inv @ operator, operator.conj().T @ n_inv @ data,
            float(np.real(data.conj() @ n_inv @ data)))


class TestBuildGramMatrix:
    def test_matches_dense_product(self, problem):
        """``G`` equals ``T^H N^-1 T`` with an explicit diagonal ``N^-1``."""
        operator, inverse_noise, _ = problem
        gram = build_gram_matrix(jnp.asarray(operator), jnp.asarray(inverse_noise))
        np.testing.assert_allclose(gram, _dense(problem)[0], rtol=1e-12)

    def test_is_hermitian_positive_semidefinite(self, problem):
        """``G`` is Hermitian with non-negative eigenvalues."""
        operator, inverse_noise, _ = problem
        gram = np.asarray(build_gram_matrix(jnp.asarray(operator), jnp.asarray(inverse_noise)))
        np.testing.assert_allclose(gram, gram.conj().T, rtol=1e-12)
        assert np.linalg.eigvalsh(gram).min() > -1e-10


class TestBuildDataProjection:
    def test_matches_dense_product(self, problem):
        """``dbar`` equals ``T^H N^-1 d``."""
        operator, inverse_noise, data = problem
        projection = build_data_projection(jnp.asarray(operator), jnp.asarray(inverse_noise), jnp.asarray(data))
        np.testing.assert_allclose(projection, _dense(problem)[1], rtol=1e-12)


class TestBuildDataPower:
    def test_matches_dense_product(self, problem):
        """The data power equals ``d^H N^-1 d`` and is real."""
        _, inverse_noise, data = problem
        power = build_data_power(jnp.asarray(inverse_noise), jnp.asarray(data))
        assert np.isrealobj(np.asarray(power))
        assert float(power) == pytest.approx(_dense(problem)[2], rel=1e-12)


class TestLoadInverseNoise:
    def test_returns_real_values(self, tmp_path):
        """A valid vector is returned as real values."""
        np.save(tmp_path / "Ninv.npy", np.array([1.0, 2.0, 3.0]) + 0j)
        values = _load_inverse_noise(tmp_path / "Ninv.npy", 3)
        assert np.isrealobj(values)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])

    def test_length_check_is_optional(self, tmp_path):
        """Without ``n_visibility`` any vector length is accepted."""
        np.save(tmp_path / "Ninv.npy", np.ones(4))
        assert _load_inverse_noise(tmp_path / "Ninv.npy").shape == (4,)

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            (np.ones((2, 2)), "vector"),
            (np.ones(5), "vector"),
            (np.array([1.0, 0.0, 1.0]), "positive"),
            (np.array([1.0, -1.0, 1.0]), "positive"),
            (np.array([1.0, np.inf, 1.0]), "finite"),
            (np.array([1.0, 1.0 + 1.0j, 1.0]), "real"),
        ],
    )
    def test_rejects_invalid_vectors(self, tmp_path, values, match):
        """Matrices, wrong lengths, non-positive, nonfinite or complex values raise ValueError."""
        np.save(tmp_path / "Ninv.npy", values)
        with pytest.raises(ValueError, match=match):
            _load_inverse_noise(tmp_path / "Ninv.npy", 3)


class TestLoadShardedOperator:
    def test_returns_padded_global_arrays(self, saved, problem):
        """The loaded arrays hold ``T`` and ``N^-1`` with zero padding only."""
        mesh, operator, inverse_noise, n_parameter, row0, row1, n_rows = _load_sharded_operator(*saved[:2])
        assert n_parameter == N_PAR and (row0, row1) == (0, N_VIS) and n_rows >= N_VIS
        loaded = np.asarray(operator)
        np.testing.assert_array_equal(loaded[:N_VIS, :N_PAR], problem[0])
        assert not np.any(loaded[N_VIS:]) and not np.any(loaded[:, N_PAR:])
        np.testing.assert_array_equal(np.asarray(inverse_noise)[:N_VIS], problem[1])
        assert mesh.axis_names == ("host", "dev")

    @pytest.mark.parametrize(
        "operator",
        [np.ones((4, 3)), np.ones(4, dtype=np.complex128), np.ones((0, 3), dtype=np.complex128)],
    )
    def test_rejects_invalid_operator(self, tmp_path, operator):
        """Non-complex128, non-matrix or empty operators raise ValueError."""
        np.save(tmp_path / "T.npy", operator)
        np.save(tmp_path / "Ninv.npy", np.ones(4))
        with pytest.raises(ValueError, match="complex128 matrix"):
            _load_sharded_operator(tmp_path / "T.npy", tmp_path / "Ninv.npy")

    def test_rejects_mismatched_noise(self, tmp_path, saved):
        """An inverse-noise vector of the wrong length raises ValueError."""
        np.save(tmp_path / "Ninv.npy", np.ones(N_VIS + 1))
        with pytest.raises(ValueError, match="matching the visibility rows"):
            _load_sharded_operator(saved[0], tmp_path / "Ninv.npy")


class TestShardData:
    def test_matches_input_with_padding(self, saved):
        """Sharded data hold the observations followed by zero padding."""
        loaded = _load_sharded_operator(*saved[:2])
        sharded = np.asarray(_shard_data(saved[2], saved[0], loaded))
        np.testing.assert_array_equal(sharded[:N_VIS], saved[2])
        assert not np.any(sharded[N_VIS:])

    @pytest.mark.parametrize(
        "change", [lambda d: d[:-1], lambda d: np.where(np.arange(d.size) == 0, np.nan, d)],
    )
    def test_rejects_invalid_data(self, saved, change):
        """Data of the wrong length or with nonfinite values raise ValueError."""
        loaded = _load_sharded_operator(*saved[:2])
        with pytest.raises(ValueError, match="finite vector"):
            _shard_data(change(saved[2]), saved[0], loaded)


class TestGramBlock:
    def test_matches_dense_block(self, saved, problem):
        """A single block equals the corresponding slice of ``T^H N^-1 T``."""
        import jax

        mesh, operator, inverse_noise, *_ = _load_sharded_operator(*saved[:2])
        with jax.set_mesh(mesh):
            block = np.asarray(_gram_block(operator, inverse_noise, 1, 2, 2, 3))
        np.testing.assert_allclose(block, _dense(problem)[0][1:3, 2:5], rtol=1e-12)


class TestSaveGramMatrix:
    @pytest.mark.parametrize(("block_cols", "batch"), [(1, 1), (2, 1), (2, 3), (3, 2), (16384, 4)])
    def test_matches_dense_for_any_blocking(self, tmp_path, saved, problem, block_cols, batch):
        """The saved Gram matrix equals the dense product for any block layout."""
        path = save_gram_matrix(*saved[:2], output_path=tmp_path / "G.npy",
                                block_cols=block_cols, right_block_batch=batch)
        np.testing.assert_allclose(np.load(path), _dense(problem)[0], rtol=1e-12)

    @pytest.mark.parametrize(("block_cols", "batch"), [(0, 1), (2, 0)])
    def test_rejects_nonpositive_blocks(self, tmp_path, saved, block_cols, batch):
        """Non-positive block sizes raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            save_gram_matrix(*saved[:2], output_path=tmp_path / "G.npy",
                             block_cols=block_cols, right_block_batch=batch)

    def test_rejects_overwriting_input(self, saved):
        """The output may not overwrite the operator."""
        with pytest.raises(ValueError, match="overwrite"):
            save_gram_matrix(*saved[:2], output_path=saved[0])


class TestSaveDataProjection:
    def test_matches_dense(self, tmp_path, saved, problem):
        """The saved projection equals ``T^H N^-1 d``."""
        path = save_data_projection(*saved, output_path=tmp_path / "dbar.npy")
        np.testing.assert_allclose(np.load(path), _dense(problem)[1], rtol=1e-12)

    def test_rejects_wrong_length_data(self, tmp_path, saved):
        """Data that do not match the operator rows raise ValueError."""
        with pytest.raises(ValueError, match="finite vector"):
            save_data_projection(*saved[:2], saved[2][:-1], output_path=tmp_path / "dbar.npy")


class TestSaveDataPower:
    def test_matches_dense(self, tmp_path, saved, problem):
        """The saved scalar equals ``d^H N^-1 d``."""
        path = save_data_power(saved[1], saved[2], output_path=tmp_path / "d_ninv_d.npy")
        assert float(np.load(path)) == pytest.approx(_dense(problem)[2], rel=1e-12)

    @pytest.mark.parametrize(
        "change", [lambda d: d[:-1], lambda d: np.where(np.arange(d.size) == 0, np.inf, d)],
    )
    def test_rejects_invalid_data(self, tmp_path, saved, change):
        """Data of the wrong length or with nonfinite values raise ValueError."""
        with pytest.raises(ValueError, match="finite vector"):
            save_data_power(saved[1], change(saved[2]), output_path=tmp_path / "d_ninv_d.npy")


class TestSaveLikelihoodProducts:
    def test_writes_all_three_products(self, tmp_path, saved, problem):
        """The Gram matrix, projection and data power equal their dense values."""
        gram, projection, power = save_likelihood_products(*saved, output_directory=tmp_path / "out")
        dense = _dense(problem)
        assert [p.name for p in (gram, projection, power)] == ["T_Ninv_T.npy", "dbar.npy", "d_ninv_d.npy"]
        np.testing.assert_allclose(np.load(gram), dense[0], rtol=1e-12)
        np.testing.assert_allclose(np.load(projection), dense[1], rtol=1e-12)
        assert float(np.load(power)) == pytest.approx(dense[2], rel=1e-12)

    def test_reuses_gram_and_refreshes_data_terms(self, tmp_path, saved, problem):
        """New data refresh the data terms while the cached Gram file is reused."""
        out = tmp_path / "out"
        gram, *_ = save_likelihood_products(*saved, output_directory=out)
        stamp = gram.stat().st_mtime_ns
        new_data = 2.0 * saved[2]
        _, projection, power = save_likelihood_products(*saved[:2], new_data, output_directory=out)
        assert gram.stat().st_mtime_ns == stamp
        dense = _dense((problem[0], problem[1], new_data))
        np.testing.assert_allclose(np.load(projection), dense[1], rtol=1e-12)
        assert float(np.load(power)) == pytest.approx(dense[2], rel=1e-12)

    def test_rebuilds_gram_when_noise_changes(self, tmp_path, saved, problem):
        """Changing the inverse noise rebuilds the cached Gram matrix."""
        out = tmp_path / "out"
        save_likelihood_products(*saved, output_directory=out)
        new_noise = np.full(N_VIS, 3.0)
        np.save(saved[1], new_noise)
        gram, *_ = save_likelihood_products(*saved, output_directory=out)
        expected = _dense((problem[0], new_noise, saved[2]))[0]
        np.testing.assert_allclose(np.load(gram), expected, rtol=1e-12)

    def test_clobber_rebuilds_gram(self, tmp_path, saved):
        """``clobber=True`` rewrites the Gram matrix."""
        out = tmp_path / "out"
        gram, *_ = save_likelihood_products(*saved, output_directory=out)
        np.save(gram, np.zeros((N_PAR, N_PAR), dtype=np.complex128))
        save_likelihood_products(*saved, output_directory=out, clobber=True)
        assert np.any(np.load(gram))

    @pytest.mark.parametrize(("block_cols", "batch"), [(0, 1), (1, 0)])
    def test_rejects_nonpositive_blocks(self, tmp_path, saved, block_cols, batch):
        """Non-positive block sizes raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            save_likelihood_products(*saved, output_directory=tmp_path, block_cols=block_cols,
                                     right_block_batch=batch)
