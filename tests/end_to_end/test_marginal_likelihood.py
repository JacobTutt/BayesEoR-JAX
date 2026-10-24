"""End-to-end checks of the marginalised likelihood against dense references."""

from __future__ import annotations

import numpy as np
import pytest

MODELS = ("eor", "fg", "eor+fg")


@pytest.mark.parametrize("model", MODELS)
class TestSavedProducts:
    """The arrays written by setup agree with dense NumPy recomputations."""

    def test_operator_shape(self, problem, model):
        """``T.npy`` has one row per augmented visibility and one column per coefficient."""
        p = problem(model)
        assert p.T.shape == (p.d.size, int(p.setup.phi_inverse_class.n_par))

    def test_inverse_noise(self, problem, model):
        """``Ninv.npy`` is ``1/σ²`` for every unaveraged visibility, with ``σ = 1``."""
        np.testing.assert_allclose(problem(model).ninv, 1.0, rtol=1e-14)

    def test_gram_matrix(self, problem, model):
        """``T_Ninv_T.npy`` equals ``T^H N⁻¹ T``."""
        p = problem(model)
        expected = (p.T.conj().T * p.ninv) @ p.T
        np.testing.assert_allclose(np.load(p.matrix_dir / "T_Ninv_T.npy"), expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())

    def test_data_projection(self, problem, model):
        """``dbar.npy`` equals ``T^H N⁻¹ d``."""
        p = problem(model)
        expected = p.T.conj().T @ (p.ninv * p.d)
        np.testing.assert_allclose(np.load(p.matrix_dir / "dbar.npy"), expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())

    def test_data_power(self, problem, model):
        """``d_ninv_d.npy`` equals ``d^H N⁻¹ d``."""
        p = problem(model)
        expected = np.real(p.d.conj() @ (p.ninv * p.d))
        np.testing.assert_allclose(np.real(np.load(p.matrix_dir / "d_ninv_d.npy")), expected, rtol=1e-12)


@pytest.mark.parametrize("model", MODELS)
class TestLogLikelihood:
    """``MarginalisedLikelihood.log_likelihood`` reproduces the dense Gaussian marginal."""

    def test_finite(self, problem, model):
        """The log likelihood is finite at every evaluated parameter vector."""
        p = problem(model)
        values = [float(p.setup.likelihood_class.log_likelihood(t)) for t in p.thetas()]
        assert np.all(np.isfinite(values))

    def test_matches_dense_covariance(self, problem, dense, model):
        """The target equals ``-½(ln|C| + d^H C⁻¹ d)`` plus the omitted fixed terms."""
        p = problem(model)
        for theta in p.thetas():
            value = float(p.setup.likelihood_class.log_likelihood(theta))
            np.testing.assert_allclose(value, dense.log_likelihood(p, theta), rtol=0, atol=1e-8 * dense.scale(p))

    def test_matches_real_coordinate_density(self, problem, dense, model):
        """Adding ``c_fixed`` gives the Gaussian density of the independent real data."""
        p = problem(model)
        for theta in p.thetas():
            value = float(p.setup.likelihood_class.log_likelihood(theta))
            reference, _ = dense.real_log_density(p, theta)
            np.testing.assert_allclose(value + dense.fixed_constant(p), reference, rtol=0, atol=1e-8 * dense.scale(p))

    def test_real_coordinate_covariance_is_real(self, problem, dense, model):
        """The augmented covariance maps to a real covariance of ``(Re v, Im v)``."""
        p = problem(model)
        _, imaginary_fraction = dense.real_log_density(p, p.thetas()[0])
        assert imaginary_fraction < 1e-10


class TestDataVector:
    """The likelihood's data vector holds each visibility and its conjugate."""

    def test_conjugate_copies(self, problem, dense):
        """Rows of reversed baselines are the complex conjugates of the measured rows."""
        p = problem("eor")
        orig, conj = dense.conjugate_pairs(p)
        np.testing.assert_array_equal(p.d[conj], np.conj(p.d[orig]))


class TestModelComparison:
    """Joint and component likelihoods are mutually consistent."""

    def test_power_dependence(self, problem):
        """The EoR likelihood changes with the sampled power spectrum."""
        p = problem("eor")
        values = {float(p.setup.likelihood_class.log_likelihood(t)) for t in p.thetas()}
        assert len(values) == len(p.thetas())

    def test_joint_operator_concatenates_components(self, problem):
        """The joint operator is ``[T_21, T_fg]`` built from the same observation."""
        joint, eor, fg = problem("eor+fg"), problem("eor"), problem("fg")
        np.testing.assert_allclose(joint.T, np.hstack([eor.T, fg.T]), rtol=1e-12, atol=0.0)
