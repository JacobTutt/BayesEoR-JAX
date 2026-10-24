"""Tests for ``bayeseor_jax.inference.likelihood_marg``.

The reference is built independently in real coordinates. With real
coefficients ``b``, independent complex visibilities ``v = A b + n`` and
complex noise variances ``n_i``, the conjugate-augmented data are
``d = (v, v*)``, ``T = (A; A*)`` and ``N = diag(n, n)``. The equivalent real
data ``y = (Re v, Im v)`` have the Gaussian marginal density
``N(y; 0, R Psi R^T + N_r)`` with ``R = (Re A; Im A)`` and
``N_r = diag(n/2, n/2)``. The implemented target omits
``c_fixed = -0.5 [d^H N^-1 d + ln det N + N_d ln pi]`` (docs/inference/marginalisation.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from bayeseor_jax.inference.likelihood_marg import MarginalisedLikelihood
from bayeseor_jax.inference.phi import PhiInverse
from bayeseor_jax.inference.prior import Prior

BOX = dict(ps_box_size_ra_mpc=1.0, ps_box_size_dec_mpc=1.0, ps_box_size_para_mpc=1.0)
INST_TO_COSMO = 20.0  # gives O(1) prior variances, keeping the dense reference well conditioned
K_VALS = np.array([0.3, 0.9])
BIN_ID = np.array([0, 1, 0, 1, 1])
VALID = np.ones(5, bool)
FG_STD = np.array([0.7, 1.3])
BOUNDS = np.array([[-1.0, 2.0], [-1.0, 2.0]])


@dataclass
class Problem:
    """A small marginalised-likelihood problem and its independent reference pieces."""

    a: np.ndarray            # (m, p) independent complex operator
    noise: np.ndarray        # (m,) complex noise variances
    v: np.ndarray            # (m,) independent complex visibilities
    phi: PhiInverse
    t_ninv_t: np.ndarray
    dbar: np.ndarray
    d_ninv_d: float
    n_data: int

    def reference_log_density(self, power, alpha2=1.0):
        """Full marginal log density of the real data at linear power and noise scale."""
        precision = np.asarray(self.phi.evaluate(jnp.asarray(power)))
        r = np.vstack([self.a.real, self.a.imag])
        y = np.concatenate([self.v.real, self.v.imag])
        cov = r @ np.diag(1.0 / precision) @ r.T + alpha2 * np.diag(np.tile(self.noise / 2.0, 2))
        return multivariate_normal(mean=np.zeros(len(y)), cov=cov).logpdf(y)

    @property
    def c_fixed(self):
        """Omitted data/noise normalisation, ``-0.5 [Q + ln det N + N_d ln pi]``."""
        log_det_n = 2.0 * np.sum(np.log(self.noise))
        return -0.5 * (self.d_ninv_d + log_det_n + self.n_data * np.log(np.pi))

    @property
    def c_noise(self):
        """Omitted constant when fitting the noise amplitude, ``-0.5 [ln det N + N_d ln pi]``."""
        return -0.5 * (2.0 * np.sum(np.log(self.noise)) + self.n_data * np.log(np.pi))


@pytest.fixture
def problem(rng) -> Problem:
    """Twelve complex visibilities and seven real coefficients (five EoR, two foreground)."""
    m, p = 12, len(BIN_ID) + len(FG_STD)
    a = rng.normal(size=(m, p)) + 1j * rng.normal(size=(m, p))
    noise = rng.uniform(0.5, 2.0, size=m)
    v = rng.normal(size=m) + 1j * rng.normal(size=m)
    t = np.vstack([a, a.conj()])
    n_full = np.tile(noise, 2)
    d = np.concatenate([v, v.conj()])
    phi = PhiInverse(
        bin_id_per_k_mode=BIN_ID, valid_mask=VALID, k_vals=K_VALS, inst_to_cosmo_vol=INST_TO_COSMO,
        n_par=p, foreground_prior_std=FG_STD, **BOX,
    )
    return Problem(
        a=a, noise=noise, v=v, phi=phi,
        t_ninv_t=t.conj().T @ (t / n_full[:, None]),
        dbar=t.conj().T @ (d / n_full),
        d_ninv_d=float(np.real(np.vdot(d, d / n_full))),
        n_data=2 * m,
    )


def _likelihood(problem, **kwargs) -> MarginalisedLikelihood:
    """Construct the likelihood from in-memory products, as setup does from files."""
    options = dict(
        t_ninv_t=problem.t_ninv_t, dbar=problem.dbar, d_ninv_d=problem.d_ninv_d,
        n_data=problem.n_data, phi_inv_class=problem.phi, prior_class=Prior(BOUNDS),
        power_param_start=0, n_power_params=2,
    )
    options.update(kwargs)
    return MarginalisedLikelihood(**options)


class TestLogLikelihood:
    @pytest.mark.parametrize("rho", [[0.0, 0.0], [-0.5, 1.5], [1.8, -0.9]])
    def test_matches_dense_gaussian_marginal(self, problem, rho):
        """The target equals the dense real Gaussian marginal density minus ``c_fixed``."""
        rho = np.asarray(rho)
        expected = problem.reference_log_density(10.0**rho) - problem.c_fixed
        np.testing.assert_allclose(float(_likelihood(problem).log_likelihood(rho)), expected, rtol=1e-10)

    def test_matches_coefficient_space_formula(self, problem):
        """The target equals ``-0.5 [ln det Sigma + ln det Psi - dbar^H Sigma^-1 dbar]``."""
        rho = np.array([0.4, 1.1])
        precision = np.asarray(problem.phi.evaluate(jnp.asarray(10.0**rho)))
        sigma = problem.t_ninv_t + np.diag(precision)
        expected = -0.5 * (
            np.linalg.slogdet(sigma)[1] - np.sum(np.log(precision))
            - np.real(np.vdot(problem.dbar, np.linalg.solve(sigma, problem.dbar)))
        )
        np.testing.assert_allclose(float(_likelihood(problem).log_likelihood(rho)), expected, rtol=1e-10)

    def test_differences_match_bayes_factor(self, problem):
        """Differences between parameter values equal the reference log-density ratio."""
        like = _likelihood(problem)
        r1, r2 = np.array([0.0, 0.5]), np.array([1.5, -0.5])
        expected = problem.reference_log_density(10.0**r1) - problem.reference_log_density(10.0**r2)
        np.testing.assert_allclose(float(like.log_likelihood(r1) - like.log_likelihood(r2)), expected, rtol=1e-9)

    def test_uniform_prior_jacobian(self, problem):
        """Bins with linear-uniform priors add ``ln(10) * theta`` to the target."""
        rho = np.array([0.7, -0.3])
        plain = float(_likelihood(problem).log_likelihood(rho))
        flagged = float(_likelihood(problem, uniform_priors=[False, True]).log_likelihood(rho))
        np.testing.assert_allclose(flagged - plain, np.log(10.0) * rho[1])

    @pytest.mark.parametrize("alpha", [0.6, 1.0, 1.7])
    def test_intrinsic_noise_fitting(self, problem, alpha):
        """With noise fitting the target equals the reference at ``alpha^2 N`` minus ``c_noise``."""
        rho = np.array([0.2, 0.9])
        like = _likelihood(problem, intrinsic_noise_fitting=True, power_param_start=1)
        expected = problem.reference_log_density(10.0**rho, alpha2=alpha**2) - problem.c_noise
        theta = np.concatenate([[alpha], rho])
        np.testing.assert_allclose(float(like.log_likelihood(theta)), expected, rtol=1e-10)

    def test_power_param_start_offset(self, problem):
        """Power parameters are read from ``theta[start:start + n]``."""
        rho = np.array([0.3, 0.6])
        shifted = _likelihood(problem, power_param_start=1)
        np.testing.assert_allclose(
            float(shifted.log_likelihood(np.concatenate([[99.0], rho]))),
            float(_likelihood(problem).log_likelihood(rho)),
        )


class TestLogPosterior:
    def test_adds_sampler_prior(self, problem):
        """The posterior target is the likelihood plus the box log prior."""
        like = _likelihood(problem)
        rho = np.array([0.5, 0.5])
        expected = float(like.log_likelihood(rho)) - np.sum(np.log(BOUNDS[:, 1] - BOUNDS[:, 0]))
        np.testing.assert_allclose(float(like.log_posterior(rho)), expected)

    def test_outside_prior_is_minus_infinity(self, problem):
        """Parameters outside the prior box give zero posterior density."""
        assert float(_likelihood(problem).log_posterior(np.array([5.0, 0.0]))) == -np.inf


class TestConstruction:
    def test_loads_from_matrix_dir(self, problem, tmp_path):
        """Saved ``T_Ninv_T``, ``dbar`` and ``d_ninv_d`` give the same target as in-memory arrays."""
        np.save(tmp_path / "T_Ninv_T.npy", problem.t_ninv_t)
        np.save(tmp_path / "dbar.npy", problem.dbar)
        np.save(tmp_path / "d_ninv_d.npy", problem.d_ninv_d)
        from_disk = _likelihood(problem, t_ninv_t=None, dbar=None, d_ninv_d=None, matrix_dir=tmp_path)
        rho = np.array([0.1, 0.2])
        np.testing.assert_allclose(
            float(from_disk.log_likelihood(rho)), float(_likelihood(problem).log_likelihood(rho)), rtol=1e-12
        )

    def test_rejects_both_sources(self, problem, tmp_path):
        """Arrays and ``matrix_dir`` cannot both be given."""
        with pytest.raises(ValueError, match="not both"):
            _likelihood(problem, matrix_dir=tmp_path)

    def test_missing_file_raises(self, problem, tmp_path):
        """A missing saved product raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="T_Ninv_T"):
            _likelihood(problem, t_ninv_t=None, dbar=None, d_ninv_d=None, matrix_dir=tmp_path)

    def test_missing_arrays_raise(self, problem):
        """All three products are required without ``matrix_dir``."""
        with pytest.raises(ValueError, match="must all be provided"):
            _likelihood(problem, dbar=None)

    def test_rejects_wrong_uniform_prior_shape(self, problem):
        """The uniform-prior mask must have one entry per power parameter."""
        with pytest.raises(ValueError, match="uniform_priors"):
            _likelihood(problem, uniform_priors=[True])

    def test_default_uniform_prior_mask_is_false(self, problem):
        """Without a mask no bin receives the Jacobian correction."""
        np.testing.assert_array_equal(_likelihood(problem).uniform_priors, [False, False])
