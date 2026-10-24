"""Tests for ``bayeseor_jax.inference.maxaposteriori``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.inference.likelihood_marg import MarginalisedLikelihood
from bayeseor_jax.inference.maxaposteriori import MaximumAPosteriori
from bayeseor_jax.inference.phi import PhiInverse
from bayeseor_jax.inference.prior import Prior

BOX = dict(ps_box_size_ra_mpc=1.0, ps_box_size_dec_mpc=1.0, ps_box_size_para_mpc=1.0)
INST_TO_COSMO = 20.0
K_VALS = np.array([0.3, 0.9])
BIN_ID = np.array([0, 1, 0, 1, 1])
FG_STD = np.array([0.7, 1.3])
N_DATA = 24


@pytest.fixture
def products(rng):
    """Conjugate-augmented likelihood products and prior for seven real coefficients."""
    m, p = N_DATA // 2, len(BIN_ID) + len(FG_STD)
    a = rng.normal(size=(m, p)) + 1j * rng.normal(size=(m, p))
    noise = np.tile(rng.uniform(0.5, 2.0, size=m), 2)
    v = rng.normal(size=m) + 1j * rng.normal(size=m)
    t, d = np.vstack([a, a.conj()]), np.concatenate([v, v.conj()])
    phi = PhiInverse(
        bin_id_per_k_mode=BIN_ID, valid_mask=np.ones(5, bool), k_vals=K_VALS,
        inst_to_cosmo_vol=INST_TO_COSMO, n_par=p, foreground_prior_std=FG_STD, **BOX,
    )
    return dict(
        t_ninv_t=t.conj().T @ (t / noise[:, None]),
        dbar=t.conj().T @ (d / noise),
        d_ninv_d=float(np.real(np.vdot(d, d / noise))),
        n_data=N_DATA,
        phi_inv_class=phi,
    )


def _map(products, **kwargs) -> MaximumAPosteriori:
    """MAP solver for two power-spectrum parameters, with ``k_vals`` set."""
    options = dict(products, k_vals=K_VALS, n_power_params=2)
    options.update(kwargs)
    return MaximumAPosteriori(**options)


def _reference_solve(products, rho):
    """Independent NumPy solution of ``Sigma b = dbar`` and ``Sigma`` itself."""
    precision = np.asarray(products["phi_inv_class"].evaluate(jnp.asarray(10.0 ** np.asarray(rho))))
    sigma = products["t_ninv_t"] + np.diag(precision)
    return np.linalg.solve(sigma, products["dbar"]), precision


class TestMapEstimate:
    @pytest.mark.parametrize("rho", [[0.0, 0.0], [1.2, -0.4]])
    def test_coefficients_solve_normal_equations(self, products, rho):
        """The MAP coefficients equal ``Sigma^-1 dbar`` from a dense NumPy solve."""
        coeffs, _ = _map(products).map_estimate(np.asarray(rho))
        expected, _ = _reference_solve(products, rho)
        np.testing.assert_allclose(coeffs, expected, rtol=1e-10, atol=1e-12)

    def test_log_posterior_equals_marginal_likelihood(self, products):
        """The objective returned at the MAP equals the marginalised likelihood target."""
        rho = np.array([0.5, 0.9])
        likelihood = MarginalisedLikelihood(
            **products, prior_class=Prior(np.array([[-2.0, 3.0]] * 2)), n_power_params=2,
        )
        _, log_post = _map(products).map_estimate(rho)
        np.testing.assert_allclose(log_post, float(likelihood.log_likelihood(rho)), rtol=1e-12)

    def test_return_prior_precision(self, products):
        """``return_prior_precision`` appends the diagonal of ``Psi^-1``."""
        rho = np.array([0.1, 0.3])
        result = _map(products).map_estimate(rho, return_prior_precision=True)
        assert len(result) == 3
        np.testing.assert_allclose(result[2], _reference_solve(products, rho)[1])

    def test_dmps_input_matches_theta(self, products):
        """A linear ``dmps`` input is equivalent to ``theta = log10(dmps)``."""
        dmps = np.array([3.0, 0.2])
        from_dmps, _ = _map(products).map_estimate(dmps=dmps)
        from_theta, _ = _map(products).map_estimate(np.log10(dmps))
        np.testing.assert_allclose(from_dmps, from_theta)

    def test_scalar_dmps_broadcasts(self, products):
        """A scalar ``dmps`` is applied to every bin."""
        from_scalar, _ = _map(products).map_estimate(dmps=2.0)
        from_vector, _ = _map(products).map_estimate(dmps=np.array([2.0, 2.0]))
        np.testing.assert_allclose(from_scalar, from_vector)

    def test_ps_input_converts_to_dmps(self, products):
        """A dimensional ``ps`` input is converted with ``k^3 P / 2 pi^2``."""
        ps = np.array([500.0, 40.0])
        from_ps, _ = _map(products).map_estimate(ps=ps)
        from_dmps, _ = _map(products).map_estimate(dmps=K_VALS**3 * ps / (2 * np.pi**2))
        np.testing.assert_allclose(from_ps, from_dmps)

    def test_intrinsic_noise_requires_theta(self, products):
        """With noise fitting, ``theta`` must be given explicitly."""
        with pytest.raises(ValueError, match="theta must be provided"):
            _map(products, intrinsic_noise_fitting=True, power_param_start=1).map_estimate(dmps=1.0)

    def test_intrinsic_noise_scales_noise(self, products):
        """With ``alpha = 1`` the noise-fitting objective differs only by ``-0.5 d^H N^-1 d``."""
        rho = np.array([0.2, 0.7])
        fixed = _map(products).map_estimate(rho)
        fitted = _map(products, intrinsic_noise_fitting=True, power_param_start=1).map_estimate(
            np.concatenate([[1.0], rho])
        )
        np.testing.assert_allclose(fitted[0], fixed[0], rtol=1e-12)
        np.testing.assert_allclose(fitted[1] - fixed[1], -0.5 * products["d_ninv_d"], rtol=1e-12)

    def test_uniform_prior_jacobian(self, products):
        """Flagged bins add ``ln(10) * theta`` to the objective."""
        rho = np.array([0.6, -0.2])
        plain = _map(products).map_estimate(rho)[1]
        flagged = _map(products, uniform_priors=[True, False]).map_estimate(rho)[1]
        np.testing.assert_allclose(flagged - plain, np.log(10.0) * rho[0])

    def test_foreground_only(self, products):
        """An empty parameter vector solves the fixed-prior foreground model."""
        p = len(FG_STD)
        sub = {k: products[k] for k in ("d_ninv_d", "n_data")}
        sub["t_ninv_t"] = products["t_ninv_t"][-p:, -p:]
        sub["dbar"] = products["dbar"][-p:]
        sub["phi_inv_class"] = PhiInverse(
            bin_id_per_k_mode=np.empty(0, int), valid_mask=np.empty(0, bool), k_vals=np.empty(0),
            inst_to_cosmo_vol=1.0, n_par=p, foreground_prior_std=FG_STD, **BOX,
        )
        coeffs, _ = MaximumAPosteriori(**sub).map_estimate(np.empty(0))
        expected = np.linalg.solve(sub["t_ninv_t"] + np.diag(1 / FG_STD**2), sub["dbar"])
        np.testing.assert_allclose(coeffs, expected, rtol=1e-12)


class TestLogPosteriorAtMap:
    def test_equals_second_map_output(self, products):
        """``log_posterior_at_map`` returns the objective from ``map_estimate``."""
        rho = np.array([0.4, 0.4])
        solver = _map(products)
        assert solver.log_posterior_at_map(rho) == solver.map_estimate(rho)[1]


class TestCalculateDmps:
    def test_requires_one_input(self, products):
        """Either ``ps`` or ``dmps`` must be supplied."""
        with pytest.raises(ValueError, match="must not be None"):
            _map(products).calculate_dmps()

    def test_requires_k_vals(self, products):
        """Conversions need the bin wavenumbers."""
        with pytest.raises(ValueError, match="k_vals"):
            _map(products, k_vals=None).calculate_dmps(dmps=1.0)

    @pytest.mark.parametrize("key", ["ps", "dmps"])
    def test_rejects_wrong_length(self, products, key):
        """Array inputs must have one value per bin."""
        with pytest.raises(ValueError, match="expected 2"):
            _map(products).calculate_dmps(**{key: np.ones(3)})


class TestThetaFromDmps:
    def test_takes_log10(self, products):
        """Linear power converts to ``log10`` sampler coordinates."""
        np.testing.assert_allclose(_map(products)._theta_from_dmps(np.array([10.0, 0.01])), [1.0, -2.0])

    def test_rejects_non_positive(self, products):
        """Non-positive power cannot be mapped to log space."""
        with pytest.raises(ValueError, match="strictly positive"):
            _map(products)._theta_from_dmps(np.array([1.0, 0.0]))

    def test_requires_k_vals(self, products):
        """The conversion needs ``k_vals`` to know the number of bins."""
        with pytest.raises(ValueError, match="k_vals"):
            _map(products, k_vals=None)._theta_from_dmps(1.0)

    def test_rejects_wrong_length(self, products):
        """The input must have one value per bin."""
        with pytest.raises(ValueError, match="expected 2"):
            _map(products)._theta_from_dmps(np.ones(5))


class TestConstruction:
    def test_loads_from_matrix_dir(self, products, tmp_path):
        """Saved products give the same MAP as in-memory arrays."""
        np.save(tmp_path / "T_Ninv_T.npy", products["t_ninv_t"])
        np.save(tmp_path / "dbar.npy", products["dbar"])
        np.save(tmp_path / "d_ninv_d.npy", products["d_ninv_d"])
        disk = _map(products, t_ninv_t=None, dbar=None, d_ninv_d=None, matrix_dir=tmp_path)
        rho = np.array([0.3, 0.3])
        np.testing.assert_allclose(disk.map_estimate(rho)[0], _map(products).map_estimate(rho)[0], rtol=1e-12)

    def test_rejects_both_sources(self, products, tmp_path):
        """Arrays and ``matrix_dir`` cannot both be given."""
        with pytest.raises(ValueError, match="not both"):
            _map(products, matrix_dir=tmp_path)

    def test_missing_file_raises(self, products, tmp_path):
        """A missing saved product raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="missing"):
            _map(products, t_ninv_t=None, dbar=None, d_ninv_d=None, matrix_dir=tmp_path)

    def test_missing_arrays_raise(self, products):
        """All three products are required without ``matrix_dir``."""
        with pytest.raises(ValueError, match="must all be provided"):
            _map(products, d_ninv_d=None)

    def test_rejects_wrong_uniform_prior_shape(self, products):
        """The uniform-prior mask must have one entry per power parameter."""
        with pytest.raises(ValueError, match="uniform_priors"):
            _map(products, uniform_priors=[True, True, True])
