"""Shared setups and dense NumPy references for the end-to-end tests.

Each small model is set up once per session and reused by every test that only
reads it. The dense references are written directly from the documented
mathematics, independently of the package's cached products and solvers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayeseor_jax import run_setup
from bayeseor_jax.configuration import RunConfig

#: Sampled log10 power amplitudes, one row per evaluation, chosen where the EoR
#: prior changes the fixture's likelihood by several units rather than negligibly.
THETAS = np.array([[6.0, 7.0], [8.0, 9.0], [9.0, 7.0]])
MODELS = ("eor", "fg", "eor+fg")


def small_config(directory: Path, tiny_npy_dir: Path, latlonalt, **overrides) -> RunConfig:
    """Return the small ``.npy`` configuration used by ``tiny_config``, writing to ``directory``."""
    model = overrides.pop("model", "eor+fg")
    base = dict(
        model=model,
        data_path=tiny_npy_dir / "vis_vector.npy",
        inst_model=tiny_npy_dir,
        n_freq=4,
        f0_mhz=150.0,
        df_mhz=0.1,
        n_t=4,
        d_t=10.0,
        jd_min=2459999.1,
        telescope_latlonalt=latlonalt,
        beam_type="airy",
        diam=14.0,
        fov_eor=20.0,
        fov_fg=20.0,
        nside=16,
        sigma=1.0,
        priors=None if model == "fg" else [[-2.0, 4.0], [-2.0, 4.0]],
        array_save_directory=directory / "matrices",
        output_dir=directory / "output",
        save_dir=directory / "saved",
    )
    base.update(overrides)
    return RunConfig(**base)


def quiet_setup(config: RunConfig, **kwargs):
    """Run ``run_setup``, silencing the expected small-``nside`` resolution warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_setup(config, **kwargs)


@dataclass(frozen=True)
class Problem:
    """Dense arrays of one completed setup, loaded from its saved outputs."""

    setup: object
    matrix_dir: Path
    T: np.ndarray
    ninv: np.ndarray
    d: np.ndarray

    def thetas(self) -> list[np.ndarray]:
        """Parameter vectors to evaluate: the shared ``THETAS``, or one empty vector."""
        n = int(self.setup.likelihood_class.n_power_params)
        return [np.empty(0)] if n == 0 else [row[:n] for row in THETAS]

    def psi(self, theta: np.ndarray) -> np.ndarray:
        """Diagonal coefficient prior covariance, from ``PhiInverse`` at ``theta``."""
        return 1.0 / np.asarray(self.setup.phi_inverse_class.evaluate(10.0 ** np.asarray(theta)))


def load_problem(setup, matrix_dir: Path) -> Problem:
    """Collect the saved operator, inverse noise and data vector of a setup."""
    return Problem(
        setup=setup,
        matrix_dir=matrix_dir,
        T=np.load(matrix_dir / "T.npy"),
        ninv=np.real(np.load(matrix_dir / "Ninv.npy")),
        d=np.asarray(setup.vis_dict["vis_noisy"], dtype=np.complex128),
    )


@pytest.fixture(scope="session")
def build_setup(tiny_npy_dir, hera_latlonalt):
    """Factory running ``run_setup`` on the small configuration in ``directory``, with overrides."""

    def build(directory: Path, **overrides):
        """Return the ``SetupResult`` for the overridden small configuration."""
        return quiet_setup(small_config(directory, tiny_npy_dir, hera_latlonalt, **overrides))

    return build


@pytest.fixture(scope="session")
def problem(tmp_path_factory, tiny_npy_dir, hera_latlonalt):
    """Factory returning the cached ``Problem`` for ``model`` in {eor, fg, eor+fg}."""
    cache: dict[str, Problem] = {}

    def get(model: str) -> Problem:
        """Return the cached problem for ``model``, setting it up on first use."""
        if model not in cache:
            directory = tmp_path_factory.mktemp(model.replace("+", "_"))
            config = small_config(directory, tiny_npy_dir, hera_latlonalt, model=model)
            cache[model] = load_problem(quiet_setup(config), directory / "matrices")
        return cache[model]

    return get


@pytest.fixture(scope="session")
def kl_problem(tmp_path_factory, tiny_npy_dir, hera_latlonalt):
    """Factory returning the cached ``Problem`` for ``model`` with a KL basis built at ``kl_f``."""
    cache: dict[tuple[str, float], Problem] = {}

    def get(model: str, kl_f: float) -> Problem:
        """Return the cached KL problem for ``model`` and ``kl_f``, building it on first use."""
        key = (model, float(kl_f))
        if key not in cache:
            directory = tmp_path_factory.mktemp(f"kl_{model.replace('+', '_')}_{kl_f:g}")
            config = small_config(
                directory, tiny_npy_dir, hera_latlonalt, model=model,
                foreground_kl_path=directory / "kl" / "T_fg_reduced.npy",
            )
            setup = quiet_setup(config, build_foreground_kl=True, kl_f=kl_f)
            cache[key] = load_problem(setup, directory / "matrices")
        return cache[key]

    return get


# ---------------------------------------------------------------------------
# Dense references
# ---------------------------------------------------------------------------

def prior_predictive_covariance(problem: Problem, theta: np.ndarray) -> np.ndarray:
    """Data covariance ``C = N + T Ψ T^H`` with the coefficients marginalised."""
    T = problem.T
    return np.diag(1.0 / problem.ninv) + (T * problem.psi(theta)) @ T.conj().T


def _low_rank_gaussian(noise_std, factor, data) -> tuple[float, float]:
    """``ln|C|`` and ``x^H C⁻¹ x`` for ``C = diag(σ²) + L L^H``, via an SVD.

    Whitening gives ``I + A A^H`` with ``A = L/σ``; its eigenvalues are
    ``1 + s²`` along the left singular vectors of ``A`` and one elsewhere. This
    stays accurate when the bright foreground prior makes ``C`` ill-conditioned.
    """
    whitened = data / noise_std
    U, s, _ = np.linalg.svd(factor / noise_std[:, None], full_matrices=False)
    projected = U.conj().T @ whitened
    logdet = 2.0 * np.sum(np.log(noise_std)) + np.sum(np.log1p(s**2))
    quad = np.real(np.vdot(whitened, whitened)) - np.sum(s**2 / (1.0 + s**2) * np.abs(projected) ** 2)
    return float(logdet), float(quad)


def dense_log_likelihood(problem: Problem, theta: np.ndarray) -> float:
    """Package target from ``C``: ``-½(ln|C| + d^H C⁻¹ d) + ½(d^H N⁻¹ d + ln|N|)``."""
    noise_std = 1.0 / np.sqrt(problem.ninv)
    logdet_c, quad_c = _low_rank_gaussian(noise_std, problem.T * np.sqrt(problem.psi(theta)), problem.d)
    quad_n = np.real(problem.d.conj() @ (problem.ninv * problem.d))
    logdet_n = -np.sum(np.log(problem.ninv))
    return -0.5 * (logdet_c + quad_c) + 0.5 * (quad_n + logdet_n)


def fixed_constant(problem: Problem) -> float:
    """Omitted constant ``c_fixed = -½(d^H N⁻¹ d + ln|N| + N_d ln π)``."""
    d = problem.d
    quad_n = np.real(d.conj() @ (problem.ninv * d))
    return -0.5 * (quad_n - np.sum(np.log(problem.ninv)) + d.size * np.log(np.pi))


def conjugate_pairs(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the measured visibilities and of their conjugate copies."""
    n_bls_aug = int(problem.setup.vis_dict["uvws"].shape[1])
    half = n_bls_aug // 2
    blocks = np.arange(problem.d.size).reshape(-1, n_bls_aug)
    return blocks[:, :half].ravel(), blocks[:, half:].ravel()


def real_coordinate_log_density(problem: Problem, theta: np.ndarray) -> tuple[float, float]:
    """Log density of ``z = (Re v, Im v)`` for the independent visibilities.

    With ``d = J z``, the real covariance is ``R = J⁻¹ C J⁻ᴴ``. Its noise part is
    ``diag(σ²/2)`` and its sky part is ``Re(B B^H)`` with ``B = J⁻¹ T Ψ^½``, which
    factorises as ``L L^T`` with ``L = [Re B, Im B]``. The returned pair is the
    ordinary real Gaussian log density of ``z`` and ``max|Im(B B^H)| / max|Re(B B^H)|``,
    which vanishes when the augmented covariance describes real-valued data.
    """
    orig, conj = conjugate_pairs(problem)
    m = orig.size
    J = np.zeros((problem.d.size, 2 * m), dtype=np.complex128)
    J[orig, np.arange(m)] = 1.0
    J[orig, m + np.arange(m)] = 1.0j
    J[conj, np.arange(m)] = 1.0
    J[conj, m + np.arange(m)] = -1.0j
    J_inv = np.linalg.inv(J)
    B = J_inv @ (problem.T * np.sqrt(problem.psi(theta)))
    sky = B @ B.conj().T
    imaginary_fraction = float(np.max(np.abs(sky.imag)) / np.max(np.abs(sky.real)))
    noise_var = np.real(np.diag(J_inv @ np.diag(1.0 / problem.ninv) @ J_inv.conj().T))
    z = np.concatenate([problem.d[orig].real, problem.d[orig].imag])
    logdet, quad = _low_rank_gaussian(np.sqrt(noise_var), np.hstack([B.real, B.imag]), z)
    log_density = -0.5 * quad - 0.5 * logdet - m * np.log(2.0 * np.pi)
    return float(log_density), imaginary_fraction


def comparison_scale(problem: Problem) -> float:
    """Magnitude of the terms that cancel in the target, ``d^H N⁻¹ d + N_d``.

    The target is a small difference of large terms, so it is compared with an
    absolute tolerance relative to this scale rather than to its own value.
    """
    return float(np.real(problem.d.conj() @ (problem.ninv * problem.d)) + problem.d.size)


def dense_map(problem: Problem, theta: np.ndarray) -> np.ndarray:
    """Posterior mean ``Ψ^½ A^H (I + A A^H)⁻¹ N^-½ d`` with ``A = N^-½ T Ψ^½``, via an SVD."""
    noise_std = 1.0 / np.sqrt(problem.ninv)
    root_psi = np.sqrt(problem.psi(theta))
    U, s, Vh = np.linalg.svd(problem.T * root_psi / noise_std[:, None], full_matrices=False)
    weights = s / (1.0 + s**2) * (U.conj().T @ (problem.d / noise_std))
    return root_psi * (Vh.conj().T @ weights)


@pytest.fixture(scope="session")
def dense():
    """Namespace of dense NumPy reference functions."""
    return SimpleNamespace(
        covariance=prior_predictive_covariance,
        log_likelihood=dense_log_likelihood,
        fixed_constant=fixed_constant,
        conjugate_pairs=conjugate_pairs,
        real_log_density=real_coordinate_log_density,
        map=dense_map,
        scale=comparison_scale,
    )
