"""Class and functions for performing maximum a posteriori calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat
from typing import Any

import jax.numpy as jnp
import numpy as np
import jax.scipy as jsp
from astropy import units
from astropy.time import Time
from jax.experimental import sparse
from numpy.typing import ArrayLike

from ..inference.prior import build_phi_inverse
from ..params import BayesEoRParser
from ..setup import SetupResult, run_setup

LOGGER = logging.getLogger(__name__)


class MaximumAPosteriori:
    """
    Class for performing maximum a posteriori calculations.

    Parameters
    ----------
    config : str
        Path to a BayesEoR-JAX yaml configuration file.
    data_path : str, optional
        Path to either a pyuvdata-compatible file containing visibilities or
        a numpy-compatible file containing a preprocessed visibility vector.
        Supersedes the `data_path` in `config`. Defaults to None (use the
        `data_path` in `config`).
    array_dir : str, optional
        Path to a directory containing T, Ninv, and T_Ninv_T. Supersedes the
        `array_save_directory` created from the config. Defaults to None.
    """

    def __init__(
        self,
        *,
        config: Path | str,
        data_path: str | None = None,
        array_dir: str | None = None,
    ) -> None:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"{config} does not exist")

        parser = BayesEoRParser()
        args = parser.parse_args(["--config", str(config_path)])
        LOGGER.info("Parameters:\n%s", pformat(args.__dict__))

        if data_path is not None:
            data_path_obj = Path(data_path)
            if not data_path_obj.exists():
                raise FileNotFoundError(f"{data_path} does not exist")
            data_paths_differ = (
                data_path_obj.absolute().as_posix()
                != Path(args.data_path).absolute().as_posix()
            )
            if data_paths_differ:
                LOGGER.info("Replacing data_path: args.data_path=%s -> data_path=%s", args.data_path, data_path)
                args.data_path = str(data_path_obj)
            update_linear_terms = data_paths_differ
        else:
            update_linear_terms = False

        if array_dir is not None:
            array_dir_path = Path(array_dir)
            if not array_dir_path.exists():
                raise FileNotFoundError(f"{array_dir} does not exist")
        else:
            array_dir_path = None

        args_dict = dict(vars(args))
        setup_result: SetupResult = run_setup(**args_dict)

        if array_dir_path is not None:
            array_dirs_differ = (
                array_dir_path.absolute().as_posix()
                != Path(setup_result.bm.array_save_directory).absolute().as_posix()
            )
            if array_dirs_differ:
                LOGGER.info(
                    "Replacing array_save_directory: setup_result.bm.array_save_directory=%s -> array_dir_path=%s",
                    setup_result.bm.array_save_directory,
                    array_dir_path,
                )
                setup_result.bm.array_save_directory = array_dir_path
                update_linear_terms = True

        if update_linear_terms:
            LOGGER.info("Calculating dbar and d_Ninv_d.")
            vis_noisy = setup_result.vis_dict["vis_noisy"]

            t_matrix = setup_result.bm.load_matrix("T")
            ninv = setup_result.bm.load_matrix("Ninv")
            t_ninv_t = setup_result.bm.load_matrix("T_Ninv_T")

            ninv_d = sparse.bcoo_dot_general(
                ninv,
                vis_noisy,
                dimension_numbers=(((1,), (0,)), ((), ())),
            )
            dbar = sparse.bcoo_dot_general(
                t_matrix.conj().T,
                ninv_d,
                dimension_numbers=(((1,), (0,)), ((), ())),
            )
            d_ninv_d = jnp.real(jnp.vdot(vis_noisy, ninv_d))

            setup_result.likelihood_cfg.t_ninv_t = jnp.asarray(t_ninv_t)
            setup_result.likelihood_cfg.dbar = jnp.asarray(dbar)
            setup_result.likelihood_cfg.d_ninv_d = jnp.asarray(d_ninv_d)
            self.T = t_matrix
        else:
            self.T = setup_result.bm.load_matrix("T")

        self.args = args
        self.setup_result = setup_result
        self.likelihood = setup_result.likelihood_cfg
        self.prior = setup_result.prior_cfg
        self.phi_cfg = setup_result.phi_cfg
        self.bm = setup_result.bm
        self.k_vals = np.asarray(setup_result.k_vals)

        vis_dict = setup_result.vis_dict
        if "vis" in vis_dict:
            self.s = np.asarray(vis_dict["vis"])
        if "noise" in vis_dict:
            self.n = np.asarray(vis_dict["noise"])
        self.d = np.asarray(vis_dict["vis_noisy"])
        self.freqs = np.asarray(vis_dict["freqs_hz"]) * units.Hz
        self.jds = Time(np.asarray(vis_dict["jds"]), format="jd")
        self.uvws = np.asarray(vis_dict["uvws"])
        self.redundancy = np.asarray(vis_dict["redundancy"])
        antpairs = vis_dict.get("antpairs")
        if antpairs is not None:
            self.antpairs = antpairs

    def _calc_sigmai_dbar_wrapper(
        self,
        dmps: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Construct Sigma and solve for MAP coefficients for fixed ``dmps``.

        Returns
        -------
        map_coeffs : jax.Array
            ``Sigma^{-1} dbar``.
        dbar_sigmai_dbar : jax.Array
            Quadratic form ``dbar^H Sigma^{-1} dbar``.
        phi_i : jax.Array
            Diagonal prior precision ``Phi^{-1}``.
        log_det_sigma : jax.Array
            Log determinant of ``Sigma``.
        """
        phi_i = build_phi_inverse(jnp.asarray(dmps), self.phi_cfg)
        sigma = self.likelihood.t_ninv_t + jnp.diag(phi_i)
        chol = jnp.linalg.cholesky(sigma)
        y = jsp.linalg.solve_triangular(chol, self.likelihood.dbar, lower=True)
        map_coeffs = jsp.linalg.solve_triangular(chol.conj().T, y, lower=False)
        dbar_sigmai_dbar = jnp.vdot(self.likelihood.dbar, map_coeffs)
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))
        return map_coeffs, dbar_sigmai_dbar, phi_i, log_det_sigma

    def map_estimate(
        self,
        ps: float | ArrayLike | None = None,
        dmps: float | ArrayLike | None = None,
        return_prior_cov: bool = False,
    ):
        r"""
        Calculate the maximum a posteriori (MAP) model coefficients.

        Must pass one of `ps` or `dmps` which is required to calculate the
        prior covariance matrix.
        """
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None")
        dmps = self.calculate_dmps(ps=ps, dmps=dmps)

        map_coeffs, dbar_sigmai_dbar, prior_cov, log_det_sigma = (
            self._calc_sigmai_dbar_wrapper(jnp.asarray(dmps))
        )
        map_vis = sparse.bcoo_dot_general(
            self.T,
            map_coeffs,
            dimension_numbers=(((1,), (0,)), ((), ())),
        )

        log_det_prior_cov = -1 * jnp.sum(jnp.log(prior_cov)).real
        log_post = -0.5 * log_det_sigma - 0.5 * log_det_prior_cov + 0.5 * dbar_sigmai_dbar

        log_post += jnp.sum(jnp.log(jnp.asarray(dmps)[self.prior.uprior_inds]))
        log_post = log_post.real

        if not return_prior_cov:
            return np.asarray(map_coeffs), np.asarray(map_vis), float(log_post)
        return np.asarray(map_coeffs), np.asarray(map_vis), float(log_post), np.asarray(prior_cov)

    def calculate_prior_covariance(
        self,
        ps: float | ArrayLike | None = None,
        dmps: float | ArrayLike | None = None,
    ) -> np.ndarray:
        r"""
        Calculate the prior covariance matrix :math:`\Phi^{-1}`.
        """
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None")
        dmps = self.calculate_dmps(ps=ps, dmps=dmps)
        phi_i = build_phi_inverse(jnp.asarray(dmps), self.phi_cfg)
        return np.asarray(phi_i)

    def calculate_dmps(
        self,
        ps: float | ArrayLike | None = None,
        dmps: float | ArrayLike | None = None,
    ) -> np.ndarray:
        r"""
        Calculate the expected dimensionless power spectrum.
        """
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None")

        if ps is not None:
            if hasattr(ps, "__iter__"):
                ps = np.array(ps)
                if ps.size != self.k_vals.size:
                    raise ValueError(
                        f"'ps' has size {ps.size}, expected {self.k_vals.size}"
                    )
            else:
                ps = float(ps) * np.ones_like(self.k_vals)
            dmps = self.k_vals**3 / (2 * np.pi**2) * ps

        if dmps is not None:
            if hasattr(dmps, "__iter__"):
                dmps = np.array(dmps)
                if dmps.size != self.k_vals.size:
                    raise ValueError(
                        f"'dmps' has size {dmps.size}, expected {self.k_vals.size}"
                    )
            else:
                dmps = float(dmps) * np.ones_like(self.k_vals)

        return np.asarray(dmps)

    def extract_bl_vis(self, vis_vec: np.ndarray) -> dict[Any, np.ndarray]:
        """
        Form a dictionary of visibility waterfalls indexed by baseline.
        """
        nf = self.freqs.size
        nt = self.jds.size
        nbls = self.uvws.shape[1]

        expected_shape = (nbls * nf * nt,)
        if vis_vec.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {vis_vec.shape}"
            )

        bl_vis: dict[Any, np.ndarray] = {}
        for i_bl in range(nbls):
            key: Any
            if hasattr(self, "antpairs"):
                key = tuple(self.antpairs[i_bl])
            else:
                key = i_bl
            bl_vis[key] = vis_vec[i_bl::nbls].reshape(nt, nf)
        return bl_vis
