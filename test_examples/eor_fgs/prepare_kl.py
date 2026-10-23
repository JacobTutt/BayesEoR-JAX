"""Build the joint laptop example's foreground KL basis on CPU before inference."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.utils import iers

iers.conf.auto_download = False

from bayeseor_jax import BayesEoRParser
from bayeseor_jax.model.construction.model_builder import BuildMatrices
from bayeseor_jax.model.construction.kl_reduction import T_fg_svd_build_save
from bayeseor_jax.model.instrument.noise import build_noise_variance
from bayeseor_jax.model.observation.image_geometry import build_image_geometry


def prepare_kl():
    """Construct, weight and reduce the configured 10-degree foreground operator.

    Returns
    -------
    pathlib.Path
        Saved unwhitened operator consumed by the joint example configuration.
        Weights use the same physical foreground prior and noise as inference.
    """
    config = BayesEoRParser().parse_args(["--config", "test_examples/eor_fgs/config.yaml"])
    output = Path(config.foreground_kl_path)
    config = replace(config, model="fg", array_save_directory=output.parent)
    uvw = np.load(Path(config.inst_model) / "uvw_model.npy")
    redundancy = np.load(Path(config.inst_model) / "redundancy_model.npy")
    frequencies = (config.f0_mhz + config.df_mhz*np.arange(config.n_freq))*1e6
    geometry = build_image_geometry(
        fov_eor=config.fov_eor, fov_fg=config.fov_fg, nside=config.nside,
        telescope_latlonalt=config.telescope_latlonalt, jd_center=config.jd_center,
        n_t=config.n_t, d_t=config.d_t,
    )
    builder = BuildMatrices(config, frequencies_hz=frequencies,
                            uvw_array_m=uvw, image_geometry=geometry)
    variance = build_noise_variance(config.n_t, config.n_freq, uvw.shape[1],
                                    sigma=config.sigma, redundancy=redundancy)
    prior_std = np.repeat(config.foreground_prior_std_mk, geometry.n_pix_fg)
    weighted = builder.T_fg_weighted_build_save(
        prior_sqrt=prior_std, noise_whitener=1/np.sqrt(variance),
    )
    inverse_noise = output.parent / "Ninv.npy"
    np.save(inverse_noise, 1/variance)
    result = T_fg_svd_build_save(weighted, inverse_noise, output_path=output, f=0.1)
    print(f"Foreground coefficients: {prior_std.size} -> {np.load(result).shape[1]}")
    return result


if __name__ == "__main__":
    prepare_kl()
