"""Top-level exports for the BayesEoR-JAX workspace package."""

from importlib import import_module

from .build_matricies import (
    Beam_Model,
    BuildMatrices,
    FourierGeometry,
    ImageGeometry,
    build_fourier_geometry,
    build_foreground_spectral_operator,
    build_image_geometry,
    build_nudft_matrix,
    build_sky_to_visibility_operator,
    build_spectral_fourier_operator,
    build_T_21_core,
    build_T_fg_core,
    build_transverse_fourier_operator,
)
from .analyze import DataContainer
from .model import (
    Cosmology,
    InstrumentModel,
)
from .params import BayesEoRParser
from .utils import (
    load_numpy_dict,
    parse_uprior_inds,
    save_numpy_dict,
    write_log_files,
)
from .vis import form_pI_vis, get_vis_data, jy_to_ksr, preprocess_uvdata, uvd_to_vector


_LAZY_EXPORTS = {
    "FullLikelihood": (".inference", "FullLikelihood"),
    "MarginalisedLikelihood": (".inference", "MarginalisedLikelihood"),
    "PhiInverse": (".inference", "PhiInverse"),
    "Prior": (".inference", "Prior"),
    "build_kbin_index_per_voxel": (".inference", "build_kbin_index_per_voxel"),
    "compute_power_norm_per_bin": (".inference", "compute_power_norm_per_bin"),
    "theta_power_to_linear": (".inference", "theta_power_to_linear"),
    "MaximumAPosteriori": (".maxaposteriori", "MaximumAPosteriori"),
    "run_blackjaxns": (".run_ns", "run_blackjaxns"),
    "run_multinest": (".run_ns", "run_multinest"),
    "run_nestedsampling": (".run_ns", "run_nestedsampling"),
    "run_full_nuts": (".run_hmc", "run_full_nuts"),
    "run_marginalised_nuts": (".run_hmc", "run_marginalised_nuts"),
    "SetupResult": (".setup", "SetupResult"),
    "run_setup": (".setup", "run_setup"),
}


def __getattr__(name):
    """Load optional inference and sampler dependencies only when required."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "Beam_Model",
    "DataContainer",
    "BuildMatrices",
    "BayesEoRParser",
    "Cosmology",
    "InstrumentModel",
    "MarginalisedLikelihood",
    "FullLikelihood",
    "PhiInverse",
    "Prior",
    "build_kbin_index_per_voxel",
    "compute_power_norm_per_bin",
    "load_numpy_dict",
    "parse_uprior_inds",
    "preprocess_uvdata",
    "uvd_to_vector",
    "theta_power_to_linear",
    "SetupResult",
    "get_vis_data",
    "run_setup",
    "run_blackjaxns",
    "run_multinest",
    "run_nestedsampling",
    "run_full_nuts",
    "run_marginalised_nuts",
    "form_pI_vis",
    "jy_to_ksr",
    "save_numpy_dict",
    "write_log_files",
    "FourierGeometry",
    "ImageGeometry",
    "MaximumAPosteriori",
    "build_fourier_geometry",
    "build_foreground_spectral_operator",
    "build_image_geometry",
    "build_nudft_matrix",
    "build_sky_to_visibility_operator",
    "build_spectral_fourier_operator",
    "build_T_21_core",
    "build_T_fg_core",
    "build_transverse_fourier_operator",
]
