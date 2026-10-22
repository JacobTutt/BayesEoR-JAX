"""Top-level exports for the BayesEoR-JAX workspace package."""

from importlib import import_module

from .model import (
    T_21,
    T_fg,
    T_fg_weighted,
    weight_foreground_operator,
    T_21_build_save,
    T_fg_build_save,
    T_fg_weighted_build_save,
    T_fg_svd_build_save,
    assemble_joint_operator,
    Beam_Model,
    BuildMatrices,
    FourierGeometry,
    ImageGeometry,
    build_fourier_geometry,
    build_foreground_spectral_operator,
    build_image_geometry,
    required_healpix_nside,
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
from .configuration import BayesEoRParser, ModelConfig, RunConfig
from .io import (
    load_numpy_dict,
    save_numpy_dict,
    write_log_files,
)
from .data import form_pI_vis, get_vis_data, jy_to_ksr, preprocess_uvdata, uvd_to_vector


_LAZY_EXPORTS = {
    "parse_uprior_inds": (".inference.prior", "parse_uprior_inds"),
    "MarginalisedLikelihood": (".inference", "MarginalisedLikelihood"),
    "PhiInverse": (".inference", "PhiInverse"),
    "Prior": (".inference", "Prior"),
    "build_kbin_index_per_voxel": (".inference", "build_kbin_index_per_voxel"),
    "compute_power_norm_per_bin": (".inference", "compute_power_norm_per_bin"),
    "theta_power_to_linear": (".inference", "theta_power_to_linear"),
    "MaximumAPosteriori": (".inference", "MaximumAPosteriori"),
    "run_multinest": (".inference", "run_multinest"),
    "run_nestedsampling": (".inference", "run_nestedsampling"),
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
    "T_21",
    "T_fg",
    "T_fg_weighted",
    "weight_foreground_operator",
    "T_21_build_save",
    "T_fg_build_save",
    "T_fg_weighted_build_save",
    "T_fg_svd_build_save",
    "assemble_joint_operator",
    "Beam_Model",
    "DataContainer",
    "BuildMatrices",
    "BayesEoRParser",
    "ModelConfig",
    "RunConfig",
    "Cosmology",
    "InstrumentModel",
    "MarginalisedLikelihood",
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
    "run_multinest",
    "run_nestedsampling",
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
    "required_healpix_nside",
    "build_nudft_matrix",
    "build_sky_to_visibility_operator",
    "build_spectral_fourier_operator",
    "build_T_21_core",
    "build_T_fg_core",
    "build_transverse_fourier_operator",
]
