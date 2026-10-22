"""Physical models and their mappings into interferometric visibility space.

Instrument responses, observation geometry, and cosmological coordinates feed
the numerical operators. Construction helpers distribute and save those
operators and their noise-weighted products for inference.
"""

from .instrument.beam import Beam_Model
from .cosmology.fourier_geometry import FourierGeometry, build_fourier_geometry
from .observation.image_geometry import ImageGeometry, build_image_geometry
from .operators import (
    T_21,
    T_fg,
    T_fg_weighted,
    weight_foreground_operator,
    build_foreground_spectral_operator,
    build_nudft_matrix,
    build_sky_to_visibility_operator,
    build_spectral_fourier_operator,
    build_T_21_core,
    build_T_fg_core,
    build_transverse_fourier_operator,
)
from .construction.model_builder import BuildMatrices
from .construction.operator_construction import (
    T_21_build_save, T_fg_build_save, T_fg_weighted_build_save, assemble_joint_operator,
)
from .construction.kl_reduction import T_fg_svd_build_save
from .cosmology.transforms import Cosmology
from .instrument.baselines import InstrumentModel
from .instrument.noise import generate_gaussian_noise

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
    "BuildMatrices",
    "Cosmology",
    "FourierGeometry",
    "ImageGeometry",
    "InstrumentModel",
    "build_fourier_geometry",
    "build_image_geometry",
    "build_foreground_spectral_operator",
    "build_nudft_matrix",
    "build_sky_to_visibility_operator",
    "build_spectral_fourier_operator",
    "build_T_21_core",
    "build_T_fg_core",
    "build_transverse_fourier_operator",
    "generate_gaussian_noise",
]
