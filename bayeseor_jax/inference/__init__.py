"""Inference components (priors and likelihoods) for BayesEoR-JAX."""

from importlib import import_module

from .phi import (
    PhiInverse,
    build_kbin_index_per_voxel,
    compute_power_norm_per_bin,
)
from .prior import Prior, theta_power_to_linear


_LAZY_EXPORTS = {
    "MarginalisedLikelihood": (".likelihood_marg", "MarginalisedLikelihood"),
    "FullLikelihood": (".likelihood_full", "FullLikelihood"),
}


def __getattr__(name):
    """Load optional solver-backed likelihoods only when requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "MarginalisedLikelihood",
    "FullLikelihood",
    "PhiInverse",
    "Prior",
    "build_kbin_index_per_voxel",
    "compute_power_norm_per_bin",
    "theta_power_to_linear",
]
