"""Shared fixtures and settings for the BayesEoR-JAX test suite."""

from __future__ import annotations

import os

# Force CPU JAX and a non-interactive Matplotlib backend before any import of them.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import importlib.util
import io
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from bayeseor_jax.configuration import RunConfig  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: HERA site, as [latitude (deg), longitude (deg), altitude (m)].
HERA_LATLONALT = (-30.72152777777791, 21.428305555555557, 1073.0000000093132)


# ---------------------------------------------------------------------------
# Markers that skip when an optional dependency is unavailable
# ---------------------------------------------------------------------------

def _multinest_available() -> bool:
    """Return whether PyMultiNest and its native library can be loaded."""
    if importlib.util.find_spec("pymultinest") is None:
        return False
    try:
        # PyMultiNest prints a banner when its native library is missing; keep it quiet.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from pymultinest.solve import solve  # noqa: F401
    except (ImportError, OSError, SystemExit):
        return False
    return True


def _n_gpus() -> int:
    """Return the number of visible GPU devices, without initialising CPU-only JAX."""
    try:
        return len(jax.devices("gpu"))
    except RuntimeError:
        return 0


def pytest_collection_modifyitems(config, items):
    """Skip tests whose optional requirements are not met in this environment."""
    skip_multinest = pytest.mark.skip(reason="PyMultiNest or the MultiNest library is unavailable")
    skip_gpu = pytest.mark.skip(reason="requires at least two GPUs")
    has_multinest = _multinest_available()
    has_gpus = _n_gpus() >= 2
    for item in items:
        if "multinest" in item.keywords and not has_multinest:
            item.add_marker(skip_multinest)
        if "gpu" in item.keywords and not has_gpus:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Committed fixture files
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding the committed test fixtures."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def tiny_uvh5() -> Path:
    """Four-antenna, four-channel, four-integration drift scan in Jy, with autos."""
    return FIXTURES_DIR / "tiny.uvh5"


@pytest.fixture(scope="session")
def tiny_npy_dir() -> Path:
    """Preprocessed ``vis_vector.npy`` and instrument model for ``tiny.uvh5``."""
    return FIXTURES_DIR / "tiny_npy"


@pytest.fixture(scope="session")
def airy_beamfits() -> Path:
    """Coarse az/za Airy electric-field beam covering 149.5–151 MHz."""
    return FIXTURES_DIR / "airy_efield.beamfits"


@pytest.fixture(scope="session")
def hera_latlonalt() -> tuple[float, float, float]:
    """Geodetic location of the HERA site."""
    return HERA_LATLONALT


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded NumPy random generator, fresh for each test."""
    return np.random.default_rng(1234)


@pytest.fixture
def tiny_config(tmp_path, tiny_npy_dir):
    """Factory for a small ``RunConfig`` on the ``.npy`` fixture, writing to ``tmp_path``.

    Keyword arguments override the defaults, e.g. ``tiny_config(model="fg")``.
    """

    def make(**overrides) -> RunConfig:
        model = overrides.get("model", "eor+fg")
        base = RunConfig(
            model=model,
            data_path=tiny_npy_dir / "vis_vector.npy",
            inst_model=tiny_npy_dir,
            n_freq=4,
            f0_mhz=150.0,
            df_mhz=0.1,
            n_t=4,
            d_t=10.0,
            jd_min=2459999.1,
            telescope_latlonalt=HERA_LATLONALT,
            beam_type="airy",
            diam=14.0,
            fov_eor=20.0,
            fov_fg=20.0,
            nside=16,
            sigma=1.0,
            priors=None if model == "fg" else [[-2.0, 4.0], [-2.0, 4.0]],
            array_save_directory=tmp_path / "matrices",
            output_dir=tmp_path / "output",
            save_dir=tmp_path / "saved",
        )
        return replace(base, **overrides)

    return make
