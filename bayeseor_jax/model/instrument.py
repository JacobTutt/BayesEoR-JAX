"""Instrument-model loading for the BayesEoR-JAX pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import jax
import jax.numpy as jnp

from ..utils import load_numpy_dict

jax.config.update("jax_enable_x64", True)


class InstrumentModel:
    """
    Load a BayesEoR instrument model.

    The instrument model consists of:
    - (u, v, w) array with shape `(nt, nbls, 3)` where `nt` is the number of
      times and `nbls` is the number of baselines
    - Number of redundantly-averaged baselines per (u, v, w) with shape
      `(nt, nbls)` or `(nt, nbls, 1)`
    - (optional) a list of antenna pair tuples with length `nbls` matching the
      order of baselines in the (u, v, w) array

    This class first looks for an `instrument_model.npy` pickled dictionary
    in `inst_model_dir`. If not found, it then loads the individual numpy
    arrays specified by `uvw_file`, `red_file`, and `antpairs_file`.

    Parameters
    ----------
    inst_model_dir : pathlib.Path or str
        Path to the instrument model directory.
    uvw_file : str, optional
        File containing instrumentally sampled (u, v, w) coords. Defaults to
        `'uvw_model.npy'`.
    red_file : str, optional
        File containing baseline redundancy info. Defaults to
        `'redundancy_model.npy'`.
    antpairs_file : str, optional
        File containing baseline antenna pairs for each sampled (u, v, w).
        Defaults to `'antpairs.npy'`.

    Attributes
    ----------
    uvw_array_m : jax.Array
        Sampled (u, v, w) coordinates in meters with shape `(nt, nbls, 3)`.
    bl_red_array : jax.Array
        Number of redundantly-averaged baselines per (u, v, w) with shape
        `(nt, nbls)`.
    antpairs : array-like or None
        Baseline antenna pair tuples if `antpairs_file` is found in
        `inst_model_dir`, otherwise `None`.
    """

    def __init__(
        self,
        inst_model_dir: Path | str,
        uvw_file: str = "uvw_model.npy",
        red_file: str = "redundancy_model.npy",
        antpairs_file: str = "antpairs.npy",
    ) -> None:
        
        # Validate and normalise `inst_model_dir` path.
        if not isinstance(inst_model_dir, Path):
            inst_model_dir = Path(inst_model_dir)
        # Validate `inst_model_dir` existence
        if not inst_model_dir.exists():
            raise FileNotFoundError(f"{inst_model_dir} does not exist")
        # Validate `inst_model_dir` is a directory (not a file)
        if not inst_model_dir.is_dir():
            raise ValueError(f"{inst_model_dir} is not a directory")

        self.inst_model_dir = inst_model_dir
        self.uvw_file = uvw_file
        self.red_file = red_file
        self.antpairs_file = antpairs_file

        # Default to looking in `inst_model_dir` for a single `instrument_model.npy` file containing 
        # all arrays as a dictionary.
        if (inst_model_dir / "instrument_model.npy").exists():
            # Load the single `instrument_model.npy` file containing all arrays as a dictionary.
            data_dict = load_numpy_dict(inst_model_dir / "instrument_model.npy")
            # Warn if file exists but does not contain a dictionary with expected keys.
            if not isinstance(data_dict, dict):
                raise ValueError("instrument_model.npy must contain a dictionary with "
                    "'uvw_model' and 'redundancy_model' entries")
            # Required instrument-model arrays.
            self.uvw_array_m = jnp.asarray(data_dict["uvw_model"], dtype=jnp.float64)
            self.bl_red_array = jnp.asarray(data_dict["redundancy_model"])
            # Optional baseline-label metadata.
            self.antpairs = data_dict.get("antpairs")
        # If the single `instrument_model.npy` file is not found, look for separate `uvw_file`
        # and `red_file` (and optionally `antpairs_file`) in `inst_model_dir`.
        else:
            uvw_path = inst_model_dir / uvw_file
            red_path = inst_model_dir / red_file
            # Raise error if neither the single `instrument_model.npy` file nor 
            # both the separate `uvw_file` and `red_file` are found in `inst_model_dir`.
            if not uvw_path.exists() or not red_path.exists():
                raise FileNotFoundError(
                    "inst_model_dir must contain either 'instrument_model.npy' "
                    f"or both '{uvw_file}' and '{red_file}'"
                )
            self.uvw_array_m = load_numpy_dict(uvw_path, as_jax=True)
            self.bl_red_array = load_numpy_dict(red_path, as_jax=True)
            antpairs_path = inst_model_dir / antpairs_file
            if antpairs_path.exists():
                self.antpairs = load_numpy_dict(antpairs_path, as_jax=False)
            else:
                self.antpairs = None

        if self.antpairs is not None:
            if hasattr(self.antpairs, "tolist"):
                self.antpairs = self.antpairs.tolist()

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        """Validate and normalize loaded instrument-model arrays."""
        if self.uvw_array_m.ndim != 3:
            raise ValueError(
                "uvw_array_m must have shape (n_t, n_bls, 3); "
                f"got shape {self.uvw_array_m.shape}"
            )
        if self.uvw_array_m.shape[-1] != 3:
            raise ValueError(
                "uvw_array_m must have shape (n_t, n_bls, 3); "
                f"got shape {self.uvw_array_m.shape}"
            )

        if self.bl_red_array.ndim == 3 and self.bl_red_array.shape[-1] == 1:
            self.bl_red_array = jnp.squeeze(self.bl_red_array, axis=-1)
        if self.bl_red_array.ndim != 2:
            raise ValueError(
                "bl_red_array must have shape (n_t, n_bls) or (n_t, n_bls, 1); "
                f"got shape {self.bl_red_array.shape}"
            )

        expected_shape = self.uvw_array_m.shape[:2]
        if self.bl_red_array.shape != expected_shape:
            raise ValueError(
                "bl_red_array shape must match uvw_array_m time/baseline axes; "
                f"expected {expected_shape}, got {self.bl_red_array.shape}"
            )

    @property
    def n_t(self) -> int:
        """Number of time integrations."""
        return int(self.uvw_array_m.shape[0])

    @property
    def n_bls(self) -> int:
        """Number of sampled baselines."""
        return int(self.uvw_array_m.shape[1])
