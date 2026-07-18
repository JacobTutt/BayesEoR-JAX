"""Shared utility helpers for BayesEoR-JAX."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding

from .build_matricies.beam_model import Beam_Model

jax.config.update("jax_enable_x64", True)
LOGGER = logging.getLogger(__name__)


def block_until_ready(value: Any) -> Any:
    """Synchronize a JAX-backed value and return the realized object."""
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def mpiprint(*args: Any, rank: int = 0, **kwargs: Any) -> None:
    """
    Lightweight rank-gated print helper.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to logging.
    rank : int, optional
        Rank identifier. Output is emitted only when ``rank == 0``.
    **kwargs : Any
        Unused. Accepted for compatibility with legacy callsites.
    """
    if rank == 0:
        message = " ".join(str(arg) for arg in args)
        LOGGER.info(message)


def parse_uprior_inds(upriors_str: str, nkbins: int) -> jnp.ndarray:
    """
    Parse a uniform-prior bin selection string.

    Parameters
    ----------
    upriors_str : str
        Selection string. Supported forms:
        - ``"all"``
        - ``""`` (empty)
        - slice form, e.g. ``"1:4"``
        - comma list, e.g. ``"1,4,6"``
        - single index, e.g. ``"3"`` or ``"-1"``
    nkbins : int
        Total number of k-bins.

    Returns
    -------
    jax.numpy.ndarray
        Boolean array of length ``nkbins`` where True indicates uniform prior.
    """
    if upriors_str.lower() == "all":
        return jnp.ones(nkbins, dtype=bool)

    uprior_inds = jnp.zeros(nkbins, dtype=bool)
    if upriors_str == "":
        return uprior_inds

    if ":" in upriors_str:
        bounds = list(map(int, upriors_str.split(":")))
        uprior_inds = uprior_inds.at[slice(*bounds)].set(True)
    elif "," in upriors_str:
        up_inds = list(map(int, upriors_str.split(",")))
        uprior_inds = uprior_inds.at[up_inds].set(True)
    else:
        uprior_inds = uprior_inds.at[int(upriors_str)].set(True)

    return uprior_inds


def _resolve_uv_spacing(
    *,
    fov_deg: float,
    du: float | None,
    dv: float | None,
    label: str,
    rtol: float = 5e-2,
) -> tuple[float, float]:
    """Resolve uv-grid spacings from FoV and warn on explicit mismatches."""
    fov_deg = float(fov_deg)
    if fov_deg <= 0:
        raise ValueError(f"{label}: fov must be > 0 deg, got {fov_deg}")

    derived = float(1.0 / np.deg2rad(fov_deg))
    du_val = derived if du is None else float(du)
    dv_val = derived if dv is None else float(dv)

    if not np.isclose(du_val, derived, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: du={du_val:.6g} is inconsistent with FoV-derived value "
            f"{derived:.6g} (=1/deg2rad({fov_deg:.6g})).",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.isclose(dv_val, derived, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: dv={dv_val:.6g} is inconsistent with FoV-derived value "
            f"{derived:.6g} (=1/deg2rad({fov_deg:.6g})).",
            RuntimeWarning,
            stacklevel=2,
        )

    return du_val, dv_val


def _resolve_uv_spacing_from_image_extents(
    *,
    extent_u: float,
    extent_v: float,
    du: float | None,
    dv: float | None,
    label: str,
    rtol: float = 5e-2,
) -> tuple[float, float]:
    """
    Resolve anisotropic uv-grid spacings from local angular chart widths.

    The input extents are expected to be angular widths in radians along the
    two transverse axes of the fixed image-domain chart. In the current
    wide-field setup these are built from the full ``(l, m, n)`` vectors using
    ``atan2(l, n)`` and ``atan2(m, n)`` so that the widths remain true angular
    offsets even when the raw direction cosines become strongly non-linear.
    """
    extent_u = float(extent_u)
    extent_v = float(extent_v)
    if extent_u <= 0 or extent_v <= 0:
        raise ValueError(
            f"{label}: extent_u and extent_v must both be > 0, got "
            f"{extent_u} and {extent_v}"
        )

    derived_du = float(1.0 / extent_u)
    derived_dv = float(1.0 / extent_v)
    du_val = derived_du if du is None else float(du)
    dv_val = derived_dv if dv is None else float(dv)

    if not np.isclose(du_val, derived_du, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: du={du_val:.6g} is inconsistent with the local-image "
            f"extent-derived value {derived_du:.6g} (=1/{extent_u:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.isclose(dv_val, derived_dv, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: dv={dv_val:.6g} is inconsistent with the local-image "
            f"extent-derived value {derived_dv:.6g} (=1/{extent_v:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return du_val, dv_val


def _resolve_model_grid_size(
    *,
    value: int | None,
    required: int,
    label: str,
) -> int:
    """Resolve one uv-axis model size, warning when explicit overrides differ."""
    if required < 1:
        required = 1

    if value is None:
        return int(required)

    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{label} must be >= 1, got {resolved}")
    if resolved < required:
        warnings.warn(
            f"{label}={resolved} is smaller than the auto-resolved requirement "
            f"{required} for the selected data, FoV, and beam model.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif resolved > required:
        warnings.warn(
            f"{label}={resolved} is larger than the auto-resolved requirement "
            f"{required}; the run will use the explicit larger value.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved


def _resolve_uv_grid_from_extents(
    *,
    extent_u_eor: float,
    extent_v_eor: float,
    extent_u_fg: float,
    extent_v_fg: float,
    freqs_hz: np.ndarray,
    uvws_m: np.ndarray,
    beam_type: str | None,
    fwhm_deg: float | None,
    diam: float | None,
    n_u_eor: int | None,
    n_v_eor: int | None,
    du_eor: float | None,
    dv_eor: float | None,
    n_u_fg: int | None,
    n_v_fg: int | None,
    du_fg: float | None,
    dv_fg: float | None,
) -> dict[str, float | int]:
    """
    Resolve anisotropic uv-grid spacings and required axis sizes.

    The fixed central-time angular chart widths set the uv sampling through
    ``du = 1 / extent_u`` and ``dv = 1 / extent_v``. The grid extents are then
    chosen to cover the instrument support plus half the beam-aperture width on
    each side, following the old elongated-FoV validation logic.
    """
    du_eor_val, dv_eor_val = _resolve_uv_spacing_from_image_extents(
        extent_u=extent_u_eor,
        extent_v=extent_v_eor,
        du=du_eor,
        dv=dv_eor,
        label="EoR uv grid spacing",
    )
    du_fg_val, dv_fg_val = _resolve_uv_spacing_from_image_extents(
        extent_u=extent_u_fg,
        extent_v=extent_v_fg,
        du=du_fg,
        dv=dv_fg,
        label="FG uv grid spacing",
    )

    uvws_m = np.asarray(uvws_m, dtype=float).reshape(-1, 3)
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if uvws_m.size == 0:
        raise ValueError("uvws_m must contain at least one baseline sample")

    wavelength_min = float(299792458.0 / freqs_hz.max())
    uvs = uvws_m[:, :2] / wavelength_min
    u_max_inst_model = float(np.abs(uvs[:, 0]).max())
    v_max_inst_model = float(np.abs(uvs[:, 1]).max())

    beam = Beam_Model(
        beam_type=beam_type,
        fwhm_deg=fwhm_deg,
        diam=diam,
    )
    aperture_width = beam.aperture_width_uv(freqs_hz=freqs_hz)
    u_extent = u_max_inst_model + 0.5 * aperture_width
    v_extent = v_max_inst_model + 0.5 * aperture_width

    n_u_eor_required = int(np.ceil(u_extent / du_eor_val)) * 2 + 1
    n_v_eor_required = int(np.ceil(v_extent / dv_eor_val)) * 2 + 1
    n_u_fg_required = int(np.ceil(u_extent / du_fg_val)) * 2 + 1
    n_v_fg_required = int(np.ceil(v_extent / dv_fg_val)) * 2 + 1

    return {
        "du_eor": float(du_eor_val),
        "dv_eor": float(dv_eor_val),
        "du_fg": float(du_fg_val),
        "dv_fg": float(dv_fg_val),
        "n_u_eor_required": int(n_u_eor_required),
        "n_v_eor_required": int(n_v_eor_required),
        "n_u_fg_required": int(n_u_fg_required),
        "n_v_fg_required": int(n_v_fg_required),
        "n_u_eor": _resolve_model_grid_size(
            value=n_u_eor,
            required=n_u_eor_required,
            label="n_u_eor",
        ),
        "n_v_eor": _resolve_model_grid_size(
            value=n_v_eor,
            required=n_v_eor_required,
            label="n_v_eor",
        ),
        "n_u_fg": _resolve_model_grid_size(
            value=n_u_fg,
            required=n_u_fg_required,
            label="n_u_fg",
        ),
        "n_v_fg": _resolve_model_grid_size(
            value=n_v_fg,
            required=n_v_fg_required,
            label="n_v_fg",
        ),
        "u_max_inst_model": float(u_max_inst_model),
        "v_max_inst_model": float(v_max_inst_model),
        "beam_aperture_width_uv": float(aperture_width),
        "u_extent_with_buffer": float(u_extent),
        "v_extent_with_buffer": float(v_extent),
    }


def _coerce_args_dict(args: Any) -> dict[str, Any]:
    """Convert parser output or mapping-like object to a JSON-serializable dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "as_dict"):
        return dict(args.as_dict())
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    raise TypeError("args must be a mapping or namespace-like object")


def write_log_files(
    parser: Any,
    args: Any,
    out_dir: Path | str = Path("./"),
    version: str | None = None,
) -> None:
    """
    Write run metadata files for reproducibility.

    Parameters
    ----------
    parser : Any
        Parser instance. If it has a ``save`` method (e.g. jsonargparse), it
        will be used to serialize arguments.
    args : Any
        Parsed argument namespace or dictionary.
    out_dir : pathlib.Path or str, optional
        Output directory for log files.
    version : str or None, optional
        Optional version string to write to ``version.txt``.
    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if version is not None:
        ver_file = out_dir / "version.txt"
        if not ver_file.exists():
            ver_file.write_text(f"{version}\n")

    args_file = out_dir / "args.json"
    if not args_file.exists():
        if hasattr(parser, "save"):
            parser.save(args, args_file, format="json", skip_none=False)
        else:
            args_dict = _coerce_args_dict(args)
            args_file.write_text(json.dumps(args_dict, indent=2, default=str) + "\n")


def write_analysis_metadata(
    *,
    output_dir: Path | str,
    args: Any,
    k_vals: Any,
    mod_k_vo: Any,
    bin_id_per_k_mode: Any,
    valid_mask: Any,
    clobber: bool = False,
    version: str | None = None,
) -> None:
    """
    Write old-style analysis metadata files into one output directory.

    This consolidates the metadata layout expected by the legacy BayesEoR
    analysis tooling:
    - `args.json`
    - `version.txt` (optional)
    - `k-vals.txt`
    - `k-vals-bins.txt`
    - `k-vals-nsamples.txt`
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    write_log_files(
        parser=None,
        args=args,
        out_dir=output_dir,
        version=version,
    )

    k_vals_path = output_dir / "k-vals.txt"
    k_bins_path = output_dir / "k-vals-bins.txt"
    k_counts_path = output_dir / "k-vals-nsamples.txt"
    if not clobber and all(p.exists() for p in (k_vals_path, k_bins_path, k_counts_path)):
        return

    k_vals_np = np.asarray(k_vals, dtype=np.float64).reshape(-1)
    mod_k_np = np.asarray(mod_k_vo, dtype=np.float64).reshape(-1)
    bid_np = np.asarray(bin_id_per_k_mode, dtype=np.int32).reshape(-1)
    valid_np = np.asarray(valid_mask, dtype=bool).reshape(-1)

    if k_vals_np.size == 0:
        raise ValueError("Cannot save k-bin metadata with zero non-empty bins.")

    kbin_edges: list[float] = []
    nsamples: list[int] = []
    last_max = None
    for i_bin in range(k_vals_np.size):
        in_bin = valid_np & (bid_np == i_bin)
        bin_vals = mod_k_np[in_bin]
        if bin_vals.size == 0:
            raise ValueError(f"Compact k-bin {i_bin} has no assigned modes.")
        kbin_edges.append(float(bin_vals.min()))
        nsamples.append(int(bin_vals.size))
        last_max = float(bin_vals.max())
    kbin_edges.append(last_max)

    np.savetxt(k_vals_path, k_vals_np)
    np.savetxt(k_bins_path, np.asarray(kbin_edges, dtype=np.float64))
    np.savetxt(k_counts_path, np.asarray(nsamples, dtype=np.int64), fmt="%d")


def estimate_average_eval_time(
    likelihood_fn,
    theta_ref: np.ndarray,
    avg_iters: int,
) -> float:
    """
    Estimate the average scalar likelihood evaluation time.

    Parameters
    ----------
    likelihood_fn : callable
        Scalar log-likelihood callback.
    theta_ref : numpy.ndarray
        Reference parameter vector used for repeated timing calls.
    avg_iters : int
        Number of repeated evaluations.
    """
    start = time.perf_counter()
    for _ in range(int(avg_iters)):
        value = likelihood_fn(theta_ref)
        value = block_until_ready(value)
        if not np.isfinite(float(np.asarray(value))):
            raise ValueError("Non-finite likelihood value encountered during timing.")
    return (time.perf_counter() - start) / max(1, int(avg_iters))


def save_numpy_dict(
    fp: Path | str,
    arr: Any,
    args: dict[str, Any] | Any,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
    clobber: bool = False,
) -> None:
    """
    Save array data with metadata as a NumPy dictionary payload.

    Parameters
    ----------
    fp : pathlib.Path or str
        Output file path.
    arr : Any
        Data payload to persist under ``data``.
    args : dict or namespace-like
        Associated metadata persisted under ``args``.
    version : str or None, optional
        Optional version tag persisted under ``version``.
    extra : dict or None, optional
        Optional extra metadata persisted under ``extra``.
    clobber : bool, optional
        Overwrite output file if it already exists.
    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if fp.exists() and not clobber:
        raise ValueError(f"clobber is false but file already exists: {fp}")
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra must be a dictionary")

    fp.parent.mkdir(exist_ok=True, parents=True)
    out_dict: dict[str, Any] = {
        "data": arr,
        "args": _coerce_args_dict(args),
    }
    if version is not None:
        out_dict["version"] = version
    if extra is not None:
        out_dict["extra"] = extra
    np.save(fp, out_dict)


def load_numpy_dict(fp: Path | str, as_jax: bool = False) -> Any:
    """
    Load data array from disk saved either as a plain NumPy array or a legacy
    BayesEoR dictionary payload containing a ``data`` entry.

    Parameters
    ----------
    fp : pathlib.Path or str
        File path for the saved array or dictionary payload.
    as_jax : bool, optional
        If True, convert numeric array-like payloads to JAX arrays before
        returning. Defaults to False.

    Returns
    -------
    data : Any
        The stored payload. If the file contains a dictionary with a ``data``
        key, that value is returned.
    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"{fp} does not exist")

    loaded = np.load(fp, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        item = loaded.item()
        if isinstance(item, dict):
            loaded = item.get("data", item)
        else:
            loaded = item

    if as_jax:
        try:
            return jnp.asarray(loaded)
        except (TypeError, ValueError):
            return loaded
    return loaded


def load_sharded_npy(
    fp: Path | str,
    sharding: NamedSharding,
    *,
    dtype: np.dtype | None = None,
) -> jax.Array:
    """
    Load a dense ``.npy`` array directly into a sharded JAX array.

    Parameters
    ----------
    fp : pathlib.Path or str
        Path to the ``.npy`` file.
    sharding : jax.sharding.NamedSharding
        Target sharding for the returned global array.
    dtype : numpy.dtype or None, optional
        Optional dtype override applied shard-by-shard while loading.

    Returns
    -------
    jax.Array
        Global array whose addressable shards are read lazily from the file.
    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"{fp} does not exist")

    mm = np.load(fp, mmap_mode="r")
    out_dtype = np.dtype(mm.dtype if dtype is None else dtype)

    def _callback(index):
        return np.asarray(mm[index], dtype=out_dtype)

    return jax.make_array_from_callback(mm.shape, sharding, _callback)


class MultiNestPathManager:
    """Hide long output directories from MultiNest behind a short symlink.

    MultiNest output filenames must not exceed 100 characters. Its longest
    filename in this pipeline is ``<file_root>post_equal_weights.dat``. If the
    requested path would exceed that limit, this class passes MultiNest a
    short path of the form ``.mn_chains_symlinks/mn_<hash>/<file_root>``.
    The symlink targets the requested output directory, so files are written
    directly to their final location without copying or staging.

    The hash is derived from the resolved output directory. Repeating a run,
    including a resumed run, therefore recreates the same link. The relative
    link directory must be on a filesystem shared by all JAX processes and
    the working directory must remain unchanged while MultiNest is running.

    Parameters
    ----------
    output_dir : pathlib.Path or str
        Real directory in which MultiNest products should remain.
    file_root : str, optional
        MultiNest filename prefix. It must be a filename rather than a path.

    Notes
    -----
    ``prepare`` must be called before using ``outputfiles_basename`` and
    ``cleanup`` after MultiNest has finished reading its products. Cleanup
    removes only a verified symlink and never removes the real output data.
    """

    def __init__(
        self,
        output_dir: Path | str,
        file_root: str = "data-",
    ) -> None:
        if not file_root:
            raise ValueError("MultiNest file_root must not be empty.")
        if Path(file_root).name != file_root:
            raise ValueError(
                "MultiNest file_root must be a filename prefix, not a path: "
                f"{file_root!r}"
            )

        self.output_dir = Path(output_dir)
        self.real_output_dir = self.output_dir.expanduser().resolve()
        self.file_root = file_root
        self.logical_outputfiles_basename = (
            self.real_output_dir / self.file_root
        ).as_posix()

        direct_basename = (self.output_dir / self.file_root).as_posix()
        self.using_short_path = (
            len(f"{direct_basename}post_equal_weights.dat") > 100
        )

        digest = hashlib.sha256(os.fsencode(self.real_output_dir)).hexdigest()[:16]
        self.short_link = Path(".mn_chains_symlinks") / f"mn_{digest}"
        basename_directory = (
            self.short_link if self.using_short_path else self.output_dir
        )
        self.outputfiles_basename = (
            basename_directory / self.file_root
        ).as_posix()
        self._prepared = False

    def _resolved_link_target(self) -> Path:
        """Resolve the current symlink target for safety comparisons."""
        target = Path(os.readlink(self.short_link))
        if not target.is_absolute():
            target = self.short_link.parent / target
        return target.resolve()

    def prepare(self) -> str:
        """Prepare the output path and return the basename for MultiNest.

        Short paths are returned unchanged. For long paths, the method creates
        or reuses the deterministic symlink. A stale symlink is repaired, but
        a non-symlink collision is never deleted. The final length check is a
        guard against future changes to the fixed short-link layout.

        Returns
        -------
        str
            The ``outputfiles_basename`` to pass to ``pymultinest.solve``.
        """
        self.real_output_dir.mkdir(parents=True, exist_ok=True)
        if not self.using_short_path:
            self._prepared = True
            return self.outputfiles_basename

        self.short_link.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(self.short_link):
            if not self.short_link.is_symlink():
                raise FileExistsError(
                    "Refusing to replace non-symlink MultiNest short path: "
                    f"{self.short_link}"
                )
            if self._resolved_link_target() != self.real_output_dir:
                self.short_link.unlink()

        if not os.path.lexists(self.short_link):
            self.short_link.symlink_to(
                self.real_output_dir,
                target_is_directory=True,
            )

        longest_path = f"{self.outputfiles_basename}post_equal_weights.dat"
        if len(longest_path) > 100:
            raise ValueError(
                "The short MultiNest path still exceeds 100 characters: "
                f"{longest_path}"
            )

        self._prepared = True
        return self.outputfiles_basename

    def cleanup(self) -> None:
        """Remove the temporary link without touching the output directory.

        The link is removed only when this object prepared a shortened path,
        the path is still a symlink, and it still targets the expected real
        output directory. Any unexpected filesystem object is left intact and
        reported as an error.
        """
        if not self._prepared or not self.using_short_path:
            return
        if not os.path.lexists(self.short_link):
            self._prepared = False
            return
        if not self.short_link.is_symlink():
            raise RuntimeError(
                "Refusing to remove non-symlink MultiNest short path: "
                f"{self.short_link}"
            )
        if self._resolved_link_target() != self.real_output_dir:
            raise RuntimeError(
                "Refusing to remove a MultiNest short link whose target changed: "
                f"{self.short_link}"
            )
        self.short_link.unlink()
        self._prepared = False


def _prepare_multinest_path(path_manager: MultiNestPathManager) -> str:
    """Prepare a MultiNest path safely across JAX processes.

    In a normal single-process run this simply calls ``path_manager.prepare``.
    In a distributed run, process zero performs the filesystem mutation and
    broadcasts whether it succeeded before any process raises an exception.
    This ordering prevents other processes hanging at a barrier when link
    creation fails. All processes then wait until the shared link is visible
    and use the same deterministic basename.
    """
    distributed = bool(
        getattr(jax, "distributed", None) is not None
        and jax.distributed.is_initialized()
    )
    if not distributed:
        return path_manager.prepare()

    process_index = int(jax.process_index())
    source_error: Exception | None = None
    if process_index == 0:
        try:
            path_manager.prepare()
        except Exception as exc:  # broadcast failure before any rank raises
            source_error = exc

    source_ok = np.asarray(1 if source_error is None else 0, dtype=np.int32)
    source_ok = multihost_utils.broadcast_one_to_all(
        source_ok,
        is_source=process_index == 0,
    )
    if not bool(np.asarray(source_ok).item()):
        if source_error is not None:
            raise source_error
        raise RuntimeError(
            "JAX process 0 could not prepare the shared MultiNest short path."
        )

    multihost_utils.sync_global_devices("multinest-short-path-ready")
    return path_manager.outputfiles_basename


def _cleanup_multinest_path(path_manager: MultiNestPathManager) -> None:
    """Remove a short link after every sampler process has finished.

    Direct output paths require no work. In distributed runs, all processes
    first confirm that MultiNest has returned; process zero then removes the
    verified symlink and broadcasts the cleanup result. Cleanup failure emits
    a warning rather than invalidating an otherwise completed inference run.
    """
    if not path_manager.using_short_path:
        return

    distributed = bool(
        getattr(jax, "distributed", None) is not None
        and jax.distributed.is_initialized()
    )
    if not distributed:
        path_manager.cleanup()
        return

    multihost_utils.sync_global_devices("multinest-short-path-finished")
    process_index = int(jax.process_index())
    source_error: Exception | None = None
    if process_index == 0:
        try:
            path_manager.cleanup()
        except Exception as exc:  # cleanup must not invalidate completed samples
            source_error = exc

    source_ok = np.asarray(1 if source_error is None else 0, dtype=np.int32)
    source_ok = multihost_utils.broadcast_one_to_all(
        source_ok,
        is_source=process_index == 0,
    )
    if not bool(np.asarray(source_ok).item()):
        warnings.warn(
            "MultiNest completed, but process 0 could not remove its short "
            f"symlink: {source_error or path_manager.short_link}",
            RuntimeWarning,
            stacklevel=2,
        )
    multihost_utils.sync_global_devices("multinest-short-path-cleaned")
