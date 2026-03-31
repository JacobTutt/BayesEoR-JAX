"""Shared utility helpers for BayesEoR-JAX."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding

jax.config.update("jax_enable_x64", True)
LOGGER = logging.getLogger(__name__)


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


def validate_multinest_output_path(
    out_dir: Path,
    file_root: str,
    longest_suffix: str = "post_equal_weights.dat",
) -> str:
    """
    Validate a MultiNest output prefix against the 100-character path limit.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory that will contain the MultiNest output files.
    file_root : str
        Filename prefix passed to MultiNest, for example ``"data-"``.
    longest_suffix : str, optional
        Longest expected suffix emitted by MultiNest. Defaults to
        ``"post_equal_weights.dat"``.

    Returns
    -------
    str
        The full ``outputfiles_basename`` string to pass to MultiNest.
    """
    outputfiles_basename = (out_dir / file_root).as_posix()
    longest_path = f"{outputfiles_basename}{longest_suffix}"
    if len(longest_path) > 100:
        raise ValueError(
            "MultiNest requires output file paths of length <= 100 characters. "
            f"The derived path '{longest_path}' has length {len(longest_path)}."
        )
    return outputfiles_basename


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
        if hasattr(value, "block_until_ready"):
            value = value.block_until_ready()
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
