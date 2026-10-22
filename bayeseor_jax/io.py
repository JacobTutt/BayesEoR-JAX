"""Shared array I/O and run metadata for data preparation and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding


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
