"""Decompose, truncate and save a reduced KL foreground operator using distributed SVD.

The input already contains the physical foreground prior. Only the retained
left singular modes are saved, with noise whitening undone so the resulting
operator maps standard-normal coefficients into physical visibility units.
"""

import json
import logging
import math
import operator
from pathlib import Path
import time

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .storage import (
    _check_output_path, _prepare_memmap_output, _slice_bounds,
    publish_output, save_array,
)

LOGGER = logging.getLogger(__name__)


def select_foreground_rank(singular_values, n_vis, f=0.1):
    r"""Find the smallest rank meeting the discarded prior-predictive RMS limit.

    Parameters
    ----------
    singular_values : array-like
        Non-negative singular values of A = N^-1/2 T_fg Sigma_q^1/2, ordered
        from largest to smallest, with modes introduced by padding excluded.
    n_vis : int
        Number of measured complex visibilities, before solver padding.
    f : float, optional
        Maximum discarded RMS relative to thermal-noise RMS; defaults to 0.1.
        Zero retains every mode with a nonzero computed singular value.

    Returns
    -------
    rank : int
        Smallest retained dimension satisfying the criterion, possibly zero.
    tail_rms : numpy.ndarray
        Discarded RMS for every rank, including the zero tail at full rank.

    Notes
    -----
    The criterion is epsilon(r) = sqrt(sum(s[r:]**2) / n_vis) <= f. This is
    an average in noise-whitened visibility space, not a bound on each sample
    or on each realisation. The foreground prior is not applied a second time.
    """
    values = np.asarray(singular_values)
    n_vis = operator.index(n_vis)
    if values.ndim != 1 or np.iscomplexobj(values):
        raise ValueError("singular_values must be a real vector")
    values = values.astype(np.float64)
    if n_vis < 1 or not np.isfinite(f) or f < 0:
        raise ValueError("n_vis must be positive and f finite and non-negative")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("singular values must be finite and non-negative")
    if np.any(values[1:] > values[:-1]):
        raise ValueError("singular values must be ordered from largest to smallest")
    # Accumulate from the small end, avoiding cancellation and squared overflow.
    tail_rms = np.r_[np.hypot.accumulate(values[::-1] / np.sqrt(n_vis))[::-1], 0.0]
    rank = int(np.flatnonzero(tail_rms <= f)[0])
    return rank, tail_rms


def _svd_mesh():
    """Arrange the existing one-process-per-GPU job on a near-square mesh.

    Returns
    -------
    jax.sharding.Mesh
        Two-dimensional GPU mesh with axes svd_rows and svd_cols. Distributed
        JAX must be initialised by the launcher; this does not start workers.
    """
    devices = jax.devices()
    if any(device.platform != "gpu" for device in devices):
        raise RuntimeError("Foreground SVD requires a JAX GPU backend")
    if jax.local_device_count() != 1 or jax.process_count() != len(devices):
        raise RuntimeError("JAXMG foreground SVD requires one Python process per GPU")
    n_rows = math.isqrt(len(devices))
    while len(devices) % n_rows:
        n_rows -= 1
    return Mesh(np.asarray(devices).reshape(n_rows, -1), ("svd_rows", "svd_cols"))


def _svd_padded_shape(shape, process_grid, tile_size):
    """Choose zero-padded dimensions legal for both the input and thin U output.

    Parameters
    ----------
    shape : tuple of int
        Original visibility-by-coefficient dimensions.
    process_grid : tuple of int
        Numbers of GPU rows and columns.
    tile_size : int
        Positive square JAXMG solver-tile width.

    Returns
    -------
    tuple of int
        Dimensions divisible by both mesh axes and the solver tile width.
    """
    # min(M, N) becomes U's column count, so padding must also fit that layout.
    alignment = math.lcm(*process_grid) * tile_size
    return tuple((size + alignment - 1) // alignment * alignment for size in shape)


def _raise_if_any(failed, message):
    """Raise on every process if any rank reports invalid data or a solver failure.

    Parameters
    ----------
    failed : bool
        Failure flag for the local process.
    message : str
        Explanation included in the RuntimeError.

    Returns
    -------
    None
        Returns only when all participating processes report success.
    """
    flags = np.asarray([failed], dtype=np.int32)
    if jax.distributed.is_initialized():
        flags = np.asarray(multihost_utils.process_allgather(flags, tiled=False)).ravel()
    if np.any(flags):
        raise RuntimeError(f"{message}; failing process indices: {np.flatnonzero(flags).tolist()}")


def T_fg_svd_build_save(weighted_operator_path, ninv_path, *, output_path, f=0.1, tile_size=128):
    r"""Distribute, decompose and truncate A, then save T_fg,r = N^1/2 U_r S_r.

    Parameters
    ----------
    weighted_operator_path : pathlib.Path or str
        Saved complex128 matrix A = N^-1/2 T_fg Sigma_q^1/2, shape
        (n_visibility, n_foreground_coefficient), in time/frequency/baseline
        row order. Prior and noise weighting must already have been applied.
    ninv_path : pathlib.Path or str
        Saved positive diagonal inverse noise E[|n_i|^2]^-1, shape
        (n_visibility,). Must be the same diagonal noise used to construct A.
        Correlated-noise whitening is not supported by this saved-file routine.
    output_path : pathlib.Path or str
        Shared destination, normally T_fg_reduced.npy. Also saves singular
        values to <stem>.singular_values.npy and diagnostics to <stem>.svd.json.
    f : float, optional
        Maximum discarded RMS as a fraction of thermal-noise RMS, averaged
        over the original visibility vector. Defaults to 0.1; see
        select_foreground_rank for the precise criterion.
    tile_size : int, optional
        Positive JAXMG square tile width, default 128.

    Returns
    -------
    pathlib.Path
        Completed complex128 operator, shape (n_visibility, retained_rank),
        ready for joint assembly with a standard-normal foreground prior.

    Notes
    -----
    Every process calls together, with one GPU per process and shared storage.
    Only local shards are read from disk; zero padding permits the distributed
    JAXMG GESVD. Full U is never gathered, and Vh is not computed. This is a
    dense SVD: the input, U and solver workspace must fit the allocated GPUs.
    Padding is excluded from the output and the RMS normalisation.

    This function always computes its requested product, independently of setup
    and cache policy. It neither constructs A nor changes the inference prior.
    Existing outputs are replaced only after successful computation. No GPU
    fallback or approximate/randomised decomposition is used.
    """
    tile_size = operator.index(tile_size)
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    source = np.load(weighted_operator_path, mmap_mode="r")
    if source.ndim != 2 or source.dtype != np.complex128 or min(source.shape) < 1:
        raise ValueError("Weighted operator must be a nonempty complex128 matrix")
    n_vis, n_parameter = source.shape
    select_foreground_rank([], n_vis, f)  # Validate the threshold before GPU allocation.
    inverse_noise = np.load(ninv_path, mmap_mode="r")
    if inverse_noise.shape != (n_vis,):
        raise ValueError("Inverse noise must match the original visibility rows")
    if (not np.all(np.isfinite(inverse_noise)) or np.any(inverse_noise.imag != 0)
            or np.any(inverse_noise.real <= 0)):
        raise ValueError("Inverse noise must be finite, real and positive")
    noise_rms = 1.0 / np.sqrt(inverse_noise.real)

    path = Path(output_path)
    spectrum_path = path.with_name(f"{path.stem}.singular_values.npy")
    metadata_path = path.with_name(f"{path.stem}.svd.json")
    products = (path, spectrum_path, metadata_path)
    for product in products:
        _check_output_path(product, weighted_operator_path, ninv_path,
                           *(other for other in products if other != product))
    _, temporary = _check_output_path(path, weighted_operator_path, ninv_path)
    mesh = _svd_mesh()
    try:
        from jaxmg import gesvd
    except ImportError as exc:
        raise RuntimeError("Foreground SVD requires JAXMG with the gesvd interface") from exc
    specs = P("svd_rows", "svd_cols")
    padded_shape = _svd_padded_shape(source.shape, mesh.devices.shape, tile_size)
    start = time.perf_counter()
    invalid_input = False

    def load_shard(index):
        # Read only this GPU's rectangle; nonexistent rows/columns stay zero.
        nonlocal invalid_input
        r0, r1 = _slice_bounds(index[0], padded_shape[0])
        c0, c1 = _slice_bounds(index[1], padded_shape[1])
        block = np.zeros((r1-r0, c1-c0), dtype=np.complex128)
        nr, nc = max(0, min(r1, n_vis)-r0), max(0, min(c1, n_parameter)-c0)
        if nr and nc:
            block[:nr, :nc] = source[r0:r0+nr, c0:c0+nc]
            invalid_input |= not np.all(np.isfinite(block[:nr, :nc]))
        return block

    matrix = jax.make_array_from_callback(padded_shape, NamedSharding(mesh, specs), load_shard)
    matrix.block_until_ready()
    _raise_if_any(invalid_input, "Weighted operator contains nonfinite entries")
    if jax.process_index() == 0:
        LOGGER.info("Foreground SVD: %s padded to %s on GPU grid %s",
                    source.shape, padded_shape, mesh.devices.shape)
    svd_start = time.perf_counter()
    u, singular_values, status = gesvd(
        matrix, T_A=tile_size, mesh=mesh, matrix_specs=specs,
        compute_u=True, compute_vh=False, full_matrices=False,
        return_status=True, donate=True,
    )
    status.block_until_ready()
    svd_seconds = time.perf_counter() - svd_start
    _raise_if_any(any(np.any(np.asarray(shard.data) != 0) for shard in status.addressable_shards),
                  "JAXMG GESVD failed")
    # JAXMG replicates s; discard the extra singular modes introduced by padding.
    values = np.asarray(singular_values)[:min(n_vis, n_parameter)]
    rank, tail_rms = select_foreground_rank(values, n_vis, f)
    output = _prepare_memmap_output(temporary, shape=(n_vis, rank), dtype=np.dtype(np.complex128),
                                    sync_name=f"{path.name}-allocated")
    invalid_output = False
    for shard in u.addressable_shards:
        if shard.replica_id != 0:
            continue
        r0, r1 = _slice_bounds(shard.index[0], u.shape[0])
        c0, c1 = _slice_bounds(shard.index[1], u.shape[1])
        r1, c1 = min(r1, n_vis), min(c1, rank)
        if r1 <= r0 or c1 <= c0:
            continue
        # Transfer only retained physical rows/columns, not padded or omitted U.
        block = np.array(shard.data[:r1-r0, :c1-c0], copy=True)
        block *= values[c0:c1][None, :]
        block *= noise_rms[r0:r1, None]
        invalid_output |= not np.all(np.isfinite(block))
        output[r0:r1, c0:c1] = block
    output.flush()
    del output, u
    _raise_if_any(invalid_output, "Reduced foreground operator contains nonfinite entries")
    publish_output(temporary, path)
    save_array(spectrum_path, values)
    metadata = dict(
        definition="A = N^-1/2 T_fg Sigma_q^1/2 = U S Vh; T_fg,r = N^1/2 U_r S_r",
        weighted_operator_path=str(Path(weighted_operator_path).resolve()),
        inverse_noise_path=str(Path(ninv_path).resolve()),
        reduced_operator_path=str(path.resolve()),
        original_shape=list(source.shape), padded_shape=list(padded_shape),
        process_grid=list(mesh.devices.shape), tile_size=tile_size,
        retained_rank=rank, threshold_over_noise_rms=float(f),
        discarded_rms_over_noise_rms=float(tail_rms[rank]),
        coefficient_prior="standard-normal", svd_seconds=svd_seconds,
        total_seconds=time.perf_counter()-start,
    )
    _, metadata_temporary = _check_output_path(metadata_path, *products[:2])
    if jax.process_index() == 0:
        metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    publish_output(metadata_temporary, metadata_path)
    return path
