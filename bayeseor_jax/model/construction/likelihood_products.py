"""Prepare likelihood products from an explicitly selected, saved joint operator.

The physical model and any foreground compression have already been constructed.
This module knows only T, the inverse-noise diagonal, and the observed data. Pure
contractions define the mathematics; save functions handle device sharding and
publication. No operator is implicitly built, concatenated or reduced here.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from .parallel_layout import padded_visibility_partition
from .storage import _check_output_path, _prepare_memmap_output, _slice_bounds, ensure_cached, publish_output, save_array


def build_gram_matrix(operator, inverse_noise):
    r"""Compute G = T^H N^-1 T for supplied coefficient responses.

    Parameters
    ----------
    operator : jax.Array
        Joint coefficient-to-visibility matrix, shape (n_visibility, n_parameter).
    inverse_noise : jax.Array
        Real positive inverse variances E[|n_i|^2]^-1, shape (n_visibility,).

    Returns
    -------
    jax.Array
        Hermitian coefficient overlap matrix, shape (n_parameter, n_parameter).

    Notes
    -----
    Pure JAX contraction; the caller controls placement and validates inputs.
    Neither the coefficient prior nor the observed data enters this product.
    """
    return jnp.conj(operator).T @ (inverse_noise[:, None] * operator)


def build_data_projection(operator, inverse_noise, data):
    r"""Compute dbar = T^H N^-1 d without fitting the model coefficients.

    Parameters
    ----------
    operator : jax.Array
        Joint response matrix, shape (n_visibility, n_parameter).
    inverse_noise : jax.Array
        Positive inverse variances, shape (n_visibility,).
    data : jax.Array
        Complex observations in matching row order, shape (n_visibility,).

    Returns
    -------
    jax.Array
        Noise-weighted data projection, shape (n_parameter,).
    """
    return jnp.conj(operator).T @ (inverse_noise * data)


def build_data_power(inverse_noise, data):
    r"""Compute the scalar data term d^H N^-1 d in the Gaussian likelihood.

    Parameters
    ----------
    inverse_noise : jax.Array
        Positive inverse variances, shape (n_visibility,).
    data : jax.Array
        Complex observations in matching row order, shape (n_visibility,).

    Returns
    -------
    jax.Array
        Real scalar noise-weighted squared magnitude of the data.
    """
    return jnp.real(jnp.vdot(data, inverse_noise * data))


def _load_inverse_noise(path, n_visibility=None):
    """Read and validate the small inverse-noise vector before device placement."""
    values = np.load(path, mmap_mode="r")
    if values.ndim != 1 or (n_visibility is not None and values.size != n_visibility):
        raise ValueError("Inverse noise must be a vector matching the visibility rows")
    if not np.all(np.isfinite(values)) or np.any(values.imag != 0) or np.any(values.real <= 0):
        raise ValueError("Inverse noise must be finite, real and positive")
    return values.real


def _load_sharded_operator(t_path, ninv_path):
    """Load each process's visibility stripe onto the existing device mesh.

    Returns the mesh, padded T/Ninv arrays, true parameter count, row interval,
    and padded local row count. Each stripe must fit its assigned devices;
    output-column blocking does not stream the input operator from disk.
    """
    source = np.load(t_path, mmap_mode="r")
    if source.ndim != 2 or source.dtype != np.complex128 or source.shape[0] == 0:
        raise ValueError("T must be a complex128 matrix with nonempty visibility rows")
    n_visibility, n_parameter = source.shape
    inverse_noise = _load_inverse_noise(ninv_path, n_visibility)
    n_process, n_local = jax.process_count(), jax.local_device_count()
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.size != n_process * n_local:
        raise ValueError("Likelihood construction requires equal device counts per process")
    row0, row1, n_rows = padded_visibility_partition(
        n_visibility, n_process, n_local, jax.process_index(),
    )
    # Zero padding makes row/column sharding legal without changing the products.
    local_t = np.asarray(source[row0:row1])
    local_ninv = np.asarray(inverse_noise[row0:row1])
    n_columns = ((n_parameter + devices.size - 1) // devices.size) * devices.size
    pad_rows, pad_columns = n_rows - local_t.shape[0], n_columns - n_parameter
    if pad_rows or pad_columns:
        local_t = np.pad(local_t, ((0, pad_rows), (0, pad_columns)))
    if pad_rows:
        local_ninv = np.pad(local_ninv, (0, pad_rows))
    mesh = Mesh(devices.reshape(n_process, n_local), ("host", "dev"))
    with jax.set_mesh(mesh):
        operator = multihost_utils.host_local_array_to_global_array(local_t, mesh, P(("host", "dev"), None))
        inverse_noise = multihost_utils.host_local_array_to_global_array(local_ninv, mesh, P(("host", "dev")))
    operator.block_until_ready()
    inverse_noise.block_until_ready()
    return mesh, operator, inverse_noise, n_parameter, row0, row1, n_rows


@partial(
    jax.jit,
    in_shardings=(P(("host", "dev"), None), P(("host", "dev")), None, None),
    out_shardings=P(("host", "dev"), None),
    static_argnums=(4, 5),
)
def _gram_block(operator, inverse_noise, i0, j0, left_width, right_width):
    """Contract a fixed-shape pair of coefficient blocks over all visibility rows."""
    left = jax.lax.dynamic_slice(operator, (0, i0), (operator.shape[0], left_width))
    right = jax.lax.dynamic_slice(operator, (0, j0), (operator.shape[0], right_width))
    return jnp.conj(left).T @ (inverse_noise[:, None] * right)


def save_gram_matrix(t_path, ninv_path, *, output_path, block_cols=16384, right_block_batch=4):
    r"""Compute and save G = T^H N^-1 T in distributed coefficient blocks.

    Parameters
    ----------
    t_path, ninv_path : pathlib.Path or str
        Saved complex128 response matrix and positive inverse-noise vector.
    output_path : pathlib.Path or str
        Shared output file, normally T_Ninv_T.npy.
    block_cols : int, optional
        Target coefficient-block width, rounded up to a device-count multiple.
    right_block_batch : int, optional
        Number of adjacent right-hand blocks contracted in one launch.

    Returns
    -------
    pathlib.Path
        Complete unpadded coefficient overlap matrix.

    Notes
    -----
    All participating processes call together. Each process caches its input
    stripe on the default backend (typically GPU). Only the output is streamed
    in blocks. Strictly off-diagonal blocks are mirrored by Hermitian symmetry;
    disjoint output shards avoid overlapping writes. No prior is applied.
    """
    if block_cols < 1 or right_block_batch < 1:
        raise ValueError("Gram block sizes must be positive")
    path, temporary = _check_output_path(output_path, t_path, ninv_path)
    loaded = _load_sharded_operator(t_path, ninv_path)
    return _save_gram_matrix(loaded, path, temporary, block_cols, right_block_batch)


def _save_gram_matrix(loaded, path, temporary, block_cols, right_block_batch):
    """Write coefficient blocks using an already resident distributed operator."""
    mesh, operator, inverse_noise, n_parameter, *_ = loaded
    # Every left block, including the final padded block, must divide the mesh.
    width = ((int(block_cols) + mesh.size - 1) // mesh.size) * mesh.size
    output = _prepare_memmap_output(temporary, shape=(n_parameter, n_parameter),
                                    dtype=np.dtype(np.complex128), sync_name=f"{path.name}-allocated")
    with jax.set_mesh(mesh):
        for i0 in range(0, n_parameter, width):
            left_width = min(width, operator.shape[1] - i0)
            i1 = min(i0 + left_width, n_parameter)
            for j0 in range(i0, n_parameter, width * right_block_batch):
                right_width = min(width * right_block_batch, operator.shape[1] - j0)
                block = _gram_block(operator, inverse_noise, i0, j0, left_width, right_width)
                block.block_until_ready()
                for shard in block.addressable_shards:
                    r0, r1 = _slice_bounds(shard.index[0], left_width)
                    c0, c1 = _slice_bounds(shard.index[1], right_width)
                    r1, c1 = min(r1, n_parameter - i0), min(c1, n_parameter - j0)
                    if r1 <= r0 or c1 <= c0:
                        continue
                    values = np.asarray(shard.data)[:r1-r0, :c1-c0]
                    output[i0+r0:i0+r1, j0+c0:j0+c1] = values
                    # Do not mirror the diagonal tile: its rows already have owners.
                    mirror_start = max(c0, i1 - j0)
                    if mirror_start < c1:
                        output[j0+mirror_start:j0+c1, i0+r0:i0+r1] = values[:, mirror_start-c0:].conj().T
            output.flush()
    del output
    publish_output(temporary, path)
    return path


def _shard_data(data, t_path, loaded):
    """Validate observations and give them the operator's padded row partition."""
    data = np.asarray(data, dtype=np.complex128)
    shape = np.load(t_path, mmap_mode="r").shape
    if len(shape) != 2 or data.shape != (shape[0],) or not np.all(np.isfinite(data)):
        raise ValueError("data must be a finite vector matching the visibility rows")
    mesh, _, _, _, row0, row1, n_rows = loaded
    local_data = data[row0:row1]
    local_data = np.pad(local_data, (0, n_rows - local_data.size))
    with jax.set_mesh(mesh):
        return multihost_utils.host_local_array_to_global_array(local_data, mesh, P(("host", "dev")))


_project_data = jax.jit(
    build_data_projection,
    in_shardings=(P(("host", "dev"), None), P(("host", "dev")), P(("host", "dev"))),
    out_shardings=P(None),
)
_data_power = jax.jit(
    build_data_power,
    in_shardings=(P(("host", "dev")), P(("host", "dev"))),
    out_shardings=P(),
)


def save_data_projection(t_path, ninv_path, data, *, output_path):
    r"""Save dbar = T^H N^-1 d using the same device mesh for one or many devices.

    Parameters
    ----------
    t_path, ninv_path : pathlib.Path or str
        Saved response matrix and positive inverse-noise vector.
    data : array-like
        Complex observations, shape (n_visibility,), in the operator's row order.
    output_path : pathlib.Path or str
        Shared destination, normally dbar.npy.

    Returns
    -------
    pathlib.Path
        Completed coefficient-space data projection. Recomputed on every call.
    """
    path, _ = _check_output_path(output_path, t_path, ninv_path)
    loaded = _load_sharded_operator(t_path, ninv_path)
    mesh, operator, inverse_noise, n_parameter, *_ = loaded
    observations = _shard_data(data, t_path, loaded)
    with jax.set_mesh(mesh):
        projection = _project_data(operator, inverse_noise, observations)[:n_parameter]
    return save_array(path, projection)


def save_likelihood_products(t_path, ninv_path, data, *, output_directory,
                             clobber=False, block_cols=16384, right_block_batch=4):
    r"""Prepare G, dbar and data power with one distributed load of T and N^-1.

    Parameters
    ----------
    t_path, ninv_path : pathlib.Path or str
        Saved complex128 joint response and positive inverse-noise vector.
    data : array-like
        Finite complex observations in the operator's visibility-row order.
    output_directory : pathlib.Path or str
        Shared destination for T_Ninv_T.npy, dbar.npy and d_ninv_d.npy.
    clobber : bool, optional
        Recompute G even when its cached input-file signatures match.
    block_cols : int, optional
        Target Gram-block width, rounded up to a device-count multiple.
    right_block_batch : int, optional
        Number of adjacent right-hand Gram blocks contracted together.

    Returns
    -------
    tuple of pathlib.Path
        Gram matrix, data projection and scalar data-power paths, respectively.

    Notes
    -----
    All processes call together, using every device in the initialised JAX job.
    G is reused when T and N^-1 are unchanged; data-dependent terms are always
    refreshed. Resident input stripes are shared across all three contractions.
    """
    if block_cols < 1 or right_block_batch < 1:
        raise ValueError("Gram block sizes must be positive")
    directory = Path(output_directory)
    gram_path, temporary = _check_output_path(directory / "T_Ninv_T.npy", t_path, ninv_path)
    projection_path, _ = _check_output_path(directory / "dbar.npy", t_path, ninv_path)
    power_path, _ = _check_output_path(directory / "d_ninv_d.npy", t_path, ninv_path)
    loaded = _load_sharded_operator(t_path, ninv_path)
    mesh, operator, inverse_noise, n_parameter, *_ = loaded
    observations = _shard_data(data, t_path, loaded)
    ensure_cached(
        gram_path,
        lambda: _save_gram_matrix(loaded, gram_path, temporary, block_cols, right_block_batch),
        dependencies=(t_path, ninv_path), clobber=clobber,
    )
    with jax.set_mesh(mesh):
        projection = _project_data(operator, inverse_noise, observations)[:n_parameter]
        power = _data_power(inverse_noise, observations)
    save_array(projection_path, projection)
    save_array(power_path, power)
    return gram_path, projection_path, power_path


def save_data_power(ninv_path, data, *, output_path):
    r"""Save d^H N^-1 d independently of the foreground/cosmological representation.

    Parameters
    ----------
    ninv_path : pathlib.Path or str
        Saved positive inverse-noise vector.
    data : array-like
        Complex visibility vector in matching row order.
    output_path : pathlib.Path or str
        Shared destination, normally d_ninv_d.npy.

    Returns
    -------
    pathlib.Path
        Completed scalar data term. Recomputed on every call.
    """
    path, _ = _check_output_path(output_path, ninv_path)
    inverse_noise = _load_inverse_noise(ninv_path)
    data = np.asarray(data, dtype=np.complex128)
    if data.shape != inverse_noise.shape or not np.all(np.isfinite(data)):
        raise ValueError("data must be a finite vector matching the visibility rows")
    return save_array(path, build_data_power(jnp.asarray(inverse_noise), jnp.asarray(data)))
