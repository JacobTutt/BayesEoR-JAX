"""Distribute pure forward-operator evaluations and save their complete matrices.

Each process writes disjoint visibility-row blocks to shared storage. Numerical
evaluation stays in model.operators; this module owns CPU placement and file
publication. No prior weighting or KL reduction is performed here.
"""

from pathlib import Path

import jax
import numpy as np

from ..operators import T_21, T_fg
from .distribution import discover_cpu_resources, partition_blocks
from .storage import _prepare_memmap_output, publish_output


def _build_save(operator, lmn, uvw_lambda, beam, pixel_area_sr, factors, output_path) -> Path:
    """Evaluate one component in CPU time-frequency blocks and save it to shared storage.

    Parameters
    ----------
    operator : callable
        Pure JAX component function, either T_21 or T_fg.
    lmn : array-like
        Pixel directions, shape ``(n_time, 3, n_pixel)``.
    uvw_lambda : array-like
        Baselines in wavelengths, shape ``(n_frequency, 3, n_baseline)``.
    beam : array-like
        Power gains, shape ``(n_time, n_frequency, n_pixel)``.
    pixel_area_sr : float
        Positive solid angle per pixel in steradians.
    factors : tuple of array-like
        Component factors: ``(F_prime, F_z)`` or ``(Q,)``. The final factor's
        first axis always indexes frequency; preceding factors are reused.
    output_path : str or pathlib.Path
        Output NumPy file, accessible at the same path from every process.

    Returns
    -------
    path : pathlib.Path
        Completed operator file with time/frequency/baseline row ordering.

    Notes
    -----
    Uses already launched JAX processes, without creating worker pools or
    changing XLA threading. Every process must enter this function with the
    same global inputs and output path. Existing output is replaced only
    after successful construction; interrupted builds may leave a partial file.
    """
    resources = discover_cpu_resources()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")

    # Hold reusable inputs on the host; only the current response block is materialised.
    lmn, uvw_lambda, beam = (np.asarray(value) for value in (lmn, uvw_lambda, beam))
    factors = tuple(np.asarray(value) for value in factors)
    if pixel_area_sr <= 0:
        raise ValueError("pixel_area_sr must be positive")
    # Trace shapes and validate matching axes without constructing the full operator.
    shape = jax.eval_shape(operator, lmn, uvw_lambda, beam, pixel_area_sr, *factors).shape
    n_time, n_frequency, n_baseline = lmn.shape[0], uvw_lambda.shape[0], uvw_lambda.shape[2]
    if n_time < 1 or n_frequency < 1 or n_baseline < 1:
        raise ValueError("Construction requires nonempty time, frequency and baseline axes")
    start, stop = partition_blocks(n_time * n_frequency, resources.process_count, resources.process_index)
    output = _prepare_memmap_output(
        partial_path, shape=shape, dtype=np.dtype(np.complex128),
        sync_name=f"{path.name}-allocated",
    )
    print(
        f"[BuildMatrices] {path.name}: process {resources.process_index}/{resources.process_count}, "
        f"{resources.cpu_count} allocated logical CPUs, blocks [{start}, {stop})",
        flush=True,
    )

    # Device placement belongs here, not in the pure mathematical functions.
    evaluate = jax.jit(operator)
    with jax.default_device(resources.device):
        fixed_factors = tuple(jax.device_put(value, resources.device) for value in factors[:-1])
        for block_index in range(start, stop):
            i_t, i_f = divmod(block_index, n_frequency)
            block_inputs = (
                lmn[i_t:i_t + 1], uvw_lambda[i_f:i_f + 1],
                beam[i_t:i_t + 1, i_f:i_f + 1], pixel_area_sr,
            )
            block_inputs = tuple(jax.device_put(value, resources.device) for value in block_inputs)
            spectral = jax.device_put(factors[-1][i_f:i_f + 1], resources.device)
            block = evaluate(*block_inputs, *fixed_factors, spectral)
            # Converting to NumPy waits for CPU completion before the shared-file write.
            row_start = block_index * n_baseline
            output[row_start:row_start + n_baseline] = np.asarray(block)
        output.flush()
    del output
    publish_output(partial_path, path)
    return path


def T_21_build_save(
    lmn, uvw_lambda, beam, pixel_area_sr, transverse_operator, spectral_operator, *, output_path,
) -> Path:
    r"""Construct and save T_21 = F^-1 P F' F_z in distributed CPU blocks.

    Parameters
    ----------
    lmn : array-like
        Cosmological pixel directions, shape ``(n_time, 3, n_pixel)``.
    uvw_lambda : array-like
        Baselines in wavelengths, shape ``(n_frequency, 3, n_baseline)``.
    beam : array-like
        Horizon-masked power gains, shape ``(n_time, n_frequency, n_pixel)``.
    pixel_area_sr : float
        Positive pixel solid angle in steradians.
    transverse_operator : array-like
        F' on the same pixels, shape ``(n_pixel, n_uv_mode)``.
    spectral_operator : array-like
        F_z on the same frequencies, shape ``(n_frequency, n_eta_mode)``.
    output_path : str or pathlib.Path
        Shared output file, conventionally ``T_21.npy``.

    Returns
    -------
    path : pathlib.Path
        Complete complex128 matrix, shape
        ``(n_time*n_frequency*n_baseline, n_uv_mode*n_eta_mode)``.

    Notes
    -----
    All participating JAX processes call this function together. It evaluates
    the pure ``T_21`` function on CPU and writes disjoint rows. Use ``T_21``
    directly for an in-memory JAX computation with no file or hardware policy.
    """
    return _build_save(T_21, lmn, uvw_lambda, beam, pixel_area_sr,
                       (transverse_operator, spectral_operator), output_path)


def T_fg_build_save(
    lmn, uvw_lambda, beam, pixel_area_sr, spectral_operator, *, output_path,
) -> Path:
    r"""Construct and save T_fg = F^-1 P Q_z in distributed CPU blocks.

    Parameters
    ----------
    lmn : array-like
        Foreground pixel directions, shape ``(n_time, 3, n_pixel)``.
    uvw_lambda : array-like
        Baselines in wavelengths, shape ``(n_frequency, 3, n_baseline)``.
    beam : array-like
        Horizon-masked power gains, shape ``(n_time, n_frequency, n_pixel)``.
    pixel_area_sr : float
        Positive pixel solid angle in steradians.
    spectral_operator : array-like
        Q on the same frequencies, shape ``(n_frequency, n_spectral_component)``.
    output_path : str or pathlib.Path
        Shared output file, conventionally ``T_fg.npy``.

    Returns
    -------
    path : pathlib.Path
        Complete complex128 matrix, shape
        ``(n_time*n_frequency*n_baseline, n_spectral_component*n_pixel)``.

    Notes
    -----
    All participating JAX processes call this function together. It evaluates
    the pure ``T_fg`` function on CPU and writes disjoint rows. Foreground priors
    and KL retention are not applied: the saved columns remain physical spectral
    amplitudes at individual pixels.
    """
    return _build_save(T_fg, lmn, uvw_lambda, beam, pixel_area_sr,
                       (spectral_operator,), output_path)


def assemble_joint_operator(T_21_path, T_fg_path, *, output_path, block_rows=1024) -> Path:
    """Concatenate saved component columns without loading the full joint matrix.

    Parameters
    ----------
    T_21_path, T_fg_path : str or pathlib.Path
        Saved two-dimensional complex128 operators with identical visibility-row
        ordering. The foreground input may be the original physical operator or a
        retained operator with its prior already absorbed; the caller supplies
        the corresponding coefficient prior during inference. Both files must
        describe the same observation.
    output_path : str or pathlib.Path
        Shared destination, distinct from either input file; conventionally T.npy.
    block_rows : int, optional
        Positive number of visibility rows per copy block, default 1024.

    Returns
    -------
    path : pathlib.Path
        Complete matrix with cosmological columns followed by foreground columns.

    Notes
    -----
    Copies disjoint row ranges across already participating JAX processes.
    No numerical operator is rebuilt and neither input file is modified.
    """
    path = Path(output_path)
    inputs = (Path(T_21_path), Path(T_fg_path))
    if any(path.resolve() == source.resolve() for source in inputs):
        raise ValueError("Joint output must not overwrite a component input")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    # Map each input read-only and check the interface before allocating an output.
    left, right = (np.load(source, mmap_mode="r") for source in inputs)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Component operators must be matrices with the same row count")
    if left.dtype != np.complex128 or right.dtype != np.complex128:
        raise ValueError("Component operators must use complex128 precision")
    resources = discover_cpu_resources()
    n_rows, n_left = left.shape
    start, stop = partition_blocks((n_rows + block_rows - 1) // block_rows,
                                   resources.process_count, resources.process_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
    if any(partial_path.resolve() == source.resolve() for source in inputs):
        raise ValueError("Joint temporary output must not overwrite a component input")
    output = _prepare_memmap_output(
        partial_path, shape=(n_rows, n_left + right.shape[1]), dtype=np.dtype(np.complex128),
        sync_name=f"{path.name}-allocated",
    )
    # Copy into column slices directly rather than allocating a concatenated block.
    for index in range(start, stop):
        rows = slice(index * block_rows, min((index + 1) * block_rows, n_rows))
        output[rows, :n_left] = left[rows]
        output[rows, n_left:] = right[rows]
    output.flush()
    del output
    publish_output(partial_path, path)
    return path
