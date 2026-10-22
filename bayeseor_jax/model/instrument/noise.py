"""Noise-model helpers for BayesEoR-JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def generate_gaussian_noise(
    sigma: float,
    s: jnp.ndarray,
    n_freq: int,
    n_t: int,
    uvw_array_m: jnp.ndarray,
    bl_red_array: jnp.ndarray,
    random_seed: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[int, int]]:
    """
    Generate and add Hermitian complex Gaussian noise to a noiseless vector.

    The random draw intentionally follows the old BayesEoR NumPy
    implementation so the realized noise matches the original pipeline for the
    same ``sigma`` and ``random_seed``:

    - real part seeded with ``random_seed``
    - imaginary part seeded with ``random_seed * 123``

    Parameters
    ----------
    sigma : float
        RMS amplitude for the complex noise. Real and imaginary components are
        drawn with standard deviation ``sigma/sqrt(2)``.
    s : jax.Array
        Noiseless visibility vector with shape ``(n_t * n_freq * n_bls,)``.
    n_freq : int
        Number of frequency channels.
    n_t : int
        Number of time integrations.
    uvw_array_m : jax.Array
        UVW coordinates in meters, shape ``(n_bls, 3)`` or ``(n_t, n_bls, 3)``.
    bl_red_array : jax.Array
        Baseline redundancy array with shape ``(n_t, n_bls)``,
        ``(n_t, n_bls, 1)``, ``(n_bls,)``, or ``(n_bls, 1)``.
    random_seed : int or None, optional
        Seed for deterministic noise generation. 

    Returns
    -------
    d : jax.Array
        Noisy visibility vector ``d = s + n``.
    complex_noise_hermitian : jax.Array
        Generated Hermitian noise vector with the same shape as ``s``.
    bl_conjugate_pairs_map : dict[int, int]
        Map from baseline index to its conjugate baseline index.
    """
    # Normalize UVW and redundancy shapes so later indexing logic is stable.
    uvw_ref = np.asarray(_normalise_uvw_for_pairing(uvw_array_m), dtype=np.float64)
    n_bls = int(uvw_ref.shape[0])
    n_data = n_t * n_freq * n_bls

    # Flatten and validate the noiseless input vector against expected size.
    s = np.asarray(s, dtype=np.complex128).reshape(-1)
    if s.size != n_data:
        raise ValueError(f"s has size {s.size}, expected {n_data} (= n_t*n_freq*n_bls)")

    # Redundancy is used as a per-baseline noise scaling factor.
    red_tf = np.asarray(_normalise_redundancy(bl_red_array, n_t=n_t, n_bls=n_bls))

    if sigma == 0.0:
        # Fast path: deterministic zero-noise output.
        complex_noise_hermitian = np.zeros(n_data, dtype=np.complex128)
    else:
        # Real and imaginary parts use separate NumPy seeds rather than a 
        # split JAX key.
        std = float(sigma) / np.sqrt(2.0)
        if random_seed:
            np.random.seed(int(random_seed))
        real_noise = np.random.normal(0.0, std, n_data)

        if random_seed:
            np.random.seed(int(random_seed) * 123)
        imag_noise = np.random.normal(0.0, std, n_data)
        complex_noise = (real_noise + 1j * imag_noise).astype(np.complex128)
        complex_noise_hermitian = complex_noise * (float(sigma) / complex_noise.std())

    # Build mapping between each baseline and its (-u,-v) conjugate partner.
    bl_conjugate_pairs_dict: dict[tuple[float, float], int] = {}
    bl_conjugate_pairs_map: dict[int, int] = {}
    for i in range(n_bls):
        uv = tuple(uvw_ref[i, :2].tolist())
        uv_neg = tuple((-uvw_ref[i, :2]).tolist())
        if uv_neg in bl_conjugate_pairs_dict:
            key = bl_conjugate_pairs_dict[uv_neg]
            bl_conjugate_pairs_dict[uv] = key
            bl_conjugate_pairs_map[key] = i
        else:
            bl_conjugate_pairs_dict[uv] = i

    # Hermitian enforcement and redundancy scaling follow the old nested-loop
    # implementation so baseline ordering and conjugate placement remain
    # identical between pipelines.
    for i_t in range(n_t):
        time_ind = i_t * n_bls * n_freq
        for i_freq in range(n_freq):
            freq_ind = i_freq * n_bls
            start_ind = time_ind + freq_ind
            for bl_ind, conj_bl_ind in bl_conjugate_pairs_map.items():
                complex_noise_hermitian[start_ind + conj_bl_ind] = np.conjugate(
                    complex_noise_hermitian[start_ind + bl_ind]
                )
            complex_noise_hermitian[start_ind : start_ind + n_bls] /= np.sqrt(red_tf[i_t])

    # Compose final noisy data vector.
    d = s + complex_noise_hermitian
    return (
        jnp.asarray(d, dtype=jnp.complex128),
        jnp.asarray(complex_noise_hermitian, dtype=jnp.complex128),
        bl_conjugate_pairs_map,
    )


def _normalise_uvw_for_pairing(uvw_array_m: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize UVW input for conjugate-baseline pairing logic.

    The Hermitian mapping step only needs a single baseline list
    ``(u, v, w)`` with shape ``(n_bls, 3)``.

    For the time-stacked case, this helper takes the first time slice
    ``uvw_array_m[0]`` to build the conjugate baseline map and reuses it 
    across all times/frequencies.

    Parameters
    ----------
    uvw_array_m : jax.Array
        UVW coordinates in meters with shape ``(n_bls, 3)`` or
        ``(n_t, n_bls, 3)``.

    Returns
    -------
    jax.Array
        UVW array with shape ``(n_bls, 3)`` used for conjugate-pair lookup.

    Raises
    ------
    ValueError
        If input does not have a valid UVW shape with trailing axis size 3.
    """
    # Validate either a single baseline list or a time-stacked sequence.
    uvw = jnp.asarray(uvw_array_m, dtype=jnp.float64)
    if uvw.ndim == 2:
        if uvw.shape[1] != 3:
            raise ValueError(f"uvw_array_m must have trailing axis size 3, got {uvw.shape}")
        return uvw
    if uvw.ndim == 3:
        if uvw.shape[2] != 3:
            raise ValueError(f"uvw_array_m must have trailing axis size 3, got {uvw.shape}")
        # Pair identities are determined once from the first integration.
        return uvw[0]
    raise ValueError("uvw_array_m must have shape (n_bls, 3) or (n_t, n_bls, 3)")


def _normalise_redundancy(bl_red_array: jnp.ndarray, n_t: int, n_bls: int) -> jnp.ndarray:
    """
    Normalize redundancy input to shape ``(n_t, n_bls)``.

    One-dimensional forms are broadcast across time, so each time integration
    uses the same per-baseline redundancy weights.

    Parameters
    ----------
    bl_red_array : jax.Array
        Redundancy array in one of the accepted shapes.
    n_t : int
        Number of time integrations.
    n_bls : int
        Number of baselines.

    Returns
    -------
    jax.Array
        Redundancy array with shape ``(n_t, n_bls)``.

    Raises
    ------
    ValueError
        If the input shape is incompatible with ``n_t`` and ``n_bls``.
    """
    # A static redundancy vector is broadcast, not reweighted, across integrations.
    red = jnp.asarray(bl_red_array)
    if red.ndim == 1:
        if red.shape[0] != n_bls:
            raise ValueError(f"bl_red_array has incompatible shape {red.shape} for n_bls={n_bls}")
        return jnp.broadcast_to(red[jnp.newaxis, :], (n_t, n_bls))
    # Accept time-dependent weights or a column vector of static weights.
    if red.ndim == 2:
        if red.shape == (n_t, n_bls):
            return red
        if red.shape == (n_bls, 1):
            return jnp.broadcast_to(jnp.squeeze(red, axis=1)[jnp.newaxis, :], (n_t, n_bls))
        if red.shape == (n_bls,):
            return jnp.broadcast_to(red[jnp.newaxis, :], (n_t, n_bls))
        raise ValueError(f"bl_red_array has incompatible shape {red.shape}")
    # Older saved arrays may carry an otherwise redundant singleton weight axis.
    if red.ndim == 3 and red.shape == (n_t, n_bls, 1):
        return jnp.squeeze(red, axis=-1)
    raise ValueError(f"bl_red_array must be 1D/2D or (n_t, n_bls, 1), got {red.shape}")
