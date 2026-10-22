"""Define the thermal-noise covariance of the interferometric observation.

The diagonal entries of N are E[|n_i|**2], in squared visibility units, rather
than the variance of either real component alone. This module prepares model
weights without generating noise realisations or constructing a dense matrix;
simulation of noise belongs to ``data.noise``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def build_noise_variance(
    n_time: int,
    n_frequency: int,
    n_baseline: int,
    *,
    sigma: float | None = None,
    redundancy=None,
    effective_noise=None,
) -> jax.Array:
    r"""Prepare the diagonal complex-visibility noise covariance for an observation.

    Parameters
    ----------
    n_time, n_frequency, n_baseline : int
        Positive observation dimensions in the forward operator's row order.
    sigma : float or None, optional
        Positive complex noise RMS before redundancy averaging, in visibility
        units. Required unless effective_noise supplies per-visibility RMS.
    redundancy : array-like or None, optional
        Positive baseline redundancies, shape (n_time, n_baseline) or
        (n_time, n_baseline, 1). Required when effective_noise is absent.
    effective_noise : array-like or None, optional
        Per-visibility noise amplitudes, shape (n_time*n_frequency*n_baseline,).
        Their squared magnitudes override the sigma/redundancy model.

    Returns
    -------
    variance : jax.Array
        Positive float64 variances E[|n_i|^2], in squared visibility units,
        flattened in time/frequency/baseline order.

    Notes
    -----
    For redundancy R_tb, N_(tfb) = sigma^2 / R_tb. Real and imaginary noise
    components each have half this variance. This setup-time function validates
    inputs on the host; it does not create a dense covariance or save a file.
    """
    if any(size < 1 for size in (n_time, n_frequency, n_baseline)):
        raise ValueError("Noise observation dimensions must be positive")
    n_visibility = n_time * n_frequency * n_baseline
    if effective_noise is not None:
        amplitude = np.asarray(effective_noise)
        if amplitude.shape != (n_visibility,):
            raise ValueError(f"effective_noise must have shape {(n_visibility,)}")
        variance = np.abs(amplitude) ** 2
    else:
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        red = np.asarray(redundancy, dtype=np.float64)
        if red.shape == (n_time, n_baseline, 1):
            red = red[..., 0]
        if red.shape != (n_time, n_baseline):
            raise ValueError("redundancy must have shape (n_time, n_baseline) or (n_time, n_baseline, 1)")
        if not np.all(np.isfinite(red) & (red > 0)):
            raise ValueError("redundancy must be finite and positive")
        # Repeat each time's baseline variances at every observing frequency.
        variance = np.broadcast_to(
            (sigma**2 / red)[:, None, :], (n_time, n_frequency, n_baseline),
        ).reshape(-1)
    if not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError("noise variances must be finite and positive")
    return jnp.asarray(variance, dtype=jnp.float64)
