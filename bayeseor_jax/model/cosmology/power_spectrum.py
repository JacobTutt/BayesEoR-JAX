"""
Utilities for constructing Fourier-space `k` coordinates and assigning k-modes to spherical/cylindrical bins.

Conventions
-----------
- Spatial-frequency units are `Mpc^-1`.
- Flattened arrays use `(uv, eta)` block ordering with `eta` varying fastest
  within each transverse mode.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def generate_k_modes_in_physical_coordinates(
    uv: jax.Array,
    eta: jax.Array,
    transverse_distance_mpc: float,
    line_of_sight_distance_mpc_per_hz: float,
) -> jax.Array:
    """Convert retained instrumental Fourier modes to cosmological wavenumbers.

    Parameters
    ----------
    uv : jax.Array
        Transverse angular Fourier coordinates, shape ``(2, n_uv_mode)``,
        in inverse radians; these are model modes, not sampled baselines.
    eta : jax.Array
        Line-of-sight Fourier coordinates, shape ``(n_eta_mode,)``, in inverse Hz.
    transverse_distance_mpc : float
        Transverse comoving distance ``D_M`` at the reference redshift, in Mpc.
    line_of_sight_distance_mpc_per_hz : float
        Positive conversion factor ``|dchi/df|`` at that redshift, in Mpc/Hz.

    Returns
    -------
    mod_k : jax.Array
        Wavenumber magnitudes in inverse Mpc, shape ``(n_uv_mode*n_eta_mode,)``,
        with eta varying fastest to match the cosmological operator columns.

    Notes
    -----
    The conversion is ``(k_x, k_y) = 2*pi*(u, v)/D_M`` and
    ``k_z = 2*pi*eta/|dchi/df|``. Each Cartesian combination receives a
    spherical magnitude used to assign its power-spectrum prior bin.
    """
    # Preserve the retained-mode ordering supplied by the Fourier geometry.
    uv = jnp.asarray(uv, dtype=jnp.float64)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    if uv.ndim != 2 or uv.shape[0] != 2:
        raise ValueError("uv must have shape (2, n_uv_mode)")
    if eta.ndim != 1:
        raise ValueError("eta must have shape (n_eta_mode,)")

    # Convert angular and frequency Fourier scales into comoving wavenumbers.
    k_x = 2.0 * jnp.pi * uv[0] / transverse_distance_mpc
    k_y = 2.0 * jnp.pi * uv[1] / transverse_distance_mpc
    k_z = 2.0 * jnp.pi * eta / line_of_sight_distance_mpc_per_hz
    # Broadcast every transverse mode against every eta mode before flattening.
    return jnp.sqrt(
        k_x[:, None] ** 2
        + k_y[:, None] ** 2
        + k_z[None, :] ** 2
    ).reshape(-1)


def _spherical_bin_edges(
    ps_box_size_para_mpc: float,
    num_k_bins: int = 50,
    growth_factor: float = 1.35,
):
    """
    Construct geometric spherical-bin edges in `|k|`.

    Parameters
    ----------
    ps_box_size_para_mpc : float
        Line-of-sight box size `L_parallel` in `Mpc`.
    num_k_bins : int, default=50
        Number of spherical `|k|` bins.
    growth_factor : float, default=1.35
        Geometric ratio `g` used for successive upper edges.

    Returns
    -------
    bin_low : jnp.ndarray
        Shape `(num_k_bins,)`, lower edges in `Mpc^-1`.
    bin_high : jnp.ndarray
        Shape `(num_k_bins,)`, upper edges in `Mpc^-1`.

    Notes
    -----
    `Delta k_parallel = 2*pi/L_parallel`, `bin_high[i] = 2*Delta k_parallel*g^i`,
    and `bin_low[i] = bin_high[i-1]` for `i>0` with `bin_low[0]=0`.
    """
    if num_k_bins <= 0:
        raise ValueError("num_k_bins must be > 0")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be > 1")

    # Construct the fundamental LOS Fourier spacing from the parallel box size.
    # This sets the physical scale anchor for all spherical-bin edges.
    delta_k_para = (2.0 * jnp.pi) / ps_box_size_para_mpc
    # Construct the first bin upper edge as `2 * Delta k_parallel`.
    # This follows the historical setup where the very smallest LOS mode is not isolated as its own bin.
    first_high = 2.0 * delta_k_para

    # Construct geometric upper edges and matching lower edges.
    # Equations: `bin_high[i] = first_high * growth_factor^i`, `bin_low[i] = bin_high[i-1]`, `bin_low[0] = 0`.
    i = jnp.arange(num_k_bins, dtype=jnp.float64)
    bin_high = first_high * (growth_factor**i)
    bin_low = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), bin_high[:-1]])

    # Return edge arrays with shape `(num_k_bins,)`.
    return bin_low, bin_high


def generate_k_cube_model_spherical_binning(
    mod_k_vo: jnp.ndarray,
    ps_box_size_para_mpc: float,
    num_k_bins: int = 50,
    growth_factor: float = 1.35,
):
    """
    Assign each k-mode to a spherical `|k|` bin.

    Parameters
    ----------
    mod_k_vo : jnp.ndarray
        Shape `(n_k_modes,)`, flattened `|k|` values in `Mpc^-1`.
    ps_box_size_para_mpc : float
        Line-of-sight box size in `Mpc`, used to define geometric bin edges.
    num_k_bins : int, default=50
        Number of spherical bins.
    growth_factor : float, default=1.35
        Geometric ratio between successive upper edges.

    Returns
    -------
    bin_id_per_k_mode : jnp.ndarray
        Shape `(n_k_modes,)`, integer bin index in `[0, num_k_bins)` or `-1` if invalid.
    valid_mask : jnp.ndarray
        Shape `(n_k_modes,)`, `True` where k-mode is inside configured bins.
    bin_low, bin_high : jnp.ndarray
        Shape `(num_k_bins,)`, spherical bin edges in `Mpc^-1`.
    counts : jnp.ndarray
        Shape `(num_k_bins,)`, number of valid k-modes per bin.

    Notes
    -----
    With `searchsorted(..., side="left")` and `k > bin_low`, intervals are `(bin_low, bin_high]`.
    Values above the last `bin_high` are marked invalid.
    """
    k = jnp.asarray(mod_k_vo, dtype=jnp.float64)
    # Construct deterministic spherical bin edges in physical units.
    bin_low, bin_high = _spherical_bin_edges(
        ps_box_size_para_mpc=ps_box_size_para_mpc,
        num_k_bins=num_k_bins,
        growth_factor=growth_factor,
    )

    # Construct raw bin indices from upper-edge search.
    # `searchsorted(..., side="left")` yields the first `i` such that `k <= bin_high[i]`.
    raw_bin = jnp.searchsorted(bin_high, k, side="left")
    in_range = raw_bin < num_k_bins
    clipped = jnp.clip(raw_bin, 0, num_k_bins - 1)
    # Enforce the lower-edge condition so intervals are exactly `(bin_low, bin_high]`.
    # This removes ambiguity for k-modes exactly on shared boundaries.
    lower_ok = k > bin_low[clipped]
    valid_mask = in_range & lower_ok

    # Convert invalid assignments to sentinel `-1`.
    # Shape is fixed: `(n_k_modes,)`.
    bin_id_per_k_mode = jnp.where(valid_mask, clipped, -1).astype(jnp.int32)

    # Construct per-bin counts with a static-shape reduction.
    # Invalid k-modes are routed to bin 0 but weighted by zero, so they do not contribute.
    safe_bin = jnp.where(valid_mask, clipped, 0)
    counts = jnp.bincount(
        safe_bin,
        weights=valid_mask.astype(jnp.int32),
        length=num_k_bins,
    ).astype(jnp.int32)

    # Return fixed-shape outputs so downstream JAX transforms avoid ragged structures.
    return bin_id_per_k_mode, valid_mask, bin_low, bin_high, counts


def calc_mean_binned_k_vals(
    mod_k_vo: jnp.ndarray,
    bin_id_per_k_mode: jnp.ndarray,
    valid_mask: jnp.ndarray,
    num_k_bins: int,
):
    """
    Compute mean `|k|` per spherical bin from k-mode assignments.

    Parameters
    ----------
    mod_k_vo : jnp.ndarray
        Shape `(n_k_modes,)`, k-mode `|k|` values in `Mpc^-1`.
    bin_id_per_k_mode : jnp.ndarray
        Shape `(n_k_modes,)`, integer bin IDs with invalid k-modes encoded as `-1`.
    valid_mask : jnp.ndarray
        Shape `(n_k_modes,)`, marks valid k-mode bin assignments.
    num_k_bins : int
        Total number of bins.

    Returns
    -------
    k_vals : jnp.ndarray
        Shape `(num_k_bins,)`, per-bin mean `|k|` in `Mpc^-1`; empty bins are `NaN`.
    """
    k = jnp.asarray(mod_k_vo, dtype=jnp.float64)
    bid = jnp.asarray(bin_id_per_k_mode, dtype=jnp.int32)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)

    # Construct a safe bin-id vector for reduction.
    # Invalid k-modes are temporarily remapped to bin 0 and weighted to zero.
    safe_bin = jnp.where(valid, bid, 0)
    weights = valid.astype(jnp.float64)

    # Compute per-bin numerator and denominator in vectorized form.
    # Mean equation per bin `b`: `k_mean[b] = sum(k_i)/count_i` over valid assignments in bin `b`.
    sums = jnp.bincount(safe_bin, weights=jnp.where(valid, k, 0.0), length=num_k_bins)
    counts = jnp.bincount(safe_bin, weights=weights, length=num_k_bins)

    # Return NaN for empty bins to make missing support explicit.
    return jnp.where(counts > 0.0, sums / counts, jnp.nan)


def compress_to_nonempty_k_bins(
    k_vals_full: jnp.ndarray,
    counts: jnp.ndarray,
    bin_id_per_k_mode: jnp.ndarray,
    valid_mask: jnp.ndarray,
):
    """
    Drop empty spherical bins and remap voxel bin IDs to compact indices.

    Parameters
    ----------
    k_vals_full : jnp.ndarray
        Shape ``(num_k_bins,)`` mean ``|k|`` values, including empty bins.
    counts : jnp.ndarray
        Shape ``(num_k_bins,)`` k-mode counts per bin.
    bin_id_per_k_mode : jnp.ndarray
        Shape ``(n_k_modes,)`` original bin IDs with invalid entries as ``-1``.
    valid_mask : jnp.ndarray
        Shape ``(n_k_modes,)`` validity mask for original bin assignments.

    Returns
    -------
    k_vals : jnp.ndarray
        Mean ``|k|`` values for non-empty bins only.
    compact_bid : jnp.ndarray
        Shape ``(n_k_modes,)`` remapped bin IDs in ``[0, n_nonempty)`` and
        ``-1`` for invalid/empty-bin assignments.
    compact_valid : jnp.ndarray
        Shape ``(n_k_modes,)`` mask for valid remapped assignments.
    """
    # Keep only bins that actually contain at least one k-mode.
    keep_bins = counts > 0
    k_vals = k_vals_full[keep_bins]

    # Build old-id -> new-id map, assigning -1 to empty bins.
    n_bins = int(k_vals_full.shape[0])
    old_to_new = -jnp.ones((n_bins,), dtype=jnp.int32)
    kept_old_ids = jnp.where(keep_bins)[0]
    old_to_new = old_to_new.at[kept_old_ids].set(
        jnp.arange(kept_old_ids.shape[0], dtype=jnp.int32)
    )

    bid = jnp.asarray(bin_id_per_k_mode, dtype=jnp.int32)
    valid = jnp.asarray(valid_mask, dtype=bool)
    bid_clip = jnp.clip(bid, 0, n_bins - 1)
    mapped = old_to_new[bid_clip]
    compact_valid = valid & (mapped >= 0)
    compact_bid = jnp.where(compact_valid, mapped, -1).astype(jnp.int32)

    return k_vals, compact_bid, compact_valid


def generate_k_cube_model_cylindrical_binning(
    mod_k_masked: jnp.ndarray,
    k_z_masked: jnp.ndarray,
    k_y_masked: jnp.ndarray,
    k_x_masked: jnp.ndarray,
    n_k_perp_bins: int,
    ps_box_size_para_mpc: float,
    num_k_parallel_bins: int = 50,
    growth_factor_parallel: float = 1.5,
):
    """
    Assign k-modes to cylindrical bins in `(k_parallel, k_perp)`.

    Parameters
    ----------
    mod_k_masked : jnp.ndarray
        Shape `(n_k_modes,)`, masked `|k|` values in `Mpc^-1` (currently passed through for API consistency).
    k_z_masked, k_y_masked, k_x_masked : jnp.ndarray
        Shape `(n_k_modes,)`, Cartesian Fourier components in `Mpc^-1`.
    n_k_perp_bins : int
        Number of `k_perp` edge points. Number of `k_perp` intervals is `n_k_perp_bins - 1`.
    ps_box_size_para_mpc : float
        Line-of-sight box size in `Mpc`, used for geometric `k_parallel` edges.
    num_k_parallel_bins : int, default=50
        Number of `k_parallel` intervals.
    growth_factor_parallel : float, default=1.5
        Geometric ratio for `k_parallel` upper edges.

    Returns
    -------
    cyl_bin_id_per_k_mode : jnp.ndarray
        Shape `(n_k_modes,)`, flattened 2D bin IDs (`-1` invalid).
    valid_mask : jnp.ndarray
        Shape `(n_k_modes,)`, valid cylindrical assignments.
    k_parallel_low, k_parallel_high : jnp.ndarray
        Shape `(num_k_parallel_bins,)`, `k_parallel` edges in `Mpc^-1`.
    k_perp_low, k_perp_high : jnp.ndarray
        Shape `(n_k_perp_bins - 1,)`, `k_perp` interval edges in `Mpc^-1`.
    counts_2d : jnp.ndarray
        Shape `(num_k_parallel_bins, n_k_perp_bins - 1)`, k-mode counts per cylindrical bin.

    Notes
    -----
    `k_parallel = |k_z|` and `k_perp = sqrt(k_x^2 + k_y^2)`.
    Flattening convention is `flat_id = k_parallel_id * n_k_perp_intervals + k_perp_id`,
    so reshaping back uses row-major `(k_parallel, k_perp)` ordering.
    """
    if n_k_perp_bins <= 1:
        raise ValueError("n_k_perp_bins must be > 1")

    mod_k = jnp.asarray(mod_k_masked, dtype=jnp.float64)
    k_z = jnp.asarray(k_z_masked, dtype=jnp.float64)
    k_x = jnp.asarray(k_x_masked, dtype=jnp.float64)
    k_y = jnp.asarray(k_y_masked, dtype=jnp.float64)

    # Construct cylindrical coordinates from Cartesian Fourier components.
    # Definitions: `k_parallel = |k_z|`, `k_perp = sqrt(k_x^2 + k_y^2)`.
    k_parallel = jnp.abs(k_z)
    k_perp = jnp.sqrt(k_x**2 + k_y**2)

    # Construct geometric bin edges along the line-of-sight direction.
    # This currently mirrors the spherical-bin scaling schedule for continuity of bin sizing.
    # TODO(jacobtutt): evaluate a pure cylindrical edge model defined directly in `k_parallel`
    # (independent of historical spherical-style scaling) and compare inference stability.
    k_parallel_low, k_parallel_high = _spherical_bin_edges(
        ps_box_size_para_mpc=ps_box_size_para_mpc,
        num_k_bins=num_k_parallel_bins,
        growth_factor=growth_factor_parallel,
    )

    # Construct `k_parallel` bin IDs with interval rule `(low, high]`.
    kp_raw = jnp.searchsorted(k_parallel_high, k_parallel, side="left")
    kp_in_range = kp_raw < num_k_parallel_bins
    kp_clipped = jnp.clip(kp_raw, 0, num_k_parallel_bins - 1)
    kp_lower_ok = k_parallel > k_parallel_low[kp_clipped]
    # Exclude `k_z = 0` by construction to avoid the LOS DC plane in cylindrical power estimates.
    kz_nonzero = k_z != 0.0
    kp_valid = kp_in_range & kp_lower_ok & kz_nonzero

    # Construct transverse-bin edges from the observed `k_perp` range.
    # A small padding keeps extrema inside bins under floating-point rounding.
    kperp_min = jnp.nanmin(k_perp) * 0.99
    kperp_max = jnp.nanmax(k_perp) * 1.01
    # Match legacy convention: n_k_perp_bins is number of edge points,
    # so number of intervals is n_k_perp_bins - 1.
    kperp_edges = jnp.linspace(kperp_min, kperp_max, n_k_perp_bins, dtype=jnp.float64)
    k_perp_low = kperp_edges[:-1]
    k_perp_high = kperp_edges[1:]
    n_k_perp_intervals = n_k_perp_bins - 1

    # Construct `k_perp` bin IDs with the same half-open/closed convention `(low, high]`.
    kt_raw = jnp.searchsorted(kperp_edges[1:], k_perp, side="left")
    kt_in_range = kt_raw < n_k_perp_intervals
    kt_clipped = jnp.clip(kt_raw, 0, n_k_perp_intervals - 1)
    kt_lower_ok = k_perp > k_perp_low[kt_clipped]
    kt_valid = kt_in_range & kt_lower_ok

    # A k-mode is valid for cylindrical statistics only if both axis-bin assignments are valid.
    valid_mask = kp_valid & kt_valid

    # Build safe indices for reduction.
    kp_safe = jnp.where(valid_mask, kp_clipped, 0)
    kt_safe = jnp.where(valid_mask, kt_clipped, 0)
    # Flatten 2D `(k_parallel, k_perp)` bin coordinates into one integer ID:
    # `flat_id = k_parallel_id * n_k_perp_intervals + k_perp_id`.
    flat_id = kp_safe * n_k_perp_intervals + kt_safe
    cyl_bin_id_per_k_mode = jnp.where(valid_mask, flat_id, -1).astype(jnp.int32)

    # Accumulate counts over flattened IDs, then reshape back to a 2D histogram.
    flat_counts = jnp.bincount(
        jnp.where(valid_mask, flat_id, 0),
        weights=valid_mask.astype(jnp.int32),
        length=num_k_parallel_bins * n_k_perp_intervals,
    ).astype(jnp.int32)
    # Inverse map of `flat_id`: rows index `k_parallel` bins and columns index `k_perp` bins.
    counts_2d = flat_counts.reshape((num_k_parallel_bins, n_k_perp_intervals))

    # Return fixed-shape cylindrical bin assignment and histogram products.
    return (
        cyl_bin_id_per_k_mode,
        valid_mask,
        k_parallel_low,
        k_parallel_high,
        k_perp_low,
        k_perp_high,
        counts_2d,
    )
