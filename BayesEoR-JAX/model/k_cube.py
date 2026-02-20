"""
Utilities for constructing Fourier-space `k` coordinates and assigning k-modes to spherical/cylindrical bins.

Conventions
-----------
- Spatial-frequency units are `Mpc^-1`.
- Native cube axis order is `(eta, v, u)` (parallel, Dec-like transverse, RA-like transverse).
- Flattened masked arrays use `(uv, eta)` block ordering with `eta` varying fastest within each `uv` block.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def generate_k_cube_in_physical_coordinates(
    n_u: int,
    n_v: int,
    n_eta: int,
    ps_box_size_ra_mpc: float,
    ps_box_size_dec_mpc: float,
    ps_box_size_para_mpc: float,
):
    """
    Construct Fourier coordinates on an `(eta, v, u)` lattice and compute `|k|`.

    Parameters
    ----------
    n_u, n_v, n_eta : int
        Number of discrete Fourier modes along `(u, v, eta)` axes.
    ps_box_size_ra_mpc : float
        Comoving box size along RA-like transverse axis, in `Mpc`.
    ps_box_size_dec_mpc : float
        Comoving box size along Dec-like transverse axis, in `Mpc`.
    ps_box_size_para_mpc : float
        Comoving box size along line-of-sight axis, in `Mpc`.

    Returns
    -------
    mod_k : jnp.ndarray
        Shape `(n_eta, n_v, n_u)`, magnitude `sqrt(k_x^2 + k_y^2 + k_z^2)` in `Mpc^-1`.
    k_x, k_y, k_z : jnp.ndarray
        Shape `(n_eta, n_v, n_u)`, Cartesian Fourier components in `Mpc^-1`.
    x, y, z : jnp.ndarray
        Shape `(n_eta, n_v, n_u)`, centered integer mode indices corresponding to `(u, v, eta)`.

    Notes
    -----
    Mode indices are centered with `arange(n) - n//2`, which gives one consistent
    zero-mode location for both even and odd axis lengths.
    """
    # Construct each K-Space axis with shape `(n_axis,)` correctly centered around zero
    # Approach is consistent for both even and odd `n_axis`
    u_inds = jnp.arange(n_u, dtype=jnp.int32) - (n_u // 2)
    v_inds = jnp.arange(n_v, dtype=jnp.int32) - (n_v // 2)
    eta_inds = jnp.arange(n_eta, dtype=jnp.int32) - (n_eta // 2)

    # Construct a meshgrid of mode indices with shape `(n_eta, n_v, n_u)`
    z, y, x = jnp.meshgrid(eta_inds, v_inds, u_inds, indexing="ij")

    # Convert index-based modes to physical Fourier coordinates in `Mpc^-1` using the box size.
    # Calculate the conversion factor for each axis: `delta_k = 2*pi / L`, where `L` is the box size along that axis.
    delta_kx = (2.0 * jnp.pi) / ps_box_size_ra_mpc
    delta_ky = (2.0 * jnp.pi) / ps_box_size_dec_mpc
    delta_kz = (2.0 * jnp.pi) / ps_box_size_para_mpc

    # Apply the conversion to get physical Fourier coordinates of each axis
    k_x = x.astype(jnp.float64) * delta_kx
    k_y = y.astype(jnp.float64) * delta_ky
    k_z = z.astype(jnp.float64) * delta_kz

    # Compute the magnitude of the Fourier vector at each k-mode: `|k| = sqrt(k_x^2 + k_y^2 + k_z^2)`.
    mod_k = jnp.sqrt(k_x**2 + k_y**2 + k_z**2)

    # Return
    return mod_k, k_x, k_y, k_z, x, y, z


def mask_k_cube(mod_k: jnp.ndarray) -> jnp.ndarray:
    """
    Mask DC-like modes and flatten to 1D in a deterministic `(uv, eta)` order.

    Parameters
    ----------
    mod_k : jnp.ndarray
        Shape `(n_eta, n_v, n_u)`, Fourier magnitude values in `Mpc^-1`.

    Returns
    -------
    jnp.ndarray
        Shape `((n_u*n_v - 1) * (n_eta - 1),)`.
        Excludes:
        1) the `eta=0` plane (line-of-sight DC mode),
        2) the transverse monopole `(u, v) = (0, 0)`.
        Ordering is by transverse pixel (`uv`) first, with `eta` varying fastest.
    """
    n_eta, n_v, n_u = mod_k.shape

    # Remove the `eta=0` slice to, removing the constant DC mode along the line of sight.
    mod_k_no_eta0 = jnp.delete(mod_k, n_eta // 2, axis=0)

    # Flatten `(v, u)` at fixed `eta`; centered zero-mode index maps to `(v//2, u//2)`.
    mod_k_uv = mod_k_no_eta0.reshape((n_eta - 1, n_u * n_v))

    # The `(u, v) = (0, 0)` flat index in `(v-major, u-minor)` ordering is:
    # `i_uv0 = (n_v//2) * n_u + (n_u//2)`.
    # This is valid for both even and odd `(n_u, n_v)` because centered indices
    # are built with `arange(n) - n//2`, so zero always lives at index `n//2`.
    uv_monopole_flat_index = (n_v // 2) * n_u + (n_u // 2)
    mod_k_no_monopole = jnp.delete(mod_k_uv, uv_monopole_flat_index, axis=1)

    # (eta, uv) -> (uv, eta), then flatten to keep eta varying fastest in each uv block.
    return mod_k_no_monopole.T.reshape(-1)


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
