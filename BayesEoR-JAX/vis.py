"""
Visibility preprocessing helpers.

Global motivation
-----------------
Downstream matrix/posterior code expects a very specific visibility layout:
1) one flattened complex vector with explicit Hermitian baseline doubling;
2) matching instrument-model arrays (`uvws`, `redundancy`, `antpairs`) in the
   same baseline ordering.

This module is the translation layer from raw pyuvdata objects to that
likelihood-ready representation.
"""

import warnings
from pathlib import Path
import logging

import numpy as np
from astropy import units
from astropy.time import Time
from astropy.units import Quantity
from pyuvdata import UVData
from pyuvdata import __version__ as pyuvdata_version
from pyuvdata.utils import polstr2num
from .utils import save_numpy_dict

LOGGER = logging.getLogger(__name__)


def preprocess_uvdata(
    filepath,
    antenna_selection="cross",
    bl_cutoff=None,
    freq_idx_min=None,
    freq_min=None,
    freq_center=None,
    n_freqs_select=None,
    jd_idx_min=None,
    jd_min=None,
    jd_center=None,
    n_times_select=None,
    form_pi=True,
    pi_norm=1.0,
    pol="xx",
    redundant_avg=False,
    uniform_redundancy=False,
    calc_noise=False,
    return_uvd=False,
    save_vis=False,
    save_model=False,
    save_dir="./",
    clobber=False,
):
    """
    Read visibility data from disk and form a one-dimensional data vector.

    This function loads a `pyuvdata`-compatible file containing visibilities
    for a set of baselines (Nbls), times (Ntimes), and frequencies (Nfreqs)
    and forms a one-dimensional data vector with shape
    (2*Nbls * Ntimes * Nfreqs,) and an accompanying instrument model of
    sampled (u, v, w) coordinates and a redundancy model (when averaging
    redundant baselines). The factor of 2 in the data vector shape comes from
    the fact that the input visibility vector must be Hermitian, so we copy
    all baselines at (u, v) to (-u, -v) and conjugate the data accordingly.
    This is captured in the instrument model as the instrument model also
    contains all (u, v) and conjugated (-u, -v).

    Optionally, if `calc_noise` is True, a noise vector is also produced.
    The noise is estimated by differencing visibilities at even and odd time
    steps per baseline and frequency.

    Notes
    -----
    Compared to the original BayesEoR function, phasor-vector generation is
    intentionally not implemented in BayesEoR-JAX.

    Parameters
    ----------
    filepath : Path or str
        Path to pyuvdata-compatible file containing visibilities.
    antenna_selection : str, optional
        Antenna downselect string. This determines what baselines to keep in
        the data vector. Please see `pyuvdata.UVData.select` for more details.
        Defaults to 'cross' (cross-correlation baselines only).
    bl_cutoff : astropy.Quantity or float, optional
        Baseline length cutoff in meters if not a Quantity. Defaults to None
        (keep all baselines).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Defaults
        to None (keep all frequencies).
    freq_min : astropy.Quantity or float, optional
        Minimum frequency to keep in the data vector in Hertz if not a
        Quantity. All frequencies greater than or equal to `freq_min` will be
        kept, unless `n_freqs_select` is specified. Defaults to None (keep all
        frequencies).
    freq_center : astropy.Quantity or float, optional
        Central frequency, in Hertz if not a Quantity, around which
        `n_freqs_select` frequencies will be kept in the data vector.
        `n_freqs_select` must also be
        passed, otherwise an error is raised. Defaults to None (keep all
        frequencies).
    n_freqs_select : int, optional
        Number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `freq_min`, or around `freq_center`. Defaults
        to None (keep all frequencies).
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_min : astropy.Time or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        Defaults to None (keep all times).
    jd_center : astropy.Time or float, optional
        Central time, as a Julian date if not a Time, around which
        `n_times_select` times will be kept in the data vector.
        `n_times_select` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    n_times_select : int, optional
        Number of times to keep starting from `jd_idx_min`, the time
        corresponding to `jd_min` or around `jd_center`. Defaults to None (keep
        all times).
    form_pi : bool, optional
        Form pseudo-Stokes I visibilities. Otherwise, use the polarization
        specified by `pol`. Defaults to True.
    pi_norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        via ``pI = N * (XX + YY)``. Defaults to 1.0.
    pol : str, optional
        Case-insensitive polarization string. Used only if `form_pi` is False.
        Defaults to 'xx'.
    redundant_avg : bool, optional
        Redundantly average the data. Defaults to False.
    uniform_redundancy : bool, optional
        Force the redundancy model to be uniform. Defaults to False.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times per baseline and frequency. Defaults to False.
    return_uvd : bool, optional
        Return the preprocessed UVData object. Defaults to False.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. If `calc_noise` is True,
        also save the noise vector. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `save_dir`. Defaults to False.
    save_dir : Path or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        './'.
    clobber : bool, optional
        Clobber files on disk if they exist. Defaults to False.
    Returns
    -------
    vis : numpy.ndarray
        Visibility vector with shape (2*Nbls*Nfreqs*Ntimes,).
    antpairs : list of tuple
        List of antenna pair tuples corresponding to the (u, v, w) coordinates
        in `uvws`, in identical order, with length (2*Nbls).
    uvws : numpy.ndarray
        Sampled (u, v, w) coordinates with shape (Ntimes, 2*Nbls, 3). The
        ordering of the Nbls axis matches the ordering of the baselines in
        `vis`.
    redundancy : numpy.ndarray
        Redundancy model containing the number of baselines averaged within
        a redundant baseline group.
    noise : numpy.ndarray or None
        Estimated noise vector with shape (2*Nbls*Nfreqs*Ntimes,). Returned
        as None when `calc_noise` is False.
    uvd : pyuvdata.UVData or None
        Preprocessed UVData object. Returned only if `return_uvd` is True.
    """
    # Validate the input path before any IO.
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")

    # Prepare output file paths up-front when persistence is requested.
    if save_vis or save_model:
        if save_dir is None:
            raise ValueError("save_dir cannot be none if save_vis or save_model is True")
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        if save_vis:
            vis_path = save_dir / "vis_vector.npy"
            if calc_noise:
                noise_path = save_dir / "noise_vector.npy"
        if save_model:
            uvw_path = save_dir / "uvw_model.npy"
            ants_path = save_dir / "antpairs.npy"
            red_path = save_dir / "redundancy_model.npy"

    # First metadata pass: read lightweight metadata to plan downselect windows.
    uvd = UVData()
    LOGGER.info("Reading data from: %s", filepath)
    uvd.read(filepath, read_data=False)
    uvd.select(ant_str=antenna_selection)

    if uvd.vis_units not in ["Jy", "K str"]:
        raise ValueError(
            "This code requires calibrated input visibilities in units of "
            "Janskys or kelvin steradians. The input UVData object has "
            f"incompatible units of {uvd.vis_units}."
        )

    # Backwards compatibility with old pyuvdata frequency axis layout.
    trim_nspws_ax = len(uvd.freq_array.shape) > 1

    # Baseline-length filtering is performed before full data read.
    if bl_cutoff is not None:
        if not isinstance(bl_cutoff, Quantity):
            bl_cutoff = Quantity(bl_cutoff, unit="m")
        LOGGER.info("Baseline downselect: keeping |b| <= %s m", bl_cutoff)
        bl_lengths = Quantity(
            np.sqrt(np.sum(uvd.uvw_array[:, :2] ** 2, axis=1)),
            unit="m",
        )
        blt_inds = np.where(bl_lengths <= bl_cutoff)[0]
        uvd.select(blt_inds=blt_inds)
    bls = uvd.get_antpairs()

    # Resolve frequency selection (index-based, absolute lower bound, or centered window).
    if np.any([param is not None for param in [freq_min, freq_idx_min, freq_center]]):
        # Convert the file frequency axis to a Quantity so all comparisons are unit-safe.
        freqs = Quantity(uvd.freq_array, unit="Hz")
        # Older pyuvdata versions store freq as (Nspws, Nfreq); select the only spw.
        if trim_nspws_ax:
            freqs = freqs[0]
        if freq_center is not None:
            # Centered-window mode: choose channels around a target center frequency.
            if n_freqs_select is None:
                raise ValueError("Must pass n_freqs_select with freq_center")
            if not isinstance(freq_center, Quantity):
                freq_center = Quantity(freq_center, unit="Hz")
            # Hard bounds check before finding the nearest channel index.
            if freq_center.to("Hz") < freqs[0].to("Hz"):
                raise ValueError(
                    f"freq_center ({freq_center.to('MHz')}) < minimum frequency "
                    f"in data ({freqs[0].to('MHz')})"
                )
            if freq_center.to("Hz") > freqs[-1].to("Hz"):
                raise ValueError(
                    f"freq_center ({freq_center.to('MHz')}) > maximum frequency "
                    f"in data ({freqs[-1].to('MHz')})"
                )
            freq_idx_center = np.argmin(
                np.abs(freqs.to("Hz") - freq_center.to("Hz"))
            )
            # Enforce that the lower half-window exists in the data.
            if freq_idx_center - (n_freqs_select // 2) < 0:
                raise ValueError(
                    f"Invalid combination of freq_center ({freq_center}) "
                    f"and n_freqs_select ({n_freqs_select}). There are fewer than "
                    "n_freqs_select//2 "
                    "frequencies less than or equal to freq_center."
                )
            # Warn (not error) when the upper half-window is truncated by the data edge.
            if freq_idx_center + n_freqs_select // 2 > freqs.size:
                warnings.warn(
                    "There are less than n_freqs_select//2 frequencies greater than or "
                    "equal to freq_center. This combination of freq_center "
                    f"({freq_center}) and n_freqs_select ({n_freqs_select}) will "
                    "result in an "
                    "asymmetric set of frequencies with a central value < "
                    "freq_center."
                )
            # Convert center+width specification into the unified lower-index form.
            freq_idx_min = freq_idx_center - (n_freqs_select // 2)
            freq_min = freqs[freq_idx_min]
        elif freq_min is not None:
            # Lower-bound mode: start at first channel >= freq_min.
            if not isinstance(freq_min, Quantity):
                freq_min = Quantity(freq_min, unit="Hz")
            if freq_min.to("Hz") < freqs[0].to("Hz"):
                warnings.warn(
                    f"freq_min ({freq_min.to('MHz')}) < minimum frequency in "
                    f"data ({freqs[0].to('MHz')}). All frequencies will be kept "
                    "if n_freqs_select is not specified."
                )
            if freq_min.to("Hz") > freqs[-1].to("Hz"):
                raise ValueError(
                    f"freq_min ({freq_min.to('MHz')}) > maximum frequency "
                    f"in data ({freqs[-1].to('MHz')})"
                )
            # First selected channel index from threshold crossing.
            freq_idx_min = np.where(freqs.to("Hz") >= freq_min.to("Hz"))[0][0]
        else:
            # Raw-index mode: user directly specifies the first kept channel.
            if freq_idx_min < 0:
                raise ValueError("freq_idx_min must be positive")
            if freq_idx_min >= freqs.size:
                raise ValueError("freq_idx_min cannot exceed number of channels")
            freq_min = freqs[freq_idx_min]

        # If width not provided, keep all channels from freq_idx_min to the end.
        if n_freqs_select is None:
            n_freqs_select = freqs.size - freq_idx_min
        elif n_freqs_select <= 0:
            raise ValueError("n_freqs_select must be positive")

        # Guard against overshooting available channels after start index.
        if n_freqs_select > freqs[freq_idx_min: freq_idx_min + n_freqs_select].size:
            warnings.warn(
                "This combination of frequency bounds and n_freqs_select yields "
                "fewer than n_freqs_select channels in the selected data."
            )
        # Final selected frequency array passed to the second read() call.
        freqs = freqs[freq_idx_min: freq_idx_min + n_freqs_select].to("Hz").value
    else:
        # None means "do not apply frequency filtering" to pyuvdata read().
        freqs = None

    # Resolve time selection (index-based, absolute lower bound, or centered window).
    if np.any([param is not None for param in [jd_min, jd_idx_min, jd_center]]):
        # Build a unique time axis first; UVData stores times per baseline-time row.
        jds = Time(np.unique(uvd.time_array), format="jd")
        if jd_center is not None:
            # Centered-window mode: choose integrations around jd_center.
            if n_times_select is None:
                raise ValueError("Must pass n_times_select with jd_center")
            if not isinstance(jd_center, Time):
                jd_center = Time(jd_center, format="jd")
            # Hard bounds check before nearest-index lookup.
            if jd_center.jd < jds[0].jd:
                raise ValueError(
                    f"jd_center ({jd_center}) < minimum time in data ({jds[0]})"
                )
            if jd_center.jd > jds[-1].jd:
                raise ValueError(
                    f"jd_center ({jd_center}) > maximum time in data ({jds[-1]})"
                )
            jd_idx_center = np.argmin(np.abs(jds.jd - jd_center.jd))
            # Enforce lower half-window existence.
            if jd_idx_center - (n_times_select // 2) < 0:
                raise ValueError(
                    f"Invalid combination of jd_center ({jd_center}) and "
                    f"n_times_select ({n_times_select}). There are fewer than "
                    "n_times_select//2 "
                    "times less than or equal to jd_center."
                )
            # Warn if upper half-window is truncated by dataset boundary.
            if jd_idx_center + n_times_select // 2 > jds.size:
                warnings.warn(
                    "There are less than n_times_select//2 times greater than or equal "
                    "to jd_center. This combination will produce an asymmetric "
                    "time selection with a central value < jd_center."
                )
            # Convert center+width specification into unified lower-index form.
            jd_idx_min = jd_idx_center - (n_times_select // 2)
            jd_min = jds[jd_idx_min]
        elif jd_min is not None:
            # Lower-bound mode: start at first integration >= jd_min.
            if not isinstance(jd_min, Time):
                jd_min = Time(jd_min, format="jd")
            if jd_min.jd < jds[0].jd:
                warnings.warn(
                    f"jd_min ({jd_min}) < minimum time in data ({jds[0]}). "
                    "All times will be kept if n_times_select is not specified."
                )
            if jd_min.jd > jds[-1].jd:
                raise ValueError(
                    f"jd_min ({jd_min}) > maximum time in data ({jds[-1]})"
                )
            # First selected time index from threshold crossing.
            jd_idx_min = np.where(jds.jd >= jd_min.jd)[0][0]
        else:
            # Raw-index mode: user directly specifies the first kept time.
            if jd_idx_min < 0:
                raise ValueError("jd_idx_min must be positive")
            if jd_idx_min >= jds.size:
                raise ValueError("jd_idx_min cannot exceed number of times")
            jd_min = jds[jd_idx_min]

        # If width not provided, keep all times from jd_idx_min onward.
        if n_times_select is None:
            n_times_select = jds.size - jd_idx_min
        elif n_times_select <= 0:
            raise ValueError("n_times_select must be positive")

        # Guard against overshooting available times after start index.
        if n_times_select > jds[jd_idx_min: jd_idx_min + n_times_select].size:
            warnings.warn(
                "This combination of time bounds and n_times_select yields fewer "
                "than n_times_select samples in the selected data."
            )
        # Final selected time array passed to the second read() call.
        jds = jds[jd_idx_min: jd_idx_min + n_times_select].jd
    else:
        # None means "do not apply time filtering" to pyuvdata read().
        jds = None

    # Full data read with computed baseline/time/frequency selections applied.
    uvd = UVData()
    uvd.read(filepath, times=jds, frequencies=freqs, bls=bls)

    # Remove fully flagged baselines so they do not pollute the data/model vectors.
    if np.sum(uvd.flag_array) > 0:
        antpairs = uvd.get_antpairs()
        good_antpairs = []
        for antpair in antpairs:
            flags = uvd.get_flags(antpair)
            if np.sum(flags) < flags.size:
                good_antpairs.append(antpair)
        uvd.select(bls=good_antpairs)

    # Keep the legacy assumption: vectorization starts from unprojected visibilities.
    if np.any(~uvd._check_for_cat_type("unprojected")):
        uvd.unproject_phase()

    # Reduce to one analysis polarization: pseudo-I or selected linear pol.
    if form_pi:
        uvd = form_pI_vis(uvd, norm=pi_norm)
        pol = "pI"
    else:
        if polstr2num(pol) not in uvd.polarization_array:
            raise ValueError(f"{pol} polarization not present in visibilities")
        uvd.select(polarizations=[pol])

    # Convert to mK sr so the output convention matches the old analysis stack.
    if uvd.vis_units == "K str":
        uvd.data_array *= 1e3
    else:
        if trim_nspws_ax:
            uvd.data_array[:, 0, :, 0] = jy_to_ksr(
                uvd.data_array[:, 0, :, 0],
                uvd.freq_array[0],
                mK=True,
            )
        else:
            uvd.data_array[..., 0] = jy_to_ksr(
                uvd.data_array[..., 0],
                uvd.freq_array,
                mK=True,
            )

    # Optionally average redundant baselines and track group sizes as weights.
    red_grps = None
    if redundant_avg:
        if not uniform_redundancy:
            red_grps = uvd.get_redundancies()[0]
        uvd.compress_by_redundancy(method="average")

    # Enforce deterministic BLT ordering before flattening.
    if uvd.blt_order != ("time", "baseline"):
        uvd.reorder_blts(order="time")

    antpairs = sorted(uvd.get_antpairs(), key=lambda element: (element[0], element[1]))

    # Build instrument-model arrays with explicit conjugate-baseline copies:
    # first half contains (u, v, w), second half contains (-u, -v, -w).
    n_bls = uvd.Nbls
    n_bls_vec = 2 * n_bls
    uvws = np.zeros((uvd.Ntimes, n_bls_vec, 3))
    redundancy = np.ones((uvd.Ntimes, n_bls_vec, 1))
    for i_bl, antpair in enumerate(antpairs):
        uvw_ind = uvd.antpair2ind(antpair)
        if isinstance(uvw_ind, slice):
            uvw_ind = uvw_ind.start
        else:
            uvw_ind = uvw_ind[0]
        uvw = uvd.uvw_array[uvw_ind]
        uvws[:, i_bl] = uvw
        uvws[:, n_bls + i_bl] = -1 * uvw
        if redundant_avg and not uniform_redundancy and red_grps is not None:
            bl = uvd.antnums_to_baseline(*antpair)
            for bls_group in red_grps:
                if bl in bls_group:
                    redundancy[:, i_bl] = len(bls_group)
                    redundancy[:, n_bls + i_bl] = len(bls_group)

    # Build final Hermitian-expanded visibility/noise vectors in matching order.
    vis, noise = uvd_to_vector(
        uvd=uvd,
        antpairs=antpairs,
        pol=pol,
        calc_noise=calc_noise,
    )
    # Mirror antpairs to match positive-uv then negative-uv vector layout.
    antpairs += [antpair[::-1] for antpair in antpairs]

    # Save vectors/model products with argument metadata for reproducibility.
    if save_vis or save_model:
        args = dict(
            filepath=filepath.as_posix(),
            antenna_selection=antenna_selection,
            bl_cutoff=bl_cutoff,
            freq_min=freq_min,
            freq_idx_min=freq_idx_min,
            freq_center=freq_center,
            n_freqs_select=n_freqs_select,
            jd_min=jd_min,
            jd_idx_min=jd_idx_min,
            jd_center=jd_center,
            n_times_select=n_times_select,
            form_pi=form_pi,
            pi_norm=pi_norm,
            pol=pol,
            redundant_avg=redundant_avg,
            uniform_redundancy=uniform_redundancy,
            calc_noise=calc_noise,
        )
        extra = dict(pyuvdata_version=pyuvdata_version)

    # Persist requested outputs.
    if save_vis:
        save_numpy_dict(vis_path, vis, args, extra=extra, clobber=clobber)
        if calc_noise:
            save_numpy_dict(noise_path, noise, args, extra=extra, clobber=clobber)
    if save_model:
        save_numpy_dict(ants_path, antpairs, args, extra=extra, clobber=clobber)
        save_numpy_dict(uvw_path, uvws, args, extra=extra, clobber=clobber)
        save_numpy_dict(red_path, redundancy, args, extra=extra, clobber=clobber)

    # Return signature mirrors old pipeline shape/order (without phasor).
    return_vals = vis, antpairs, uvws, redundancy, noise
    if return_uvd:
        return_vals += (uvd,)
    return return_vals


def uvd_to_vector(
    uvd,
    antpairs,
    pol="xx",
    calc_noise=False,
):
    """
    Form a BayesEoR visibility vector from a single-polarization UVData object.

    The output vector is explicitly Hermitian by appending conjugated baseline
    copies, resulting in shape ``(2*Nbls*Nfreqs*Ntimes,)``.

    If `calc_noise` is True, an empirical noise vector is built by differencing
    adjacent even/odd time samples per baseline and frequency, then expanded
    with the same Hermitian ordering as the visibility vector.
    """
    if uvd.Npols > 1:
        raise ValueError(
            "Input UVData object has visibilities for more than one "
            "polarization. Please select one polarization before "
            "calling uvd_to_vector."
        )

    # pyuvdata changed frequency axis conventions across versions; support both.
    trim_nspws_ax = len(uvd.freq_array.shape) > 1

    if calc_noise:
        LOGGER.info("Calculating noise estimate from visibilities")
        uvd_noise = uvd.copy()
        for antpair in antpairs:
            # Old BayesEoR estimator: even-odd time differencing per baseline/frequency.
            vis = uvd_noise.get_data(antpair + (pol,), force_copy=True)
            even_times = vis[::2]
            odd_times = np.zeros_like(even_times)
            odd_times[:-1] = vis[1::2]
            if uvd.Ntimes % 2 == 1:
                odd_times[-1] = vis[-2]
            eo_diff = even_times - odd_times
            noise = np.zeros_like(vis)
            noise[::2] = eo_diff
            if uvd.Ntimes % 2 == 1:
                noise[1::2] = eo_diff[:-1]
            else:
                noise[1::2] = eo_diff
            blt_inds = uvd_noise.antpair2ind(*antpair)
            if trim_nspws_ax:
                uvd_noise.data_array[blt_inds, 0, :, 0] = noise
            else:
                uvd_noise.data_array[blt_inds, :, 0] = noise

    # Allocate Hermitian-expanded output vectors.
    vis_vec = np.zeros(2 * uvd.Nblts * uvd.Nfreqs, dtype=complex)
    if calc_noise:
        noise_vec = np.zeros_like(vis_vec)

    n_bls = uvd.Nbls
    n_bls_vec = 2 * n_bls
    for i_bl, antpair in enumerate(antpairs):
        antpairpol = antpair + (pol,)
        vis = uvd.get_data(antpairpol)
        # Positive-uv half.
        vis_vec[i_bl::n_bls_vec] = vis.flatten()
        # Negative-uv conjugate half.
        vis_vec[n_bls + i_bl::n_bls_vec] = vis.flatten().conj()
        if calc_noise:
            noise = uvd_noise.get_data(antpairpol)
            noise_vec[i_bl::n_bls_vec] = noise.flatten()
            noise_vec[n_bls + i_bl::n_bls_vec] = noise.flatten().conj()

    if calc_noise:
        return vis_vec, noise_vec

    return vis_vec, None


def form_pI_vis(uvd, norm=1.0):
    """
    Form pseudo-Stokes I visibilities by summing XX and YY visibilities.

    Parameters
    ----------
    uvd : pyuvdata.UVData
        UVData object containing XX and YY polarization visibilities.
    norm : float, optional
        Normalization, ``N``, used in ``pI = N * (XX + YY)``.
    """
    assert isinstance(uvd, UVData), "uvd must be a pyuvdata.UVData object"

    # If pyuvdata does not already carry pI, build it explicitly from XX and YY.
    if polstr2num("pI") not in uvd.polarization_array:
        xx_pol_num = polstr2num("xx")
        yy_pol_num = polstr2num("yy")
        xpol_ind = np.where(uvd.polarization_array == xx_pol_num)[0]
        ypol_ind = np.where(uvd.polarization_array == yy_pol_num)[0]
        uvd.data_array[..., xpol_ind] += uvd.data_array[..., ypol_ind]
        uvd.data_array *= norm
        # Collapse to a single polarization axis and relabel it as pI.
        uvd.select(polarizations=["xx"])
        uvd.polarization_array[0] = polstr2num("pI")

    return uvd


def jy_to_ksr(data, freqs, mK=False):
    """
    Convert visibilities from Janskys to kelvin-steradians.

    Parameters
    ----------
    data : numpy.ndarray
        Visibility array with frequency on the last axis.
    freqs : astropy.Quantity or numpy.ndarray
        Frequencies in Hertz.
    mK : bool, optional
        Convert to ``mK sr`` if True, otherwise ``K sr``.
    """
    # Normalize frequencies to an astropy Quantity for unit-safe conversion.
    if not isinstance(freqs, Quantity):
        freqs = Quantity(freqs, units.Hz)

    # Convert Jy -> K using brightness-temperature equivalency with 1 sr beam area.
    equiv = units.brightness_temperature(freqs, beam_area=1 * units.sr)
    if mK:
        conv_factor = (1 * units.Jy).to(units.mK, equivalencies=equiv)
    else:
        conv_factor = (1 * units.Jy).to(units.K, equivalencies=equiv)
    conv_factor *= units.sr / units.Jy
    return data * conv_factor[np.newaxis, :].value
