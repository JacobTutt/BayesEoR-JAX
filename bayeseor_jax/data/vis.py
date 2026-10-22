"""
Load calibrated interferometric measurements into BayesEoR visibility order.

``get_vis_data`` accepts the following input formats:

- ``.uvh5``: UVH5 files read through pyuvdata.
- ``.uvfits``: UVFITS files read through pyuvdata.
- ``.ms``: Measurement Set directories, subject to pyuvdata reader availability.
- ``.npy``: preprocessed visibility arrays, either plain arrays or BayesEoR
  dictionary payloads containing a ``data`` entry. These require a separate
  instrument-model directory and explicit frequency/time metadata.

Raw inputs are selected by baseline, frequency and time, unprojected, reduced
to one polarisation, and converted from Jy or K sr to mK sr. Optional redundant
averaging precedes vectorisation. Each time/frequency block contains the selected
baselines followed by their complex conjugates, with matching reversed baseline
coordinates; these copies are not additional independent measurements.

Preprocessed arrays must already have these units and conventions. Supplied or
estimated noise leaves the data unchanged; otherwise synthetic noise is added.
The module prepares data and metadata, not a sky model or forward operator.
NumPy dictionary files use pickle and must come from a trusted source.
"""

import warnings
from pathlib import Path
import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units
from astropy.time import Time
from astropy.units import Quantity
from pyuvdata import UVData
from pyuvdata import __version__ as pyuvdata_version
from pyuvdata.utils import polstr2num

from ..model.instrument.baselines import InstrumentModel
from .noise import generate_gaussian_noise
from ..io import load_numpy_dict, save_numpy_dict

LOGGER = logging.getLogger(__name__)

def get_vis_data(
    *,
    data_path: Path | str,
    inst_model: Path | str | None = None,
    noise_data_path: Path | str | None = None,
    sigma: float | None = None,
    noise_seed: int = 742123,
    calc_noise: bool = False,
    # Raw-uvdata preprocessing controls.
    antenna_selection: str = "cross",
    bl_cutoff: float | None = None,
    freq_idx_min: int | None = None,
    freq_min: float | None = None,
    freq_center: float | None = None,
    n_freqs_select: int | None = None,
    jd_idx_min: int | None = None,
    jd_min: float | None = None,
    jd_center: float | None = None,
    n_times_select: int | None = None,
    form_pi: bool = True,
    pi_norm: float = 1.0,
    pol: str = "xx",
    redundant_avg: bool = False,
    uniform_redundancy: bool = False,
    # For preprocessed vectors.
    n_freq: int | None = None,
    f0_mhz: float | None = None,
    df_mhz: float | None = None,
    n_t: int | None = None,
    d_t: float | None = None,
    # Save hooks for raw preprocessing.
    save_vis: bool = False,
    save_model: bool = False,
    save_dir: Path | str = "./",
    clobber: bool = False,
) -> dict[str, Any]:
    """
    Load a visibility vector, its noise realisation and aligned observation metadata.

    Parameters
    ----------
    data_path : pathlib.Path or str
        A calibrated ``.uvh5`` or ``.uvfits`` file, a ``.ms`` directory, or a
        preprocessed ``.npy`` array. These lowercase suffixes select the reader.
        Raw inputs must be in Jy or K sr; NumPy inputs must already be in mK sr,
        flattened in time/frequency/baseline order with conjugate copies.
    inst_model : pathlib.Path or str, optional
        Required for NumPy inputs. Directory containing ``uvw_model.npy`` and
        ``redundancy_model.npy`` with optional ``antpairs.npy``, or a combined
        ``instrument_model.npy`` dictionary. See ``InstrumentModel``.
    noise_data_path : pathlib.Path or str, optional
        Trusted NumPy noise vector in the same units and order as the data.
        Takes precedence over estimated noise. It is returned, not added:
        the input visibilities are assumed to contain noise already.
    sigma : float, optional
        Complex noise RMS in mK sr before redundancy scaling. Required when
        neither supplied nor estimated noise is available; the input is then
        treated as noiseless and a synthetic realisation is added.
    noise_seed : int, optional
        Seed passed to ``generate_gaussian_noise``; defaults to 742123.
    calc_noise : bool, optional
        Request the legacy even/odd time-difference noise estimate for raw
        inputs. Does not estimate noise from preprocessed NumPy inputs.
    antenna_selection : str, optional
        pyuvdata antenna-selection string for raw inputs; default ``'cross'``.
    bl_cutoff : float or astropy.units.Quantity, optional
        Raw-input cutoff on sqrt(u**2 + v**2), in metres if unitless.
    freq_idx_min : int, optional
        First raw frequency-channel index, starting from zero.
    freq_min, freq_center : float or astropy.units.Quantity, optional
        Raw-input frequency lower bound or window centre, in Hz if unitless.
    n_freqs_select : int, optional
        Raw channel-window width. Supply a starting index, lower bound or
        centre to activate selection; a centre requires this width.
    jd_idx_min : int, optional
        First raw integration index, starting from zero.
    jd_min, jd_center : float, optional
        Time-window lower bound or centre as Julian date. For NumPy inputs,
        one is required to construct the time axis; ``jd_min`` takes priority.
        Raw preprocessing also accepts astropy Time values.
    n_times_select : int, optional
        Raw time-window width, paired with an index, lower bound or centre.
    form_pi : bool, optional
        Form pseudo-Stokes I from raw XX and YY, or retain existing pI.
        Defaults to True; otherwise select ``pol``.
    pi_norm : float, optional
        Scale in ``pI = pi_norm * (XX + YY)``; defaults to 1, not 1/2.
    pol : str, optional
        Raw polarisation selected when ``form_pi=False``; defaults to ``'xx'``.
    redundant_avg : bool, optional
        Average redundant raw baselines; defaults to False.
    uniform_redundancy : bool, optional
        Use unit redundancy weights even after averaging; defaults to False.
    n_freq, f0_mhz, df_mhz : int, float, float, optional
        Required NumPy frequency-axis length, first frequency and spacing in
        MHz. For a single raw channel, ``df_mhz`` supplies its spacing.
    n_t, d_t : int, float, optional
        Required NumPy time-axis length and cadence in seconds. ``d_t`` also
        supplies raw single-integration cadence (otherwise returned as zero).
    save_vis, save_model : bool, optional
        Save raw preprocessing products: visibility/noise vectors and/or
        baseline, redundancy and antenna-pair arrays. Defaults to False.
        These hooks do not save subsequently generated synthetic noise.
    save_dir : pathlib.Path or str, optional
        Directory for requested raw preprocessing products; defaults to './'.
    clobber : bool, optional
        Allow replacement of those saved products; defaults to False.

    Returns
    -------
    dict
        ``vis`` is the loaded complex128 vector in mK sr; ``vis_noisy`` is
        either that vector or ``vis + noise``. ``noise`` is the supplied,
        estimated or generated vector in matching order. ``bl_conj_pairs_map``
        records synthetic-noise baseline pairing, otherwise it is empty.
        ``uvws`` contains metre coordinates, shape (n_t, n_baseline, 3), and
        ``redundancy`` contains aligned baseline weights. ``antpairs`` holds
        baseline labels if available. ``freqs_hz`` and ``jds`` are the frequency
        and Julian-date axes; ``df_hz`` and ``dt`` are their spacings in Hz and
        seconds. ``uvd`` is the processed raw UVData object, or None for NumPy.

    Notes
    -----
    Raw selection and conversion are delegated to ``preprocess_uvdata``.
    NumPy inputs bypass these steps and must already match the model's row
    ordering and Hermitian augmentation; stored metadata are not used to infer
    the axes. NumPy dictionary inputs are unpickled: load trusted files only.
    This loader does not turn partial flags into per-visibility noise weights.
    """
    # Normalize and validate the input data path once at the boundary.
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    vis: jax.Array
    noise_est: jax.Array | None
    antpairs: Any | None
    uvd = None

    if data_path.suffix in {".uvh5", ".uvfits", ".ms"}:
        # Raw-file mode:
        # - select frequencies/times/baselines in preprocess_uvdata
        # - build Hermitian-expanded data vector + instrument arrays
        # - optionally estimate empirical noise from even/odd differencing
        out = preprocess_uvdata(
            filepath=data_path,
            antenna_selection=antenna_selection,
            bl_cutoff=bl_cutoff,
            freq_idx_min=freq_idx_min,
            freq_min=freq_min,
            freq_center=freq_center,
            n_freqs_select=n_freqs_select,
            jd_idx_min=jd_idx_min,
            jd_min=jd_min,
            jd_center=jd_center,
            n_times_select=n_times_select,
            form_pi=form_pi,
            pi_norm=pi_norm,
            pol=pol,
            redundant_avg=redundant_avg,
            uniform_redundancy=uniform_redundancy,
            calc_noise=calc_noise,
            return_uvd=True,
            save_vis=save_vis,
            save_model=save_model,
            save_dir=save_dir,
            clobber=clobber,
        )
        vis_np, antpairs, uvws_np, red_np, noise_np, uvd = out
        vis = jnp.asarray(vis_np, dtype=jnp.complex128)
        uvws = jnp.asarray(uvws_np, dtype=jnp.float64)
        redundancy = jnp.asarray(red_np)
        noise_est = None if noise_np is None else jnp.asarray(noise_np, dtype=jnp.complex128)

        # Extract frequency/time metadata directly from the selected UVData.
        freqs_hz = uvd.freq_array
        if freqs_hz.ndim > 1:
            freqs_hz = freqs_hz[0]
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        jds = np.unique(uvd.time_array)
        if jds.size > 1:
            dt = float((Time(jds[1], format="jd") - Time(jds[0], format="jd")).to("s").value)
        else:
            dt = float(d_t) if d_t is not None else 0.0
    elif data_path.suffix == ".npy":
        # Preprocessed-vector mode:
        # - load visibilities from disk
        # - load instrument model from an instrument directory
        # - construct frequency/time axes from parser inputs
        if inst_model is None:
            raise ValueError("inst_model is required when loading a preprocessed .npy vector")
        if n_freq is None or f0_mhz is None or df_mhz is None or n_t is None or d_t is None:
            raise ValueError(
                "n_freq, f0_mhz, df_mhz, n_t, and d_t are required for preprocessed .npy inputs"
            )

        vis = jnp.asarray(load_numpy_dict(data_path, as_jax=True), dtype=jnp.complex128).reshape(-1)
        inst = InstrumentModel(inst_model)
        uvws = jnp.asarray(inst.uvw_array_m, dtype=jnp.float64)
        redundancy = jnp.asarray(inst.bl_red_array)
        antpairs = inst.antpairs
        noise_est = None

        # In this mode, `n_freq/f0/df` define the analysis band explicitly.
        freqs_hz = (float(f0_mhz) + float(df_mhz) * np.arange(int(n_freq), dtype=float)) * 1e6
        if jd_min is not None:
            # Old-pipeline compatible mode: construct times from lower bound.
            jds = Time(float(jd_min), format="jd") + np.arange(int(n_t), dtype=float) * (
                float(d_t) / 86400.0
            )
            jds = jds.jd
        elif jd_center is not None:
            # Centered-window mode around jd_center.
            jds = Time(float(jd_center), format="jd") + np.arange(
                -(int(n_t) // 2), int(n_t) // 2 + int(n_t) % 2
            ) * (float(d_t) / 86400.0)
            jds = jds.jd
        else:
            raise ValueError(
                "One of jd_min or jd_center is required for preprocessed .npy inputs"
            )
        dt = float(d_t)
    else:
        raise ValueError(f"Unsupported data_path suffix '{data_path.suffix}'")

    # Optional external noise vector.
    if noise_data_path is not None:
        # If a noise vector is supplied, treat input visibilities as already noisy.
        noise = jnp.asarray(load_numpy_dict(noise_data_path, as_jax=True), dtype=jnp.complex128).reshape(-1)
        vis_noisy = vis
        bl_conj_pairs_map: dict[int, int] = {}
    elif noise_est is not None:
        # Reuse empirical noise estimated during raw UVData preprocessing.
        noise = noise_est
        vis_noisy = vis
        bl_conj_pairs_map = {}
    else:
        # Otherwise inject synthetic Gaussian noise using the instrument redundancy model.
        if sigma is None:
            raise ValueError("sigma is required when no noise vector is supplied/estimated")
        vis_noisy, noise, bl_conj_pairs_map = generate_gaussian_noise(
            sigma=float(sigma),
            s=vis,
            n_freq=int(freqs_hz.size),
            n_t=int(np.unique(jds).size),
            uvw_array_m=uvws,
            bl_red_array=redundancy,
            random_seed=noise_seed,
        )

    df_hz = float(freqs_hz[1] - freqs_hz[0]) if freqs_hz.size > 1 else float(df_mhz) * 1e6

    return {
        "vis": vis,
        "vis_noisy": vis_noisy,
        "noise": noise,
        "bl_conj_pairs_map": bl_conj_pairs_map,
        "uvws": uvws,
        "redundancy": redundancy,
        "antpairs": antpairs,
        "freqs_hz": jnp.asarray(freqs_hz, dtype=jnp.float64),
        "df_hz": float(df_hz),
        "jds": jnp.asarray(jds, dtype=jnp.float64),
        "dt": float(dt),
        "uvd": uvd,
    }



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
    baseline coordinates in metres and a redundancy model (when averaging
    redundant baselines). The factor of two comes from appending conjugate
    visibilities at reversed baselines (u, v, w) -> (-u, -v, -w). These are
    redundant copies of the measurements, not independent observations.
    Time is the slowest axis, frequency next, and baseline fastest.

    Optionally, if `calc_noise` is True, a noise vector is also produced.
    The noise is estimated by differencing visibilities at even and odd time
    steps per baseline and frequency.

    Parameters
    ----------
    filepath : Path or str
        Path read by pyuvdata (normally UVH5, UVFITS or a Measurement Set).
        Unlike ``get_vis_data``, this lower-level function delegates format
        detection directly to UVData.read. Input units must be Jy or K sr.
    antenna_selection : str, optional
        Antenna downselect string. This determines what baselines to keep in
        the data vector. Please see `pyuvdata.UVData.select` for more details.
        Defaults to 'cross' (cross-correlation baselines only).
    bl_cutoff : astropy.Quantity or float, optional
        Cutoff on sqrt(u**2 + v**2), in metres if not a Quantity. Defaults to None
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
        Complex visibility vector in mK sr, shape (2*Nbls*Nfreqs*Ntimes,).
    antpairs : list of tuple
        List of antenna pair tuples corresponding to the (u, v, w) coordinates
        in `uvws`, in identical order, with length (2*Nbls).
    uvws : numpy.ndarray
        Baseline coordinates in metres, shape (Ntimes, 2*Nbls, 3). The
        ordering of the Nbls axis matches the ordering of the baselines in
        `vis`.
    redundancy : numpy.ndarray
        Redundancy model containing the number of baselines averaged within
        a redundant baseline group, shape (Ntimes, 2*Nbls, 1).
    noise : numpy.ndarray or None
        Estimated noise vector with shape (2*Nbls*Nfreqs*Ntimes,). Returned
        as None when `calc_noise` is False.
    uvd : pyuvdata.UVData or None
        Preprocessed UVData object. Returned only if `return_uvd` is True.

    Notes
    -----
    This path assumes a common set of baselines at every selected time and
    unprojected drift-scan geometry. Fully flagged baselines are removed, but
    individual flagged samples are not masked or assigned zero weight here.
    The returned UVData data array has been converted to mK sr; its original
    ``vis_units`` label is not updated. Use the returned vector for inference.
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

    Parameters
    ----------
    uvd : pyuvdata.UVData
        Single-polarisation data with the same ordered time/frequency grid
        for every baseline. Units are preserved; no conversion is done here.
    antpairs : sequence of tuple of int
        Ordered original antenna pairs, without appended reversed copies.
    pol : str, optional
        Selected polarisation label; defaults to ``'xx'``.
    calc_noise : bool, optional
        Apply the legacy even/odd time-difference estimator; defaults to False.
        Requires multiple integrations and does not rescale by sqrt(2).

    Returns
    -------
    vis_vec : numpy.ndarray
        Complex vector in time/frequency/baseline order. Within every
        time/frequency block, the original baselines precede their conjugates.
        For n_b original baselines, row i = (t*n_freq + f)*(2*n_b) + b.
    noise_vec : numpy.ndarray or None
        Difference-based noise estimate in the same order, or None.

    Notes
    -----
    Conjugate copies are not independent noise samples. This function neither
    filters flags nor handles baseline-dependent missing time samples.
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

    Returns
    -------
    pyuvdata.UVData
        The input object, modified in place to contain pI only when formed
        from XX and YY. If pI is already present, returns the object unchanged;
        in that case ``norm`` is not applied and other polarisations are not removed.
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

    Returns
    -------
    numpy.ndarray
        Numeric visibility values converted using the Rayleigh--Jeans factor
        10**(-26) * c**2 / (2*k_B*freqs**2), multiplied by 1,000 for mK sr.
        Frequency factors broadcast over the leading data axes; a one-dimensional
        input acquires a leading singleton axis. No beam correction is applied.
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


