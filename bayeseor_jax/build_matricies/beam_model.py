"""
HEALPix-based beam model for evaluating the primary beam response at given
(za, az) coordinates of the image domain pixels calculated in `lmn_model_space.py`.
"""
import warnings

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy.constants import c
from scipy.special import j1
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils
from pathlib import Path




class Beam_Model:
    """
    Class to evaluate the primary beam response at given (za, az) coordinates.

    Parameters
    ----------
    beam_type : str, optional
        Can be either a path to a pyuvdata-compatible beam file or one of
        'uniform', 'gaussian', 'airy', 'gausscosine', 'taperairy', or 'tanhairy'.
        Defaults to 'uniform'.
    peak_amp : float, optional
        Peak amplitude of the beam. Defaults to 1.0.
    fwhm_deg : float, optional
        Full width at half maximum of the beam in degrees. Required if
        `beam_type` is 'gaussian' or 'gausscosine'.
    diam : float, optional
        Antenna (aperture) diameter in meters. Used if `beam_type` is 'airy'.
    cosfreq : float, optional
        Cosine frequency in inverse radians. Used if `beam_type` is 'gausscosine'.
    tanh_freq : float, optional
        Exponential frequency (rate parameter) in inverse radians. Used if
        `beam_type` is 'tanhairy'.
    tanh_sl_red : float, optional
        Airy sidelobe amplitude reduction as a fractional percent. Used if
        `beam_type` is 'tanhairy'.
    pol : str, optional
        Polarization string. Used only if `beam_type` is a UVBeam file path.
        Defaults to 'xx'.
    freq_interp_kind : str, optional
        Frequency interpolation kind from scipy.interpolate.interp1d. 
        Defaults to 'cubic'.
    """
    def __init__(
        self,
        beam_type: str | None = None,
        peak_amp: float = 1.0,
        fwhm_deg: float | None = None,
        diam: float | None = None,
        cosfreq: float | None = None,
        tanh_freq: float | None = None,
        tanh_sl_red: float | None = None,
        pol: str = "xx",
        freq_interp_kind: str = "cubic",
    ) -> None:
        
        self.peak_amp = peak_amp
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red
        self.pol = pol
        self.freq_interp_kind = freq_interp_kind
        self.c_ms = float(c.to("m/s").value)

        allowed_types = ["uniform", "gaussian", "airy", "gausscosine", "taperairy", "tanhairy"]

        # Default to uniform beam when no type is provided.
        if beam_type is None:
            self.beam_type = "uniform"
            self.uvb = None
            return

        # If beam_type is a valid path, load a UVBeam file.
        beam_path = Path(str(beam_type))
        if beam_path.exists():
            self.beam_type = "uvbeam"
            # Load the UVBeam file
            uvb = UVBeam()
            uvb.read_beamfits(str(beam_path))
            # Convert to power beam if necessary and select polarization.
            if uvb.beam_type == "efield" and pol in ["xx", "yy"]:
                uvb.efield_to_power()
            elif uvb.beam_type == "efield" and pol == "pI":
                uvb.efield_to_pstokes()
            uvb.select(polarizations=[uvutils.polstr2num(pol)])
            # Set frequency interpolation method for later use in `get_beam_vals`.
            uvb.freq_interp_kind = freq_interp_kind
            if uvb.pixel_coordinate_system == "healpix":
                uvb.interpolation_function = "healpix_simple"
            else:
                uvb.interpolation_function = "az_za_simple"
            # Save the uvb object for later use in `get_beam_vals`.
            self.uvb = uvb
            return

        # Otherwise, treat beam_type as an analytic beam keyword.
        beam_type = str(beam_type).lower()
        # Ensure that the provided beam_type is valid.
        if beam_type not in allowed_types:
            raise ValueError(f"beam_type must be a valid file path or one of {', '.join(allowed_types)}.")
        self.beam_type = beam_type
        self.uvb = None

    def get_beam_vals(self, coords: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
        """
        Get an array of beam values from (az, za) coordinates.

        Parameters
        ----------
        coords : jnp.ndarray
            Shape (n_t, 2, n_pix) containing [az, za] in radians.
        freqs : jnp.ndarray
            Shape (n_freq,) in Hz.

        Returns
        -------
        beam_vals : jnp.ndarray
            Shape (n_freq, n_t, n_pix).
        """
        az = coords[:, 0, :]  # (n_t, n_pix)
        za = coords[:, 1, :]  # (n_t, n_pix)

        # Reshape freqs for broadcasting: (n_freq, 1, 1)
        f = freqs[:, jnp.newaxis, jnp.newaxis]

        # Analytic beams are evaluated in a fully vectorized manner.
        if self.beam_type == "uniform":
            beam_vals = jnp.ones((len(freqs), *za.shape))
        elif self.beam_type in ["gaussian", "gausscosine"]:
            if self.fwhm_deg is not None:
                # FWHM-based Gaussian is usually frequency-independent in this model
                # unless fwhm_deg is passed as an array.
                stddev_scalar = jnp.deg2rad(self._fwhm_to_stddev(self.fwhm_deg))
                # Keep frequency axis so output is always (n_freq, n_t, n_pix).
                stddev_rad = jnp.ones_like(f) * stddev_scalar
            else:
                stddev_rad = self._diam_to_stddev(self.diam, f)  # (n_freq, 1, 1)

            if self.beam_type == "gaussian":
                beam_vals = self._gaussian_za(za, stddev_rad, self.peak_amp)
            else:
                beam_vals = self._gausscosine(
                    za,
                    stddev_rad,
                    self.peak_amp,
                    self.cosfreq,
                )
        elif self.beam_type == "airy":
            if self.diam is not None:
                beam_vals = self._airy_disk(za, self.diam, f)
            else:
                diam_eff = self._fwhm_to_diam(self.fwhm_deg, f)
                beam_vals = self._airy_disk(za, diam_eff, f)
        elif self.beam_type == "taperairy":
            stddev_rad = jnp.deg2rad(self._fwhm_to_stddev(self.fwhm_deg))
            beam_vals = self._airy_disk(
                za,
                self.diam,
                f,
            ) * self._gaussian_za(za, stddev_rad, self.peak_amp)
        elif self.beam_type == "tanhairy":
            beam_vals = self._airy_disk(
                za,
                self.diam,
                f,
            ) * self._tanh_taper(za, self.tanh_freq, self.tanh_sl_red)
        elif self.beam_type == "uvbeam":
            beam_vals = self._get_uvbeam_vals_vectorized(az, za, freqs)
        else:
            raise ValueError(f"Unsupported beam_type: {self.beam_type}")

        # The ground has zero gain. Applying this once to the cached beam cube
        # avoids rebuilding the same time-dependent horizon mask in every
        # sky-to-visibility response block.
        return jnp.where(za[jnp.newaxis, :, :] < jnp.pi / 2, beam_vals, 0.0)

    def _get_uvbeam_vals_vectorized(self, az, za, freqs):
        """Loop over time to use the non-JAX UVBeam interp function."""
        n_freq = len(freqs)
        n_t, n_pix = az.shape

        # Result container
        out = np.zeros((n_freq, n_t, n_pix))

        # Standard UVBeam.interp usually takes (n_pixels) arrays.
        # We loop over time steps.
        for it in range(n_t):
            # interp expects 1D arrays for az/za
            beam_vals, _ = self.uvb.interp(
                az_array=np.asarray(az[it]),
                za_array=np.asarray(za[it]),
                freq_array=np.asarray(freqs),
                reuse_spline=True
            )
            # UVBeam.interp returns (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, Npixels)
            # We select the first pol/freq axis, recieving values (n_freq, n_pix)
            out[:, it, :] = beam_vals[0, 0, 0, :, :].real

        return jnp.asarray(out)

    def _gaussian_za(self, za, sigma, amp):
        """
        Calculate azimuthally symmetric Gaussian beam amplitudes.

        Parameters
        ----------
        za : numpy.ndarray
            Zenith angle of each pixel in radians.
        sigma : float
            Standard deviation in radians.
        amp : float
            Peak amplitude at zenith.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        # za: (n_t, n_pix), sigma: (n_freq, 1, 1) or float
        return amp * jnp.exp(-za ** 2 / (2 * sigma ** 2))

    def _gausscosine(self, za, sigma, amp, cosfreq):
        """
        Calculate azimuthally symmetric Gaussian * cosine^2 beam amplitudes.

        Parameters
        ----------
        za : numpy.ndarray
            Zenith angle of each pixel in radians.
        sigma : float
            Standard deviation in radians.
        amp : float
            Peak amplitude at zenith.
        cosfreq : float
            Cosine squared frequency in inverse radians.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        beam_vals = amp * jnp.exp(-za ** 2 / (2 * sigma ** 2))
        beam_vals *= jnp.cos(2 * jnp.pi * za * cosfreq / 2) ** 2
        return beam_vals

    def _fwhm_to_stddev(self, fwhm):
        """
        Calculate standard deviation from full width at half maximum.

        Parameters
        ----------
        fwhm : float
            Full width half maximum in degrees.

        """
        return fwhm / 2.355

    def _airy_disk(self, za, diam, freq):
        """
        Calculate Airy disk amplitudes.

        Parameters
        ----------
        za : numpy.ndarray of floats
            Zenith angle of each pixel in radians.
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        beam_vals : numpy.ndarray
            Array of Airy disk amplitudes for each zenith angle in `za`.

        """
        # za: (n_t, n_pix), freq: (n_freq, 1, 1) -> xvals: (n_freq, n_t, n_pix)
        xvals = (diam / 2.0) * jnp.sin(za) * (2.0 * jnp.pi * freq / self.c_ms)

        # JAX friendly way to handle the sinc-like limit at x=0
        # We use jnp.where to avoid division by zero
        safe_x = jnp.where(xvals == 0, 1.0, xvals)
        bes_vals = 2.0 * j1(safe_x) / safe_x
        beam_vals = jnp.where(xvals == 0, 1.0, bes_vals)

        return beam_vals ** 2

    def _tanh_taper(self, za, tanh_freq, tanh_sl_red):
        """
        Calculate a tanh tapering function.

        Parameters
        ----------
        za : numpy.ndarray of floats
            Zenith angle of each pixel in radians.
        tanh_freq : float, optional
            Exponential frequency (rate parameter) in inverse radians.
        tanh_sl_red : float, optional
            Airy sidelobe amplitude reduction as a fractional percent.  For
            example, passing 0.99 reduces the sidelobes by 0.01, i.e. two
            orders of magnitude.
        
        Returns
        -------
        taper_vals : numpy.ndarray
            Array of tanh taper amplitudes for each zenith angle in `za`.
        """

        return 1 - tanh_sl_red * jnp.tanh(tanh_freq * za)

    def _fwhm_to_diam(self, fwhm, freq):
        """
        Calculates the effective diameter of an Airy disk from a FWHM.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum of a Gaussian beam in degrees.
        freq : float
            Frequency in Hz.

        Returns
        -------
        diam : float
            Antenna (aperture) diameter in meters with an Airy disk beam
            pattern whose main lobe is described by a Gaussian beam with a
            FWHM of `fwhm`.

        Notes
        -----
        * Modified from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).
        """
        wavelength = self.c_ms / freq
        fwhm_rad = jnp.deg2rad(fwhm)
        return 2.2150894 * wavelength / (jnp.pi * jnp.sin(fwhm_rad / jnp.sqrt(2)))

    def _diam_to_stddev(self, diam, freq):
        """
        Calculate an effective standard deviation of an Airy disk.

        Parameters
        ----------
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        sigma : float
            Standard deviation of a Gaussian envelope which describes the main
            lobe of an Airy disk with aperture `diam`.

        Notes
        -----
        * Copied from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).
        """
        wavelength = self.c_ms / freq
        sigma = jnp.arcsin(2.2150894 * wavelength / (jnp.pi * diam))
        return sigma * jnp.sqrt(2) / 2.355
    
    def aperture_width_uv(
        self,
        freqs_hz: np.ndarray,
        gaussian_nsigma: float = 5.0,
    ) -> float:
        """
        Estimate the beam support width in uv space.

        This quantity is used only for automatic uv-grid sizing. It mirrors the
        old elongated-FoV validation logic: Airy-like beams contribute an
        aperture width equal to the effective dish diameter in wavelengths,
        while Gaussian-like beams contribute ``Nsigma / (2πσ)`` using the
        broadest angular width at the lowest observing frequency.

        Parameters
        ----------
        freqs_hz : array-like
            One-dimensional observing-frequency array in Hz.
        gaussian_nsigma : float, optional
            Number of Gaussian standard deviations used to define the beam
            support in uv space. Defaults to 5.

        Returns
        -------
        float
            Approximate beam-support width in uv units. Returns ``0.0`` for
            uniform beams and UVBeam file inputs, where no simple analytic
            aperture-width estimate is available.
        """
        freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
        if freqs_hz.size == 0:
            raise ValueError("freqs_hz must contain at least one frequency")

        if self.beam_type == "uniform":
            return 0.0

        if self.beam_type == "uvbeam":
            warnings.warn(
                "Automatic uv-grid sizing with a UVBeam file does not have a "
                "simple analytic aperture-width estimate; using zero beam buffer.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        freq_min_hz = float(freqs_hz.min())
        freq_max_hz = float(freqs_hz.max())
        wavelength_max_model = float(self.c_ms / freq_max_hz)

        if self.beam_type in {"airy", "taperairy", "tanhairy"}:
            if self.diam is not None:
                diam_eff = float(self.diam)
            elif self.fwhm_deg is not None:
                diam_eff = float(self._fwhm_to_diam(float(self.fwhm_deg), freq_min_hz))
            else:
                raise ValueError(
                    f"beam_type={self.beam_type!r} requires one of diam or fwhm_deg "
                    "for automatic uv-grid sizing."
                )
            return float(diam_eff / wavelength_max_model)

        if self.beam_type in {"gaussian", "gausscosine"}:
            if gaussian_nsigma <= 0:
                raise ValueError("gaussian_nsigma must be > 0")
            if self.diam is not None:
                stddev_eff = float(self._diam_to_stddev(float(self.diam), freq_min_hz))
            elif self.fwhm_deg is not None:
                stddev_eff = float(np.deg2rad(self._fwhm_to_stddev(float(self.fwhm_deg))))
            else:
                raise ValueError(
                    f"beam_type={self.beam_type!r} requires one of diam or fwhm_deg "
                    "for automatic uv-grid sizing."
                )
            stddev_uv = 1.0 / (2.0 * np.pi * stddev_eff)
            return float(stddev_uv * gaussian_nsigma)

        return 0.0
