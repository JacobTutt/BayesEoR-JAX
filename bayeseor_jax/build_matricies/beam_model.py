"""
Evaluate the primary-beam power response along each HEALPix pixel direction.

The time-dependent (azimuth, zenith angle) coordinates from image geometry
locate each celestial pixel within the instrumental beam. The resulting
frequency-dependent gains weight sky brightness before the three-dimensional
Fourier mapping to visibilities. Pixels at or below the horizon have zero gain.

Supported beam types
--------------------
- ``uniform``: unit power response above the horizon (the default).
- ``gaussian``: Gaussian profile with a specified angular width or an
  aperture-derived, frequency-dependent width.
- ``airy``: Airy power pattern for a circular aperture.
- ``gausscosine``: Gaussian profile modulated by a cosine-squared response.
- ``taperairy``: Airy power pattern multiplied by a Gaussian taper.
- ``tanhairy``: Airy power pattern multiplied by a hyperbolic-tangent taper.
- A beamfits file path: an arbitrary tabulated beam loaded through
  ``pyuvdata.UVBeam`` and interpolated onto the sky-pixel directions and
  observing frequencies. Both HEALPix and azimuth/zenith-angle beam grids
  are supported; ``pol`` selects the polarisation and ``freq_interp_kind``
  selects the frequency interpolation method (cubic by default).
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
    r"""Supply the beam weights multiplying image-domain sky brightness.

    Parameters
    ----------
    beam_type : str, optional
        Can be either a path to a beamfits file readable by UVBeam or one of
        'uniform', 'gaussian', 'airy', 'gausscosine', 'taperairy', or 'tanhairy'.
        Defaults to 'uniform'.
    peak_amp : float, optional
        Peak power-response amplitude for Gaussian-based profiles. Defaults
        to 1.0; the uniform and untapered Airy profiles have unit peak.
    fwhm_deg : float, optional
        Full width at half maximum in degrees. Gaussian-based profiles use
        this width when supplied, otherwise deriving a width from ``diam``.
    diam : float, optional
        Antenna (aperture) diameter in meters. Used if `beam_type` is 'airy'.
    cosfreq : float, optional
        Cosine frequency in inverse radians. Used if `beam_type` is 'gausscosine'.
    tanh_freq : float, optional
        Exponential frequency (rate parameter) in inverse radians. Used if
        `beam_type` is 'tanhairy'.
    tanh_sl_red : float, optional
        Fractional suppression of the Airy power response far from zenith. Used if
        `beam_type` is 'tanhairy'.
    pol : str, optional
        Polarization string. Used only if `beam_type` is a UVBeam file path.
        Defaults to 'xx'.
    freq_interp_kind : str, optional
        Frequency interpolation kind from scipy.interpolate.interp1d. 
        Defaults to 'cubic'.

    Notes
    -----
    Analytic profiles or an interpolated UVBeam power model are evaluated at
    each pixel's topocentric direction and observing frequency. ``get_beam_vals``
    returns the diagonal entries of the primary-beam operator; it does not
    allocate a dense diagonal matrix.
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
        r"""Configure an analytic profile or load the selected UVBeam polarisation.

        Parameters
        ----------
        beam_type : str or None
            Analytic beam name or beamfits file path. None selects a uniform beam.
        peak_amp : float
            Peak power gain for Gaussian-based beam profiles; defaults to 1.
        fwhm_deg : float or None
            Configured beam full width at half maximum in degrees, when used instead
            of an aperture-derived width.
        diam : float or None
            Physical or effective aperture diameter in metres for Airy or Gaussian
            beam profiles.
        cosfreq : float or None
            Angular modulation frequency in inverse radians for the gausscosine
            profile.
        tanh_freq : float or None
            Angular taper rate in inverse radians for the tanhairy profile.
        tanh_sl_red : float or None
            Fractional far-off-axis power suppression for the tanhairy profile.
        pol : str
            Polarisation selected from a file-backed beam, for example xx or pI.
        freq_interp_kind : str
            Frequency interpolation method used by UVBeam; defaults to cubic.

        Returns
        -------
        None
            Initialises the instance without returning a value.

        Notes
        -----
        Pixel gains are evaluated later by ``get_beam_vals``. File-backed
        electric-field beams are converted to the requested power or
        pseudo-Stokes response before interpolation.
        """
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
        r"""Weight each HEALPix sky direction by the beam at every frequency.

        Parameters
        ----------
        coords : jnp.ndarray
            Shape (n_t, 2, n_pix) containing [az, za] in radians.
        freqs : jnp.ndarray
            Shape (n_freq,) in Hz.

        Returns
        -------
        beam_vals : jnp.ndarray
            Dimensionless power response, shape ``(n_freq, n_t, n_pix)``.
            The forward-operator builder reorders the first two axes before
            multiplying these gains by the fringe response.

        Notes
        -----
        For each integration, use the pixel's azimuth and zenith angle to
        evaluate ``P(nu, t, p)``. Set the response to zero where the zenith
        angle is at least pi/2, including pixels that cross the horizon
        during the observation.
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
        r"""Interpolate a loaded beam along the pixel tracks through the sky.

        Parameters
        ----------
        az : array-like
            Azimuths in radians, shape (n_time, n_pixel).
        za : array-like
            Zenith angles in radians, shape (n_time, n_pixel).
        freqs : array-like
            Observing frequencies in Hz, shape (n_frequency,).

        Returns
        -------
        beam : jax.Array
            Interpolated power response, shape (n_frequency, n_time, n_pixel),
            before horizon masking.

        Notes
        -----
        ``az`` and ``za`` have shape ``(n_time, n_pixel)`` in radians;
        ``freqs`` is a one-dimensional Hz array. UVBeam interpolation runs
        once per integration on the host. Return the real selected power
        response as ``(n_frequency, n_time, n_pixel)``; horizon masking is
        applied by ``get_beam_vals``.
        """
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
        r"""Evaluate the Gaussian power response at each pixel's zenith angle.

        Parameters
        ----------
        za : array-like
            Zenith angle of each pixel in radians.
        sigma : float or array-like
            Standard deviation in radians.
        amp : float
            Peak amplitude at zenith.

        Returns
        -------
        beam_vals : jax.Array
            Dimensionless power gains at the supplied zenith angles.

        Notes
        -----
        The profile is ``amp * exp(-za**2 / (2*sigma**2))``. Broadcasting
        allows a frequency-dependent width to weight all pixel tracks.
        """
        # za: (n_t, n_pix), sigma: (n_freq, 1, 1) or float
        return amp * jnp.exp(-za ** 2 / (2 * sigma ** 2))

    def _gausscosine(self, za, sigma, amp, cosfreq):
        r"""Modulate the Gaussian power response with angular cosine-squared structure.

        Parameters
        ----------
        za : array-like
            Zenith angle of each pixel in radians.
        sigma : float or array-like
            Standard deviation in radians.
        amp : float
            Peak amplitude at zenith.
        cosfreq : float
            Cosine squared frequency in inverse radians.

        Returns
        -------
        beam_vals : jax.Array
            Dimensionless modulated power gains at the supplied zenith angles.

        Notes
        -----
        Multiply the Gaussian by ``cos(pi * za * cosfreq)**2`` to introduce
        radial structure in the beam weights applied to image pixels.
        """
        beam_vals = amp * jnp.exp(-za ** 2 / (2 * sigma ** 2))
        beam_vals *= jnp.cos(2 * jnp.pi * za * cosfreq / 2) ** 2
        return beam_vals

    def _fwhm_to_stddev(self, fwhm):
        r"""Convert a Gaussian beam FWHM to its angular standard deviation.

        Parameters
        ----------
        fwhm : float
            Full width half maximum in degrees.

        Returns
        -------
        sigma : float or array-like
            Gaussian angular standard deviation, in the same units as fwhm.

        Notes
        -----
        Uses ``sigma = fwhm / 2.355``. Input and output have the same angular
        units; callers using degrees convert the result to radians.
        """
        return fwhm / 2.355

    def _airy_disk(self, za, diam, freq):
        r"""Evaluate the Airy power response of a circular aperture along pixel directions.

        Parameters
        ----------
        za : array-like
            Zenith angle of each pixel in radians.
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        beam_vals : jax.Array
            Dimensionless power gains, broadcast over frequency, time, and pixel.

        Notes
        -----
        With ``x = pi * diam * freq * sin(za) / c``, the response is
        ``[2 J1(x) / x]**2``, taking its unit limit at zenith. Frequency
        dependence changes the angular beam width and sidelobe positions.
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
        r"""Suppress the off-axis Airy power response with a smooth angular taper.

        Parameters
        ----------
        za : array-like
            Zenith angle of each pixel in radians.
        tanh_freq : float, optional
            Exponential frequency (rate parameter) in inverse radians.
        tanh_sl_red : float, optional
            Fractional suppression at large zenith angle. For example,
            0.99 leaves a limiting multiplicative power factor of 0.01.

        Returns
        -------
        taper_vals : jax.Array
            Array of tanh taper amplitudes for each zenith angle in `za`.

        Notes
        -----
        Return ``1 - tanh_sl_red * tanh(tanh_freq * za)``, which approaches
        ``1 - tanh_sl_red`` far from zenith and equals one at zenith.
        """

        return 1 - tanh_sl_red * jnp.tanh(tanh_freq * za)

    def _fwhm_to_diam(self, fwhm, freq):
        r"""Convert an angular main-lobe width to an effective circular-aperture diameter.

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
        The resulting diameter supplies the Airy response when a beam width
        is configured instead of a physical dish diameter.

        * Modified from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).
        """
        wavelength = self.c_ms / freq
        fwhm_rad = jnp.deg2rad(fwhm)
        return 2.2150894 * wavelength / (jnp.pi * jnp.sin(fwhm_rad / jnp.sqrt(2)))

    def _diam_to_stddev(self, diam, freq):
        r"""Approximate the Airy main lobe by a Gaussian angular width.

        Parameters
        ----------
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        sigma : float
            Standard deviation in radians of a Gaussian envelope describing the main
            lobe of an Airy disk with aperture `diam`.

        Notes
        -----
        Convert aperture diameter and wavelength to a Gaussian standard
        deviation used to evaluate a chromatic Gaussian beam or estimate
        its transverse Fourier support.

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
        r"""Estimate the transverse Fourier extent added by primary-beam weighting.

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

        Notes
        -----
        This quantity supplies the buffer around measured baselines when
        choosing the cosmological uv grid. Airy-like beams contribute an
        aperture width equal to the effective dish diameter in wavelengths,
        while Gaussian-like beams contribute ``Nsigma / (2πσ)`` using the
        broadest angular width at the lowest observing frequency.
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
