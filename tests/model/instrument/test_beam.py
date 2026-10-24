"""Tests for ``bayeseor_jax.model.instrument.beam``: primary-beam power responses."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import j1

from bayeseor_jax.model.instrument.beam import Beam_Model

C = 299792458.0
FREQS = np.array([140.0e6, 150.0e6, 160.0e6])
ZA = np.deg2rad(np.array([0.0, 3.0, 10.0, 25.0, 60.0, 89.0, 90.0, 100.0]))
AZ = np.deg2rad(np.linspace(0.0, 315.0, ZA.size))

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _coords(n_t=2):
    """Az/za coordinates of shape (n_t, 2, n_pix), identical at each time."""
    return jnp.asarray(np.broadcast_to(np.stack([AZ, ZA]), (n_t, 2, ZA.size)).copy())


def _evaluate(beam, n_t=2):
    """Beam values of shape (n_freq, n_t, n_pix) for the test directions."""
    return np.asarray(beam.get_beam_vals(_coords(n_t), jnp.asarray(FREQS)))


def _airy(za, diam, freq):
    """Independent Airy power pattern [2 J1(x)/x]^2 with unit peak."""
    x = np.pi * diam * np.sin(za) * freq / C
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x == 0, 1.0, (2 * j1(x) / x) ** 2)


def _gauss(za, sigma, amp=1.0):
    """Independent Gaussian in zenith angle."""
    return amp * np.exp(-(za**2) / (2 * sigma**2))


def _airy_sigma(diam, freq):
    """Gaussian width matched to the Airy main lobe (pyuvsim convention)."""
    return np.arcsin(2.2150894 * (C / freq) / (np.pi * diam)) * np.sqrt(2) / 2.355


def _mask(values):
    """Zero the expected response at and below the horizon."""
    return np.where(ZA[None, :] < np.pi / 2, values, 0.0)


class TestBeamModelConstruction:
    """Selecting an analytic profile or a beam file."""

    @pytest.mark.parametrize("name", ["uniform", "gaussian", "airy", "gausscosine", "taperairy", "tanhairy"])
    def test_analytic_names(self, name):
        """Each supported analytic name is stored as the beam type."""
        assert Beam_Model(name).beam_type == name

    def test_name_is_case_insensitive(self):
        """Analytic names are matched case-insensitively."""
        assert Beam_Model("Gaussian").beam_type == "gaussian"

    def test_none_is_uniform(self):
        """``beam_type=None`` selects the uniform beam."""
        beam = Beam_Model(None)
        assert beam.beam_type == "uniform"
        assert beam.uvb is None

    def test_unknown_name_raises(self):
        """An unrecognised name that is not a file raises ValueError."""
        with pytest.raises(ValueError, match="beam_type must be"):
            Beam_Model("parabolic")

    @pytest.mark.parametrize("pol, number", [("xx", -5), ("yy", -6), ("pI", 1)])
    def test_beamfits_file_is_converted_to_requested_power(self, airy_beamfits, pol, number):
        """An electric-field beamfits file becomes a power beam in the requested polarisation."""
        beam = Beam_Model(str(airy_beamfits), pol=pol, freq_interp_kind="linear")
        assert beam.beam_type == "uvbeam"
        assert beam.uvb.beam_type == "power"
        assert list(beam.uvb.polarization_array) == [number]
        assert beam.uvb.interpolation_function == "az_za_simple"
        assert beam.uvb.freq_interp_kind == "linear"


class TestGetBeamVals:
    """Analytic power responses against closed-form references."""

    def test_output_shape(self):
        """Values have shape (n_freq, n_t, n_pix)."""
        assert _evaluate(Beam_Model("uniform"), n_t=3).shape == (3, 3, ZA.size)

    def test_uniform_is_one_above_horizon(self):
        """The uniform beam is one above the horizon and zero at or below it."""
        out = _evaluate(Beam_Model("uniform"))
        np.testing.assert_array_equal(out[:, 0], _mask(np.ones((3, ZA.size))))

    @pytest.mark.parametrize("amp", [1.0, 0.6])
    def test_gaussian_from_fwhm(self, amp):
        """A FWHM Gaussian is frequency independent with peak ``peak_amp``."""
        out = _evaluate(Beam_Model("gaussian", fwhm_deg=12.0, peak_amp=amp))
        sigma = np.deg2rad(12.0 / 2.355)
        expected = _mask(np.broadcast_to(_gauss(ZA, sigma, amp), (3, ZA.size)))
        np.testing.assert_allclose(out[:, 1], expected, rtol=1e-12, atol=1e-300)

    def test_gaussian_half_power_at_half_fwhm(self):
        """The FWHM Gaussian falls to half power at half the FWHM (to the 2.355 rounding)."""
        beam = Beam_Model("gaussian", fwhm_deg=20.0)
        coords = jnp.asarray([[[0.0], [np.deg2rad(10.0)]]])
        value = float(beam.get_beam_vals(coords, jnp.asarray([150e6]))[0, 0, 0])
        assert value == pytest.approx(0.5, rel=1e-3)

    def test_gaussian_from_diameter_is_chromatic(self):
        """A diameter Gaussian narrows with frequency, matching the Airy main lobe."""
        out = _evaluate(Beam_Model("gaussian", diam=14.0))
        expected = _mask(np.stack([_gauss(ZA, _airy_sigma(14.0, f)) for f in FREQS]))
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10, atol=1e-300)

    def test_airy_from_diameter(self):
        """The Airy beam matches [2 J1(x)/x]^2 with x = pi D nu sin(za) / c."""
        out = _evaluate(Beam_Model("airy", diam=14.0))
        expected = _mask(np.stack([_airy(ZA, 14.0, f) for f in FREQS]))
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10, atol=1e-15)

    def test_airy_from_fwhm_uses_effective_diameter(self):
        """An Airy beam from a FWHM uses a frequency-dependent effective diameter."""
        fwhm = np.deg2rad(10.0)
        out = _evaluate(Beam_Model("airy", fwhm_deg=10.0))
        diam = 2.2150894 * (C / FREQS) / (np.pi * np.sin(fwhm / np.sqrt(2)))
        expected = _mask(np.stack([_airy(ZA, d, f) for d, f in zip(diam, FREQS)]))
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10, atol=1e-15)

    def test_airy_from_fwhm_is_achromatic(self):
        """With a fixed FWHM the Airy response is the same at every frequency."""
        out = _evaluate(Beam_Model("airy", fwhm_deg=10.0))
        np.testing.assert_allclose(out[0], out[-1], rtol=1e-10, atol=1e-15)

    def test_gausscosine(self):
        """The gausscosine beam is a Gaussian times cos^2(pi f_c za)."""
        out = _evaluate(Beam_Model("gausscosine", fwhm_deg=15.0, cosfreq=3.0, peak_amp=0.9))
        sigma = np.deg2rad(15.0 / 2.355)
        profile = _gauss(ZA, sigma, 0.9) * np.cos(np.pi * 3.0 * ZA) ** 2
        np.testing.assert_allclose(out[:, 0], _mask(np.broadcast_to(profile, (3, ZA.size))), atol=1e-14)

    def test_taperairy(self):
        """The taperairy beam is an Airy pattern times a FWHM Gaussian taper."""
        out = _evaluate(Beam_Model("taperairy", diam=14.0, fwhm_deg=30.0, peak_amp=0.8))
        taper = _gauss(ZA, np.deg2rad(30.0 / 2.355), 0.8)
        expected = _mask(np.stack([_airy(ZA, 14.0, f) * taper for f in FREQS]))
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10, atol=1e-15)

    def test_tanhairy(self):
        """The tanhairy beam is an Airy pattern times 1 - s tanh(k za)."""
        out = _evaluate(Beam_Model("tanhairy", diam=14.0, tanh_freq=2.0, tanh_sl_red=0.9))
        taper = 1 - 0.9 * np.tanh(2.0 * ZA)
        expected = _mask(np.stack([_airy(ZA, 14.0, f) * taper for f in FREQS]))
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-10, atol=1e-15)

    @pytest.mark.parametrize("name, kwargs", [
        ("gaussian", dict(fwhm_deg=10.0)), ("airy", dict(diam=14.0)),
    ])
    def test_horizon_is_zero(self, name, kwargs):
        """Directions at or below the horizon have zero gain."""
        out = _evaluate(Beam_Model(name, **kwargs))
        np.testing.assert_array_equal(out[..., ZA >= np.pi / 2], 0.0)

    def test_beamfits_matches_analytic_airy_at_grid_nodes(self, airy_beamfits):
        """At tabulated directions and frequencies the file beam matches the Airy power."""
        beam = Beam_Model(str(airy_beamfits), pol="xx")
        za = np.deg2rad(np.array([0.0, 10.0, 20.0, 40.0]))
        coords = jnp.asarray(np.stack([np.zeros_like(za), za])[None])
        freqs = np.array([149.5e6, 150.0e6])
        out = np.asarray(beam.get_beam_vals(coords, jnp.asarray(freqs)))
        expected = np.stack([_airy(za, 14.0, f) for f in freqs])
        np.testing.assert_allclose(out[:, 0], expected, atol=1e-3)

    def test_beamfits_pseudo_stokes_i_equals_xx(self, airy_beamfits):
        """For a symmetric beam, pI (pyuvdata's (XX + YY) / 2) equals the XX response."""
        za = np.deg2rad(np.array([0.0, 10.0, 20.0]))
        coords = jnp.asarray(np.stack([np.zeros_like(za), za])[None])
        freqs = jnp.asarray([150.0e6])
        xx = np.asarray(Beam_Model(str(airy_beamfits), pol="xx").get_beam_vals(coords, freqs))
        pi = np.asarray(Beam_Model(str(airy_beamfits), pol="pI").get_beam_vals(coords, freqs))
        np.testing.assert_allclose(pi, xx, rtol=1e-6)


class TestPrivateHelpers:
    """Width conversions and taper helpers."""

    def test_fwhm_to_stddev(self):
        """The standard deviation is FWHM / 2.355."""
        assert Beam_Model()._fwhm_to_stddev(2.355) == pytest.approx(1.0)

    @pytest.mark.parametrize("fwhm_deg", [5.0, 20.0, 60.0])
    def test_diameter_round_trip_recovers_gaussian_width(self, fwhm_deg):
        """Converting a FWHM to a diameter and back gives sigma = FWHM / 2.355."""
        beam = Beam_Model()
        diam = beam._fwhm_to_diam(fwhm_deg, 150e6)
        sigma = float(beam._diam_to_stddev(diam, 150e6))
        assert sigma == pytest.approx(np.deg2rad(fwhm_deg) / 2.355, rel=1e-6)

    def test_tanh_taper_limits(self):
        """The tanh taper is one at zenith and tends to 1 - s far from it."""
        taper = np.asarray(Beam_Model()._tanh_taper(jnp.asarray([0.0, 50.0]), 1.0, 0.99))
        np.testing.assert_allclose(taper, [1.0, 0.01], atol=1e-12)

    def test_airy_disk_is_one_at_zenith(self):
        """The Airy helper takes its unit limit where x = 0."""
        out = np.asarray(Beam_Model()._airy_disk(jnp.asarray([0.0]), 14.0, 150e6))
        np.testing.assert_allclose(out, [1.0])


class TestApertureWidthUV:
    """Beam support used to pad the cosmological uv grid."""

    def test_uniform_has_no_width(self):
        """The uniform beam adds no uv padding."""
        assert Beam_Model("uniform").aperture_width_uv(FREQS) == 0.0

    @pytest.mark.parametrize("name", ["airy", "taperairy", "tanhairy"])
    def test_airy_like_from_diameter(self, name):
        """Airy-like beams add the dish diameter in wavelengths at the highest frequency."""
        width = Beam_Model(name, diam=14.0).aperture_width_uv(FREQS)
        assert width == pytest.approx(14.0 * FREQS.max() / C)

    def test_airy_from_fwhm_uses_lowest_frequency_diameter(self):
        """With a FWHM, the effective diameter at the lowest frequency is used."""
        beam = Beam_Model("airy", fwhm_deg=10.0)
        diam = beam._fwhm_to_diam(10.0, FREQS.min())
        assert beam.aperture_width_uv(FREQS) == pytest.approx(float(diam) * FREQS.max() / C)

    @pytest.mark.parametrize("nsigma", [1.0, 5.0])
    def test_gaussian_from_fwhm(self, nsigma):
        """A Gaussian adds nsigma / (2 pi sigma) in uv units."""
        width = Beam_Model("gaussian", fwhm_deg=10.0).aperture_width_uv(FREQS, gaussian_nsigma=nsigma)
        assert width == pytest.approx(nsigma / (2 * np.pi * np.deg2rad(10.0 / 2.355)))

    def test_gaussian_from_diameter_uses_lowest_frequency(self):
        """A diameter Gaussian uses its broadest (lowest-frequency) width."""
        width = Beam_Model("gausscosine", diam=14.0).aperture_width_uv(FREQS)
        assert width == pytest.approx(5.0 / (2 * np.pi * _airy_sigma(14.0, FREQS.min())))

    def test_beamfits_warns_and_returns_zero(self, airy_beamfits):
        """A file beam has no analytic width; it warns and returns zero."""
        with pytest.warns(RuntimeWarning, match="UVBeam"):
            assert Beam_Model(str(airy_beamfits)).aperture_width_uv(FREQS) == 0.0

    @pytest.mark.parametrize("name", ["airy", "gaussian"])
    def test_missing_width_raises(self, name):
        """Without a diameter or FWHM, the width cannot be estimated."""
        with pytest.raises(ValueError, match="requires one of diam or fwhm_deg"):
            Beam_Model(name).aperture_width_uv(FREQS)

    def test_non_positive_nsigma_raises(self):
        """A non-positive Gaussian sigma count raises ValueError."""
        with pytest.raises(ValueError, match="gaussian_nsigma"):
            Beam_Model("gaussian", fwhm_deg=10.0).aperture_width_uv(FREQS, gaussian_nsigma=0.0)

    def test_empty_frequencies_raise(self):
        """An empty frequency array raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            Beam_Model("airy", diam=14.0).aperture_width_uv([])
