"""Tests for ``bayeseor_jax.model.observation.image_geometry``: sky pixels and directions."""

from __future__ import annotations

import astropy.units as u
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from bayeseor_jax.model.observation.image_geometry import (
    ImageGeometry,
    _angular_lm_extent,
    _construct_time_array,
    _observation_field_center,
    _query_disc_union,
    _select_image_pixels,
    _topocentric_coordinates,
    build_image_geometry,
    required_healpix_nside,
)

C = 299792458.0
JD = 2459999.1


@pytest.fixture(scope="module")
def location(hera_latlonalt):
    """HERA as an astropy EarthLocation."""
    lat, lon, alt = hera_latlonalt
    return EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m)


def _brute_force_disc(nside, centre_radec, radius_rad):
    """RING pixels whose centres lie strictly within ``radius_rad`` of a centre."""
    ra, dec = np.deg2rad(centre_radec)
    centre = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    vecs = np.asarray(jhp.pix2vec(nside, jnp.arange(12 * nside**2)))
    vecs = vecs if vecs.shape[-1] == 3 else vecs.T
    return np.flatnonzero(vecs @ centre > np.cos(radius_rad))


class TestRequiredHealpixNside:
    """Baseline-driven HEALPix resolution estimate."""

    @pytest.mark.parametrize("baseline, freq", [(14.6, 150e6), (100.0, 200e6), (1.0, 50e6), (300.0, 100e6)])
    def test_smallest_power_of_two_satisfying_criterion(self, baseline, freq):
        """The result is the smallest power of two at or above sqrt(pi/3)*2*b*nu/c."""
        minimum = np.sqrt(np.pi / 3) * 2 * baseline * freq / C
        nside = required_healpix_nside(baseline, freq)
        assert nside & (nside - 1) == 0
        assert nside >= minimum
        assert nside == 1 or nside / 2 < max(1, np.ceil(minimum))

    def test_zero_baseline_returns_one(self):
        """With no baseline there is no fringe requirement."""
        assert required_healpix_nside(0.0, 150e6) == 1

    def test_more_samples_never_lowers_resolution(self):
        """Increasing ``samples_per_fringe`` never decreases nside."""
        assert required_healpix_nside(40.0, 150e6, samples_per_fringe=8) >= required_healpix_nside(40.0, 150e6)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((-1.0, 150e6), {}),
            ((np.inf, 150e6), {}),
            ((10.0, 0.0), {}),
            ((10.0, np.nan), {}),
            ((10.0, 150e6), dict(samples_per_fringe=1.5)),
            ((1e12, 1e12), {}),
        ],
    )
    def test_invalid_inputs_raise(self, args, kwargs):
        """Non-physical inputs or resolutions beyond HEALPix limits raise ValueError."""
        with pytest.raises(ValueError):
            required_healpix_nside(*args, **kwargs)


class TestConstructTimeArray:
    """Integration centres around the reference time."""

    def test_odd_count_is_symmetric(self):
        """For odd n_t the reference is the central integration."""
        times, ref = _construct_time_array(JD, 5, 86400.0)
        np.testing.assert_allclose(times, JD + np.arange(-2, 3))
        assert ref == 2

    def test_even_count_has_extra_earlier_time(self):
        """For even n_t the reference index is n_t // 2, with one more time before it."""
        times, ref = _construct_time_array(JD, 4, 86400.0)
        np.testing.assert_allclose(times, JD + np.arange(-2, 2))
        assert ref == 2
        assert times[ref] == JD

    def test_spacing_in_seconds(self):
        """Consecutive integrations are separated by ``d_t`` seconds."""
        times, _ = _construct_time_array(JD, 3, 30.0)
        np.testing.assert_allclose(np.diff(times) * 86400.0, 30.0, rtol=1e-6)


class TestObservationFieldCenter:
    """Zenith on the celestial sphere."""

    def test_centre_is_at_zenith(self, location):
        """Transforming the returned RA/Dec back to the telescope frame gives altitude 90 deg."""
        ra, dec = _observation_field_center(JD, location)
        altaz = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").transform_to(
            AltAz(obstime=Time(JD, format="jd", scale="utc"), location=location)
        )
        assert altaz.alt.deg == pytest.approx(90.0, abs=1e-6)

    def test_declination_near_latitude(self, location, hera_latlonalt):
        """The zenith declination is close to the telescope latitude."""
        _, dec = _observation_field_center(JD, location)
        assert dec == pytest.approx(hera_latlonalt[0], abs=0.5)

    def test_right_ascension_advances_with_time(self, location):
        """Zenith moves east in RA by about 15 deg per sidereal hour."""
        ra0, _ = _observation_field_center(JD, location)
        ra1, _ = _observation_field_center(JD + 1 / 24, location)
        assert (ra1 - ra0) % 360 == pytest.approx(15.04, abs=0.05)


class TestQueryDiscUnion:
    """Pixels whose centres lie in the field at any time."""

    def test_single_disc_matches_brute_force(self):
        """One disc selects exactly the pixel centres within its radius."""
        centre, radius = (30.0, -30.0), np.deg2rad(12.0)
        out = _query_disc_union([centre], radius, nside=16)
        np.testing.assert_array_equal(out, _brute_force_disc(16, centre, radius))

    def test_union_of_discs(self):
        """Several centres give the sorted union of their individual discs."""
        centres, radius = [(30.0, -30.0), (40.0, -30.0)], np.deg2rad(8.0)
        out = _query_disc_union(centres, radius, nside=16)
        expected = np.union1d(*[_brute_force_disc(16, c, radius) for c in centres])
        np.testing.assert_array_equal(out, expected)
        assert np.all(np.diff(out) > 0)

    def test_no_centres_gives_empty_selection(self):
        """An empty list of centres selects no pixels."""
        assert _query_disc_union([], 0.1, nside=8).size == 0


class TestSelectImagePixels:
    """Cosmological and foreground pixel selections."""

    CENTRES = [(30.0, -30.0), (31.0, -30.0)]

    def test_equal_fields_share_every_pixel(self):
        """With equal fields the foreground and EoR selections coincide."""
        eor, fg, shared = _select_image_pixels(self.CENTRES, fov_eor=20.0, fov_fg=20.0, nside=16)
        np.testing.assert_array_equal(eor, fg)
        assert bool(jnp.all(shared))

    def test_wider_foreground_field_contains_eor(self):
        """A wider foreground field is a superset, and the mask locates the EoR pixels."""
        eor, fg, shared = _select_image_pixels(self.CENTRES, fov_eor=10.0, fov_fg=30.0, nside=16)
        assert fg.size > eor.size
        np.testing.assert_array_equal(np.asarray(fg)[np.asarray(shared)], eor)

    def test_footprint_includes_every_child(self):
        """Each pixel selected on the footprint grid contributes all its children."""
        coarse, _, _ = _select_image_pixels(self.CENTRES, fov_eor=20.0, fov_fg=20.0, nside=8)
        fine, _, _ = _select_image_pixels(
            self.CENTRES, fov_eor=20.0, fov_fg=20.0, nside=32, footprint_nside=8
        )
        assert fine.size == 16 * coarse.size
        parents = np.asarray(jhp.ring2nest(32, fine)) // 16
        np.testing.assert_array_equal(np.unique(parents), np.sort(np.asarray(jhp.ring2nest(8, coarse))))

    @pytest.mark.parametrize("nside, footprint", [(16, 3), (12, 4), (8, 16), (16, 0)])
    def test_invalid_footprint_raises(self, nside, footprint):
        """Footprint refinement needs powers of two with footprint_nside <= nside."""
        with pytest.raises(ValueError):
            _select_image_pixels(self.CENTRES, fov_eor=20.0, fov_fg=20.0, nside=nside, footprint_nside=footprint)


class TestTopocentricCoordinates:
    """Time-dependent directions of fixed celestial pixels."""

    @pytest.fixture(scope="class")
    @staticmethod
    def result(location):
        """Coordinates of a few pixels at two times."""
        pixels = jnp.asarray([0, 100, 500, 1000, 2000])
        times = np.array([JD, JD + 0.01])
        lmn, az_za = _topocentric_coordinates(times, pixels, nside=16, telescope_location=location)
        return pixels, times, np.asarray(lmn), np.asarray(az_za)

    def test_shapes(self, result):
        """Arrays have shapes (n_time, 3, n_pix) and (n_time, 2, n_pix)."""
        _, _, lmn, az_za = result
        assert lmn.shape == (2, 3, 5)
        assert az_za.shape == (2, 2, 5)

    def test_direction_cosines_have_unit_norm(self, result):
        """Each (l, m, n) is a unit vector."""
        np.testing.assert_allclose(np.linalg.norm(result[2], axis=1), 1.0, rtol=1e-12)

    def test_matches_astropy_altaz(self, result, location):
        """Azimuth and zenith angle agree with a direct astropy transform."""
        pixels, times, _, az_za = result
        ra, dec = jhp.pix2ang(16, pixels, lonlat=True)
        sky = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs")
        for it, jd in enumerate(times):
            altaz = sky.transform_to(AltAz(obstime=Time(jd, format="jd", scale="utc"), location=location))
            np.testing.assert_allclose(az_za[it, 0], altaz.az.rad, atol=1e-9)
            np.testing.assert_allclose(az_za[it, 1], np.pi / 2 - altaz.alt.rad, atol=1e-9)

    def test_lmn_follows_east_north_up_convention(self, result):
        """(l, m, n) = (sin za sin az, sin za cos az, cos za)."""
        _, _, lmn, az_za = result
        az, za = az_za[:, 0], az_za[:, 1]
        np.testing.assert_allclose(lmn[:, 0], np.sin(za) * np.sin(az), atol=1e-12)
        np.testing.assert_allclose(lmn[:, 1], np.sin(za) * np.cos(az), atol=1e-12)
        np.testing.assert_allclose(lmn[:, 2], np.cos(za), atol=1e-12)

    def test_directions_change_with_time(self, result):
        """The same celestial pixel moves in the instrument frame between times."""
        assert not np.allclose(result[2][0], result[2][1])


class TestAngularLmExtent:
    """Pixel-padded angular widths in the reference chart."""

    def test_known_extent(self):
        """Widths are the ranges of atan2(l, n) and atan2(m, n) plus one pixel width."""
        angles_l = np.deg2rad([-5.0, 0.0, 5.0])
        angles_m = np.deg2rad([-2.0, 1.0, 2.0])
        l, m = np.tan(angles_l), np.tan(angles_m)
        lmn = np.stack([l, m, np.ones(3)])
        lmn /= np.linalg.norm(lmn, axis=0)
        width_l, width_m = _angular_lm_extent(lmn, pixel_area_sr=1e-4)
        assert width_l == pytest.approx(np.deg2rad(10.0) + 1e-2)
        assert width_m == pytest.approx(np.deg2rad(4.0) + 1e-2)

    def test_transposed_input_accepted(self, rng):
        """An (n_pix, 3) array gives the same widths as its transpose."""
        lmn = rng.normal(size=(3, 7))
        lmn[2] = np.abs(lmn[2]) + 1.0
        assert _angular_lm_extent(lmn.T, 1e-4) == _angular_lm_extent(lmn, 1e-4)

    @pytest.mark.parametrize("shape", [(3,), (2, 4), (3, 2, 2)])
    def test_invalid_shapes_raise(self, shape):
        """Inputs that are not (3, n) or (n, 3) raise ValueError."""
        with pytest.raises(ValueError):
            _angular_lm_extent(np.ones(shape), 1e-4)


class TestBuildImageGeometry:
    """The complete sky geometry."""

    @pytest.fixture(scope="class")
    @staticmethod
    def geometry(hera_latlonalt):
        """A 15 deg EoR field inside a 30 deg foreground field, three integrations."""
        return build_image_geometry(
            fov_eor=15.0, fov_fg=30.0, nside=16, telescope_latlonalt=hera_latlonalt,
            jd_center=JD, n_t=3, d_t=600.0,
        )

    def test_returns_image_geometry(self, geometry):
        """The result is an ``ImageGeometry`` with consistent array shapes."""
        assert isinstance(geometry, ImageGeometry)
        assert geometry.lmn_fg.shape == (3, 3, geometry.n_pix_fg)
        assert geometry.az_za_fg.shape == (3, 2, geometry.n_pix_fg)
        assert geometry.eor_fg_shared_axis.shape == (geometry.n_pix_fg,)
        assert geometry.time_ind_center == 1

    def test_pixel_area(self, geometry):
        """Each pixel subtends 4 pi / (12 nside^2) steradians."""
        assert geometry.pixel_area_sr == pytest.approx(4 * np.pi / (12 * 16**2))

    def test_eor_field_is_a_strict_subset(self, geometry):
        """The narrower EoR field selects some but not all foreground pixels."""
        n_eor = int(np.sum(geometry.eor_fg_shared_axis))
        assert 0 < n_eor < geometry.n_pix_fg

    def test_foreground_extent_exceeds_eor_extent(self, geometry):
        """The wider foreground field has larger angular extents."""
        assert geometry.fov_l_angle_fg > geometry.fov_l_angle_eor
        assert geometry.fov_m_angle_fg > geometry.fov_m_angle_eor

    def test_extent_is_roughly_field_diameter(self, geometry):
        """The EoR width in m is close to its 15 deg diameter."""
        assert np.rad2deg(geometry.fov_m_angle_eor) == pytest.approx(15.0, abs=5.0)

    def test_field_is_centred_on_zenith_at_reference_time(self, geometry):
        """At the reference time the closest pixel is within a pixel width of zenith."""
        za = np.asarray(geometry.az_za_fg[geometry.time_ind_center, 1])
        assert za.min() < 2 * np.sqrt(geometry.pixel_area_sr)

    def test_default_foreground_field_is_eor_field(self, hera_latlonalt):
        """``fov_fg=None`` uses the EoR field, so every pixel is shared and extents agree."""
        geometry = build_image_geometry(fov_eor=20.0, nside=16, telescope_latlonalt=hera_latlonalt, jd_center=JD)
        assert geometry.n_pix_fg != 3
        assert bool(jnp.all(geometry.eor_fg_shared_axis))
        assert geometry.fov_l_angle_fg == pytest.approx(geometry.fov_l_angle_eor)
        assert geometry.fov_m_angle_fg == pytest.approx(geometry.fov_m_angle_eor)

    def test_three_pixel_field_extents_agree(self, hera_latlonalt):
        """A three-pixel field gives the same EoR and foreground extents when they coincide."""
        geometry = build_image_geometry(fov_eor=15.0, nside=8, telescope_latlonalt=hera_latlonalt, jd_center=JD)
        assert geometry.n_pix_fg == 3
        assert geometry.fov_l_angle_eor == pytest.approx(geometry.fov_l_angle_fg)

    def test_drift_scan_adds_pixels(self, hera_latlonalt):
        """A longer observation selects at least as many pixels as a single snapshot."""
        kw = dict(fov_eor=15.0, nside=16, telescope_latlonalt=hera_latlonalt, jd_center=JD)
        one = build_image_geometry(n_t=1, **kw)
        many = build_image_geometry(n_t=5, d_t=1200.0, **kw)
        assert many.n_pix_fg > one.n_pix_fg

    def test_footprint_refinement_multiplies_pixels(self, hera_latlonalt):
        """Refining with a footprint grid keeps coverage, giving 4x the pixels per level."""
        kw = dict(fov_eor=20.0, telescope_latlonalt=hera_latlonalt, jd_center=JD)
        coarse = build_image_geometry(nside=8, **kw)
        fine = build_image_geometry(nside=16, footprint_nside=8, **kw)
        assert fine.n_pix_fg == 4 * coarse.n_pix_fg
