import numpy as np
import pytest

from bayeseor_jax import ImageGeometry, build_image_geometry, required_healpix_nside
from bayeseor_jax.model.observation.image_geometry import _query_disc_union, _select_image_pixels


@pytest.mark.parametrize("baseline,frequency,samples,expected", [
    (40., 159.082e6, 2., 64),
    (80., 159.082e6, 2., 128),
    (40., 318.164e6, 2., 128),
    (40., 159.082e6, 4., 128),
    (0., 159.082e6, 2., 1),
    (0.1, 1e6, 2., 1),
])
def test_required_nside_samples_shortest_fringe(baseline, frequency, samples, expected):
    nside = required_healpix_nside(baseline, frequency, samples_per_fringe=samples)
    assert nside == expected
    minimum = np.sqrt(np.pi / 3.) * samples * baseline * frequency / 299792458.
    assert nside >= minimum
    if nside > 1:
        assert nside / 2 < minimum


@pytest.mark.parametrize("baseline,frequency,samples", [
    (-1., 1e8, 2.), (np.nan, 1e8, 2.), (np.inf, 1e8, 2.),
    (40., 0., 2.), (40., -1., 2.), (40., np.nan, 2.), (40., np.inf, 2.),
    (40., 1e8, 1.), (40., 1e8, np.nan), (40., 1e8, np.inf),
    (1e20, 1e8, 2.),
])
def test_required_nside_rejects_invalid_inputs(baseline, frequency, samples):
    with pytest.raises(ValueError):
        required_healpix_nside(baseline, frequency, samples_per_fringe=samples)


def test_image_geometry_contains_nested_time_dependent_sky_support():
    geometry = build_image_geometry(
        fov_eor=60.0,
        fov_fg=120.0,
        nside=4,
        n_t=3,
        d_t=60.0,
    )

    assert isinstance(geometry, ImageGeometry)
    assert 0 < int(np.asarray(geometry.eor_fg_shared_axis).sum()) < geometry.n_pix_fg
    assert geometry.lmn_fg.shape == (3, 3, geometry.n_pix_fg)
    assert geometry.az_za_fg.shape == (3, 2, geometry.n_pix_fg)
    np.testing.assert_allclose(
        np.sum(np.asarray(geometry.lmn_fg) ** 2, axis=1),
        1.0,
        atol=1e-14,
    )


def test_both_fields_include_pixels_from_every_integration():
    centers = [(0.0, -30.0), (45.0, -30.0), (90.0, -30.0)]
    cosmological, foreground, shared = _select_image_pixels(
        centers, fov_eor=30.0, fov_fg=60.0, nside=8,
    )
    for pixels, diameter in ((cosmological, 30.0), (foreground, 60.0)):
        snapshots = [
            _query_disc_union([center], np.deg2rad(diameter / 2), nside=8)
            for center in centers
        ]
        np.testing.assert_array_equal(pixels, np.unique(np.concatenate(snapshots)))
        assert len(pixels) > len(snapshots[1])
    np.testing.assert_array_equal(np.asarray(foreground)[np.asarray(shared)], cosmological)
