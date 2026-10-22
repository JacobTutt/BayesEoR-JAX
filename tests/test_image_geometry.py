import numpy as np

from bayeseor_jax import ImageGeometry, build_image_geometry
from bayeseor_jax.model.observation.image_geometry import _query_disc_union, _select_image_pixels


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
