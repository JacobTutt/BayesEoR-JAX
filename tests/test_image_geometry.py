import numpy as np

from bayeseor_jax import ImageGeometry, build_image_geometry


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
