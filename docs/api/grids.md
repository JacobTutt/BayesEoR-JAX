# Sky and Fourier grids

`build_image_geometry` selects the HEALPix pixels of each model field and tracks
their directions through the observation. `build_fourier_geometry` defines the
EoR coefficient grid, whose modes are converted to wavenumbers and grouped into
[power-spectrum bins](../inference/power-spectrum.md).

## Sky pixels

::: bayeseor_jax.model.observation.image_geometry.build_image_geometry
    options:
      heading_level: 3

::: bayeseor_jax.model.observation.image_geometry.ImageGeometry
    options:
      heading_level: 3

::: bayeseor_jax.model.observation.image_geometry.required_healpix_nside
    options:
      heading_level: 3

## Fourier grid

::: bayeseor_jax.model.cosmology.fourier_geometry.build_fourier_geometry
    options:
      heading_level: 3

::: bayeseor_jax.model.cosmology.fourier_geometry.FourierGeometry
    options:
      heading_level: 3

## Wavenumbers and bins

::: bayeseor_jax.model.cosmology.transforms.Cosmology
    options:
      heading_level: 3

::: bayeseor_jax.model.cosmology.power_spectrum.generate_k_modes_in_physical_coordinates
    options:
      heading_level: 3

::: bayeseor_jax.model.cosmology.power_spectrum.generate_k_cube_model_spherical_binning
    options:
      heading_level: 3
