# EoR model

The EoR model starts with a three-dimensional Fourier description of the cosmological 21-cm field and follows it through the instrument to predict the measured visibilities.

[![EoR-only forward model: Fourier coefficients pass through F_z, F prime, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/eor-workflow.png#only-light)](assets/diagrams/eor-workflow.png)
[![EoR-only forward model: Fourier coefficients pass through F_z, F prime, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/eor-workflow-dark.png#only-dark)](assets/diagrams/eor-workflow-dark.png)

Let's follow this mapping one step at a time. First we turn Fourier coefficients into brightness at each sky pixel, then follow each pixel's contribution to the visibilities. Adding those contributions gives the predicted cosmological signal.

Each step below has optional **Mathematics** and **Code and example** panels. The Python examples build on one another and use a small illustrative grid, not an analysis-ready configuration. For a complete run, use the [example scripts](examples.md).

## 1. Choose the Fourier modes

We describe the field using transverse angular modes and line-of-sight spectral modes. This makes the connection to the cosmological power spectrum natural: each coefficient belongs to a particular spatial scale, rather than to one sky pixel.

These model coordinates are distinct from the baselines sampled by the instrument. The transverse grid describes a local tangent plane; the later propagation to visibilities uses the full three-dimensional sky directions.

??? note "Mathematics: the coefficient grid"

    A coefficient $a_{he}$ belongs to transverse pair $(u_h,v_h)$ and spectral mode $\eta_e$, the Fourier dual of observing frequency. The usual spacings are the reciprocal model extents and bandwidth:

    $$
    \Delta u=\Theta_l^{-1},\qquad
    \Delta v=\Theta_m^{-1},\qquad
    \Delta\eta=B^{-1}.
    $$

    The transverse monopole $(u,v)=(0,0)$ and the $\eta=0$ plane are excluded, leaving

    $$
    N_{21,\mathrm{param}}=(N_uN_v-1)(N_\eta-1).
    $$

    When flattened, coefficients are ordered by transverse mode, with $\eta$ varying fastest.

??? example "Code and example: define the grid"

    [build_fourier_geometry](api/geometry.md#bayeseor_jax.model.cosmology.fourier_geometry.build_fourier_geometry) takes grid sizes and spacings, and returns the retained coordinates and integration weights. This example uses eight transverse pairs and three spectral modes.

    ```python
    import jax
    import jax.numpy as jnp
    from bayeseor_jax.model.cosmology.fourier_geometry import build_fourier_geometry

    jax.config.update("jax_enable_x64", True)
    frequencies_hz = 159e6 + jnp.arange(4) * 200e3
    bandwidth_hz = 4 * 200e3
    angular_extent_rad = jnp.deg2rad(20.0)
    grid = build_fourier_geometry(
        n_u=3, n_v=3, n_eta=4,
        du=float(1 / angular_extent_rad),
        dv=float(1 / angular_extent_rad),
        d_eta=1 / bandwidth_hz,
    )
    assert grid.uv.shape == (2, 8)
    assert grid.eta.shape == (3,)
    ```

    Normal setup derives the transverse spacings from the pixel-padded model footprint and can infer suitable grid sizes from the observation. The explicit sizes here are only for following the calculation.

## 2. Reconstruct the frequency dependence

At each transverse coordinate, combine the spectral modes to find the field at the observing frequencies. This is the role of the spectral Fourier operator, **F_z**. We now have a two-dimensional angular Fourier description at every frequency channel.

??? note "Mathematics: from spectral modes to frequency"

    $$
    S_h(\nu_f)=\sum_e(F_z)_{fe}a_{he},
    \qquad
    (F_z)_{fe}=\Delta\eta\,
    \exp[-2\pi i(\nu_f-\nu_{\rm ref})\eta_e].
    $$

    The configured builder uses the middle observing channel as $\nu_{\rm ref}$. The factor $\Delta\eta$ supplies the Fourier integration weight.

??? example "Code and example: build the spectral operator"

    [build_spectral_fourier_operator](api/operators.md#bayeseor_jax.model.operators.build_spectral_fourier_operator) accepts spectral coordinates, frequency offsets in Hz and the spectral spacing. It returns a matrix shaped `(n_frequency, n_eta_mode)`.

    ```python
    from bayeseor_jax.model.operators import build_spectral_fourier_operator

    frequency_offsets_hz = frequencies_hz - frequencies_hz[len(frequencies_hz) // 2]
    F_z = build_spectral_fourier_operator(
        grid.eta, frequency_offsets_hz, grid.d_eta,
    )
    # A single unit coefficient illustrates one basis response, not a simulated sky.
    a = jnp.zeros((grid.uv.shape[1], grid.eta.size), dtype=jnp.complex128)
    a = a.at[0, 0].set(1.0)
    spectral_field = F_z @ a.T
    assert spectral_field.shape == (4, 8)
    ```

## 3. Reconstruct brightness at the sky pixels

Next, the transverse Fourier operator, **F′**, maps those angular modes onto the selected HEALPix pixels. A Fourier coefficient contributes across the image; it is not an independent pixel amplitude.

The transverse transform uses pixel directions at the reference integration. Those same celestial pixels are then tracked through the observation as Earth rotates.

??? note "Mathematics: from angular modes to brightness"

    $$
    I_{21,p}(\nu_f)=\sum_hF'_{ph}S_h(\nu_f),
    \qquad
    F'_{ph}=\Delta u\,\Delta v\,
    \exp[-2\pi i(u_hl_p+v_hm_p)].
    $$

    Here $(l_p,m_p)$ are the reference-time direction cosines of pixel $p$. The pixel set contains the union of directions entering the cosmological field during the observation.

??? example "Code and example: build the image mapping"

    [build_image_geometry](api/geometry.md#bayeseor_jax.model.observation.image_geometry.build_image_geometry) selects and tracks the sky pixels. [build_transverse_fourier_operator](api/operators.md#bayeseor_jax.model.operators.build_transverse_fourier_operator) maps the Fourier grid onto their reference-time coordinates, returning `(n_pixel, n_uv_mode)`.

    ```python
    from bayeseor_jax.model.observation.image_geometry import build_image_geometry
    from bayeseor_jax.model.operators import build_transverse_fourier_operator

    image = build_image_geometry(
        fov_eor=20.0, nside=16, n_t=1, d_t=10.0,
        jd_center=2461040.5,
    )
    # The shared geometry stores pixels on its foreground axis; select the EoR set.
    selected = image.eor_fg_shared_axis
    lmn = image.lmn_fg[:, :, selected]
    az_za = image.az_za_fg[:, :, selected]
    lm_reference = lmn[image.time_ind_center, :2, :]
    F_prime = build_transverse_fourier_operator(
        grid.uv, lm_reference, grid.du * grid.dv,
    )
    brightness = spectral_field @ F_prime.T
    assert brightness.shape == (4, lmn.shape[-1])
    ```

    The small HEALPix resolution keeps this demonstration lightweight. For an analysis, setup checks the required resolution against the sampled baselines and frequencies.

## 4. Weight each pixel by the primary beam

Not every sky direction contributes equally to the observation. The primary beam weights each pixel according to its direction and frequency, with zero response at or below the horizon. You can use an analytic beam or an interpolated beam model supplied with the [instrument configuration](instrument.md#primary-beam).

??? note "Mathematics: beam-weighted brightness"

    $$
    I^{\rm beam}_{tfp}=P_{tfp}\,I_{21,p}(\nu_f).
    $$

    The beam operator is diagonal in image space. We apply its weights directly rather than constructing a dense diagonal matrix.

??? example "Code and example: evaluate the beam"

    [Beam_Model.get_beam_vals](api/geometry.md#bayeseor_jax.model.instrument.beam.Beam_Model.get_beam_vals) takes azimuth/zenith-angle coordinates and frequencies in Hz. Its output order is `(frequency, time, pixel)`; the visibility operator expects `(time, frequency, pixel)`.

    ```python
    from bayeseor_jax.model.instrument.beam import Beam_Model

    beam_model = Beam_Model(beam_type="airy", diam=14.2)
    beam = beam_model.get_beam_vals(az_za, frequencies_hz).transpose(1, 0, 2)
    beam_weighted_brightness = beam * brightness[None, :, :]
    assert beam_weighted_brightness.shape == (1, 4, lmn.shape[-1])
    ```

## 5. Add each pixel's contribution to the visibilities

Each baseline sees a different geometric phase for each pixel. Apply this full three-dimensional fringe response, then sum the beam-weighted contributions across the sky. This gives the cosmological visibility at each baseline, time and frequency.

In the code, the beam and fringe response are combined into a single operator, **R**. Its pixel integration weight is already included, so it must not be applied a second time.

??? note "Mathematics: the contribution of one pixel"

    $$
    \delta m_{21,tfbp}
    =\underbrace{\Delta\Omega\,
    e^{+2\pi i\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)}
    P_{tfp}}_{R_{tfbp}}\,I_{21,p}(\nu_f),
    \qquad
    m_{21,tfb}=\sum_p\delta m_{21,tfbp}.
    $$

    Here $\Delta\Omega$ is the pixel solid angle and $\boldsymbol u_b$ is the baseline in wavelengths. The positive fringe sign follows the code's `ant2 - ant1` baseline convention. See the [shared response](model.md#shared-sky-to-visibility-response) for the convention and units.

??? example "Code and example: construct the visibility response"

    [build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator) combines sky directions, baselines in wavelengths, beam weights and pixel area into an array shaped `(time, frequency, baseline, pixel)`.

    ```python
    from bayeseor_jax.model.operators import build_sky_to_visibility_operator

    # Two opposite 10-m baselines, in the same topocentric frame as lmn.
    baselines_m = jnp.array([[10.0, -10.0], [0.0, 0.0], [0.0, 0.0]])
    uvw_lambda = frequencies_hz[:, None, None] / 299792458.0 * baselines_m
    R = build_sky_to_visibility_operator(
        lmn, uvw_lambda, beam, image.pixel_area_sr,
    )
    # R already includes the beam: use intrinsic brightness here.
    model_by_step = jnp.einsum("tfbp,fp->tfb", R, brightness)
    assert model_by_step.shape == (1, 4, 2)
    ```

## 6. Put the steps together

We can combine these mappings into one cosmological operator, **T_21**. Each column describes how one Fourier coefficient contributes to all the selected visibilities. Multiplying by the coefficient vector gives the same result as following the individual steps above.

??? note "Mathematics: the complete operator"

    $$
    (T_{21})_{(tfb),(he)}
    =\left[\sum_pR_{tfbp}F'_{ph}\right](F_z)_{fe},
    \qquad
    \boldsymbol m_{21}=\mathbf T_{21}\boldsymbol a.
    $$

    In compact notation, $\mathbf T_{21}=\mathbf F^{-1}\mathbf P\mathbf F'\mathbf F_z$. Rows are ordered by time, frequency and baseline, with baseline varying fastest; columns are ordered by transverse mode, then spectral mode.

??? example "Code and example: assemble and check the operator"

    [build_T_21_core](api/operators.md#bayeseor_jax.model.operators.build_T_21_core) contracts the three supplied operators. [T_21](api/operators.md#bayeseor_jax.model.operators.T_21) provides the same calculation while constructing the beam-weighted response from geometry arrays.

    ```python
    import numpy as np
    from bayeseor_jax.model.operators import build_T_21_core

    T = build_T_21_core(R, F_prime, F_z)
    model = T @ a.reshape(-1)
    assert T.shape == (8, 24)
    np.testing.assert_allclose(
        model, model_by_step.reshape(-1), rtol=1e-10, atol=1e-15,
    )
    ```

    For larger calculations, [T_21_build_save](api/construction.md#bayeseor_jax.model.construction.operator_construction.T_21_build_save) evaluates CPU time-frequency blocks and saves `T_21.npy`. The [construction workflow](forward-model.md) changes how this operator is built and stored, not its mathematical definition.

## From Fourier modes to the power spectrum

The goal is to recover how much cosmological power lies on each spatial scale, not the value of every Fourier coefficient. We group modes into spherical bins and infer their power-spectrum amplitudes. Those amplitudes set the coefficient prior; the individual coefficients are integrated out in [Inference](joint-inference.md).

??? note "Mathematics: cosmological scales and power"

    At the reference redshift, the Fourier coordinates become comoving wavenumbers:

    $$
    k_x=\frac{2\pi u}{D_M},\qquad
    k_y=\frac{2\pi v}{D_M},\qquad
    k_z=\frac{2\pi\eta}{|d\chi/d\nu|}.
    $$

    Modes are binned by $k=(k_x^2+k_y^2+k_z^2)^{1/2}$, with power amplitude

    $$
    \varphi_s=\Delta^2_{21}(k_s)=\frac{k_s^3}{2\pi^2}P_{21}(k_s).
    $$

    The coefficient covariance also includes the Fourier-volume normalisation. The implementation applies this conversion through [PhiInverse](api/inference.md#bayeseor_jax.inference.phi.PhiInverse); the power amplitude is not itself the raw coefficient variance.

??? example "Code and example: find the scale of each mode"

    `generate_k_modes_in_physical_coordinates` converts the retained grid to wavenumber magnitudes in the same order as the operator columns. Setup then assigns these to power-spectrum bins.

    ```python
    from bayeseor_jax.model.cosmology.transforms import Cosmology
    from bayeseor_jax.model.cosmology.power_spectrum import (
        generate_k_modes_in_physical_coordinates,
    )

    cosmology = Cosmology()
    redshift = cosmology.f2z(jnp.mean(frequencies_hz))
    k = generate_k_modes_in_physical_coordinates(
        grid.uv, grid.eta,
        float(cosmology.dL_dth(redshift)),
        float(jnp.abs(cosmology.dL_df(redshift))),
    )
    assert k.shape == (T.shape[1],)
    assert bool(jnp.all(k > 0))
    ```

## Choose the model for your observation

For an actual analysis, configure the field of view and let setup construct the corresponding geometry. Increasing the field or Fourier grid adds coefficients to the marginalised system, so it also increases the memory and computation required.

??? example "Configuration: the EoR model settings"

    | Setting | Role |
    | --- | --- |
    | `fov_eor` | Full angular field of view, in degrees. |
    | `n_u_eor`, `n_v_eor` | Transverse grid sizes, inferred if omitted; undersized supplied values produce warnings. |
    | `du_eor`, `dv_eor` | Optional transverse spacings; otherwise derived from the model field. |
    | `n_eta` | Spectral grid size; defaults to `n_freq`. |
    | `nside` | Shared HEALPix resolution for image quadrature; inferred if omitted. |
    | `priors` | One `[min, max]` range per populated power-spectrum bin in logarithmic power coordinates. |

    The `uniform_priors` setting selects bins with a linear-uniform physical power prior while retaining logarithmic sampler coordinates. See the [configuration reference](configuration.md) and the supplied configurations in `test_examples/eor/` and `test_examples/eor_fgs/` for complete runs.

Continue to the [foreground model](foregrounds.md) to see the alternative image-domain parameterisation, or to [Inference](joint-inference.md) to combine both contributions with their priors.
