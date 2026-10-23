# EoR model

BayesEoR describes the cosmological 21-cm signal on a three-dimensional Fourier
grid. Each coefficient belongs to a single spatial scale, giving the model an
intrinsic link to the power spectrum we aim to recover, and exposing the
line-of-sight structure that distinguishes the signal from spectrally smooth
foregrounds.

[![EoR-only forward model: Fourier coefficients pass through F_z, F prime, the primary beam and the three-dimensional fringe response to modelled data.](../assets/diagrams/eor-workflow.png#only-light)](../assets/diagrams/eor-workflow.png)
[![EoR-only forward model: Fourier coefficients pass through F_z, F prime, the primary beam and the three-dimensional fringe response to modelled data.](../assets/diagrams/eor-workflow-dark.png#only-dark)](../assets/diagrams/eor-workflow-dark.png)

The coefficients, $\boldsymbol a$, are mapped to the visibilities by a sequence
of linear operators, read from right to left:

$$
\boldsymbol m_{21} = \mathbf F^{-1}\,\mathbf P\,\mathbf F'\,\mathbf F_z\,\boldsymbol a
\equiv \mathbf T_{21}\,\boldsymbol a.
$$

## 1. Choose the Fourier modes

The coefficients are defined on a grid of transverse angular modes, $(u,v)$, and
line-of-sight spectral modes, $\eta$, the Fourier dual of frequency. This model
grid is separate from the baselines sampled by the instrument: it describes a
local tangent plane, while the later propagation to visibilities uses the full
three-dimensional sky directions.

=== "Maths"

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

=== "Code"

    [build_fourier_geometry](../api/geometry.md#bayeseor_jax.model.cosmology.fourier_geometry.build_fourier_geometry) takes grid sizes and spacings, and returns the retained coordinates and integration weights. This example uses eight transverse pairs and three spectral modes.

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

At each transverse mode, the spectral Fourier operator, $\mathbf F_z$, combines
the spectral modes into the field at each observing frequency.

=== "Maths"

    $$
    S_h(\nu_f)=\sum_e(F_z)_{fe}a_{he},
    \qquad
    (F_z)_{fe}=\Delta\eta\,
    \exp[-2\pi i(\nu_f-\nu_{\rm ref})\eta_e].
    $$

    The configured builder uses the middle observing channel as $\nu_{\rm ref}$. The factor $\Delta\eta$ supplies the Fourier integration weight.

=== "Code"

    [build_spectral_fourier_operator](../api/operators.md#bayeseor_jax.model.operators.build_spectral_fourier_operator) accepts spectral coordinates, frequency offsets in Hz and the spectral spacing. It returns a matrix shaped `(n_frequency, n_eta_mode)`.

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

The transverse Fourier operator, $\mathbf F'$, maps the angular modes onto the
HEALPix pixels of the model field. Each coefficient contributes across the whole
image, rather than to a single pixel. The transform uses the pixel directions at
the reference integration, and the same celestial pixels are then tracked as the
Earth rotates.

=== "Maths"

    $$
    I_{21,p}(\nu_f)=\sum_hF'_{ph}S_h(\nu_f),
    \qquad
    F'_{ph}=\Delta u\,\Delta v\,
    \exp[-2\pi i(u_hl_p+v_hm_p)].
    $$

    Here $(l_p,m_p)$ are the reference-time direction cosines of pixel $p$. The pixel set contains the union of directions entering the cosmological field during the observation.

=== "Code"

    [build_image_geometry](../api/geometry.md#bayeseor_jax.model.observation.image_geometry.build_image_geometry) selects and tracks the sky pixels. [build_transverse_fourier_operator](../api/operators.md#bayeseor_jax.model.operators.build_transverse_fourier_operator) maps the Fourier grid onto their reference-time coordinates, returning `(n_pixel, n_uv_mode)`.

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

The primary beam, $\mathbf P$, weights each pixel by its direction and frequency,
with zero response at and below the horizon. See the
[primary beam](../data/instrument.md#primary-beam) for the supported models.

=== "Maths"

    $$
    I^{\rm beam}_{tfp}=P_{tfp}\,I_{21,p}(\nu_f).
    $$

    The beam operator is diagonal in image space. We apply its weights directly rather than constructing a dense diagonal matrix.

=== "Code"

    [Beam_Model.get_beam_vals](../api/data.md#bayeseor_jax.model.instrument.beam.Beam_Model.get_beam_vals) takes azimuth/zenith-angle coordinates and frequencies in Hz. Its output order is `(frequency, time, pixel)`; the visibility operator expects `(time, frequency, pixel)`.

    ```python
    from bayeseor_jax.model.instrument.beam import Beam_Model

    beam_model = Beam_Model(beam_type="airy", diam=14.2)
    beam = beam_model.get_beam_vals(az_za, frequencies_hz).transpose(1, 0, 2)
    beam_weighted_brightness = beam * brightness[None, :, :]
    assert beam_weighted_brightness.shape == (1, 4, lmn.shape[-1])
    ```

## 5. Add each pixel's contribution to the visibilities

Finally, $\mathbf F^{-1}$ applies the three-dimensional fringe response of each
baseline and sums over the pixels, giving the cosmological visibilities. In the
code, the beam and fringe are combined into a single operator,
$\mathbf R = \mathbf F^{-1}\mathbf P$, which already includes the pixel solid
angle, so it must not be applied again.

=== "Maths"

    $$
    \delta m_{21,tfbp}
    =\underbrace{\Delta\Omega\,
    e^{+2\pi i\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)}
    P_{tfp}}_{R_{tfbp}}\,I_{21,p}(\nu_f),
    \qquad
    m_{21,tfb}=\sum_p\delta m_{21,tfbp}.
    $$

    Here $\Delta\Omega$ is the pixel solid angle and $\boldsymbol u_b$ is the baseline in wavelengths. The positive fringe sign follows the code's `ant2 - ant1` baseline convention. See [build_sky_to_visibility_operator](../api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator) for the convention and units.

=== "Code"

    [build_sky_to_visibility_operator](../api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator) combines sky directions, baselines in wavelengths, beam weights and pixel area into an array shaped `(time, frequency, baseline, pixel)`.

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

Combining these gives $\mathbf T_{21}$, whose columns describe how each Fourier
coefficient contributes to every visibility. Applying it to $\boldsymbol a$ gives
the same result as following the steps above.

=== "Maths"

    Combining the steps, the contribution of every coefficient $a_{he}$ to the
    visibility measured by baseline $b$ at frequency $\nu_f$ and time $t$ is

    $$
    \begin{aligned}
    m_{21,tfb}
    ={}& \sum_{h,e}\sum_{p}
    \Delta\Omega\,
    \exp\!\left[+2\pi i\,\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)\right]
    P_{tfp} \\
    &\times \Delta u\,\Delta v\,
    \exp\!\left[-2\pi i\,(u_h l_p + v_h m_p)\right] \\
    &\times \Delta\eta\,
    \exp\!\left[-2\pi i\,(\nu_f-\nu_{\rm ref})\,\eta_e\right]
    a_{he}.
    \end{aligned}
    $$

    Stacking over times, frequencies and baselines gives
    $\boldsymbol m_{21} = \mathbf F^{-1}\mathbf P\mathbf F'\mathbf F_z\boldsymbol a
    \equiv \mathbf T_{21}\boldsymbol a$. Rows are ordered by time, frequency and
    baseline, with baseline varying fastest; columns are ordered by transverse
    mode, then spectral mode.

=== "Code"

    [build_T_21_core](../api/operators.md#bayeseor_jax.model.operators.build_T_21_core) contracts the three supplied operators. [T_21](../api/operators.md#bayeseor_jax.model.operators.T_21) provides the same calculation while constructing the beam-weighted response from geometry arrays.

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

    For larger calculations, [T_21_build_save](../api/construction.md#bayeseor_jax.model.construction.operator_construction.T_21_build_save) evaluates CPU time-frequency blocks and saves `T_21.npy`. The [construction workflow](construction.md) changes how this operator is built and stored, not its mathematical definition.

## Configuration

Setup builds the Fourier grid from the observation and these settings. A larger
field of view or Fourier grid adds coefficients to the model, increasing the
memory and computation required.

`fov_eor`
:   The full angular diameter of the EoR model field.

    - **Default:** 10.
    - **Units:** degrees.
    - **Note:** the model covers every pixel that enters this field during the
      observation.

`n_u_eor`, `n_v_eor`
:   The number of transverse modes along $u$ and $v$.

    - **Default:** inferred, so the grid covers the longest baseline at the
      highest frequency, plus half the beam's aperture width on each side.
    - **Note:** smaller values are kept, with a warning.

`du_eor`, `dv_eor`
:   The transverse mode spacings.

    - **Default:** the reciprocal of the model field's angular extent along each
      axis.
    - **Units:** inverse radians.
    - **Note:** values that differ from the default by more than 5% produce a
      warning.

`n_eta`
:   The number of line-of-sight modes.

    - **Default:** `n_freq`.

`nside`
:   The HEALPix resolution of the image grid, shared with the foreground model.

    - **Default:** inferred from the longest baseline and highest frequency.
    - **Note:** lower values are kept, with a warning.

See the [configuration reference](../examples/configuration.md), and the configurations in
`examples/eor/` and `examples/eor_fgs/` for complete runs.
