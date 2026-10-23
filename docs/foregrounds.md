# Foreground model

BayesEoR describes the foregrounds directly on the sky, as smooth spectra in each
pixel of a spherical HEALPix grid. Working on the sphere, rather than a tangent
plane, lets the model extend all the way to the horizon, while the spectral
smoothness of the foregrounds is what separates them from the 21-cm signal.

[![Foreground-only forward model: image-domain coefficients pass through Q_z, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/foreground-workflow.png#only-light)](assets/diagrams/foreground-workflow.png)
[![Foreground-only forward model: image-domain coefficients pass through Q_z, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/foreground-workflow-dark.png#only-dark)](assets/diagrams/foreground-workflow-dark.png)

The amplitudes, $\boldsymbol q$, are mapped to the visibilities by a sequence of
linear operators, read from right to left:

$$
\boldsymbol m_{\rm FG} = \mathbf F^{-1}\,\mathbf P\,\mathbf Q_z\,\boldsymbol q
\equiv \mathbf T_{\rm FG}\,\boldsymbol q.
$$

## 1. Choose the sky pixels

The model covers every celestial pixel that enters the foreground field of view
during the observation, and tracks its direction relative to the instrument at
each integration. Each pixel carries its own spectral amplitudes, so a wider field
adds coefficients without changing the construction.

=== "Maths"

    For spectral component $g$, collect the amplitudes of all selected pixels into

    $$
    \boldsymbol q_g=(q_{g,1},\ldots,q_{g,N_{\rm pix}})^{\mathsf T},
    \qquad
    \boldsymbol q=(\boldsymbol q_0^{\mathsf T},\ldots,
    \boldsymbol q_{N_\beta}^{\mathsf T})^{\mathsf T}.
    $$

    The configured spectral model has a constant component and one component per power-law index, giving

    $$
    N_{\rm FG,param}=(1+N_\beta)N_{\rm pix}.
    $$

    Pixel identities stay fixed on the celestial sky, but their topocentric directions $\hat{\boldsymbol s}_p(t)=(l_p(t),m_p(t),n_p(t))$ change with observing time.

=== "Code"

    [build_image_geometry](api/geometry.md#bayeseor_jax.model.observation.image_geometry.build_image_geometry) takes field extents, HEALPix resolution, telescope location and observing times. It returns the pixel directions, beam coordinates and pixel solid angle.

    ```python
    import jax
    import jax.numpy as jnp
    from bayeseor_jax.model.observation.image_geometry import build_image_geometry

    jax.config.update("jax_enable_x64", True)
    frequencies_hz = 159e6 + jnp.arange(4) * 200e3
    image = build_image_geometry(
        fov_eor=20.0, fov_fg=180.0, nside=8,
        n_t=2, d_t=10.0, jd_center=2461040.5,
    )
    lmn = image.lmn_fg
    az_za = image.az_za_fg
    n_pixel = lmn.shape[-1]
    assert lmn.shape == (2, 3, n_pixel)
    ```

    The helper constructs shared geometry for both sky components; here we use only its foreground pixels. The low resolution keeps the example lightweight, not suitable for production analysis. Normal setup checks the resolution against the observation.

## 2. Reconstruct each pixel's spectrum

The spectral operator, $\mathbf Q_z$, turns each pixel's amplitudes into its
brightness at every observing frequency. The spectrum is a constant plus a set of
power laws, whose slopes are shared across the sky and whose amplitudes are free
in each pixel.

=== "Maths"

    $$
    I_{{\rm FG},p}(\nu_f)=\sum_g Q_{fg}q_{g,p},
    \qquad
    Q_{f0}=1,\quad
    Q_{fg}=\left(\frac{\nu_f}{\nu_0}\right)^{-\beta_g}\quad(g>0).
    $$

    The full spectral operator is $\mathbf Q_z=\mathbf Q\otimes\mathbf I_{N_{\rm pix}}$ with component-major coefficient ordering. The builder uses the first observing channel as $\nu_0$.

=== "Code"

    [build_foreground_spectral_operator](api/operators.md#bayeseor_jax.model.operators.build_foreground_spectral_operator) takes frequencies in **MHz**, a reference frequency and signed power-law exponents. It returns `(n_frequency, n_component)`.

    ```python
    from bayeseor_jax.model.operators import build_foreground_spectral_operator

    frequencies_mhz = frequencies_hz / 1e6
    Q = build_foreground_spectral_operator(
        frequencies_mhz,
        reference_frequency_mhz=float(frequencies_mhz[0]),
        spectral_indices=(-2.63, -2.82),
    )
    # Excite one pixel's first power-law component to follow its basis response.
    q = jnp.zeros((Q.shape[1], n_pixel), dtype=jnp.float64)
    pixel = int(jnp.argmax(lmn[image.time_ind_center, 2, :]))
    q = q.at[1, pixel].set(1.0)
    brightness = Q @ q
    assert Q.shape == (4, 3)
    assert brightness.shape == (4, n_pixel)
    ```

    This unit amplitude illustrates a basis response, not a realistic foreground realisation. The operator preserves the supplied brightness units. In run configuration, `beta_fg: [2.63, 2.82]` uses positive slopes; the builder negates them before calling this function.

## 3. Weight the brightness by the primary beam

The primary beam, $\mathbf P$, weights each pixel by its direction and frequency.
Its response is zero at and below the horizon, so pixels stop contributing as
they set. See the [primary beam](instrument.md#primary-beam) for the supported
models.

=== "Maths"

    $$
    I^{\rm beam}_{{\rm FG},tfp}
    =P_{tfp}I_{{\rm FG},p}(\nu_f).
    $$

    The weights act independently on each image pixel. There is no need to allocate a dense diagonal beam matrix.

=== "Code"

    [Beam_Model.get_beam_vals](api/data.md#bayeseor_jax.model.instrument.beam.Beam_Model.get_beam_vals) accepts azimuth/zenith-angle coordinates and frequencies in **Hz**. Reorder its `(frequency, time, pixel)` output for the forward operators.

    ```python
    from bayeseor_jax.model.instrument.beam import Beam_Model

    beam_model = Beam_Model(beam_type="airy", diam=14.2)
    beam = beam_model.get_beam_vals(az_za, frequencies_hz).transpose(1, 0, 2)
    beam_weighted_brightness = beam * brightness[None, :, :]
    assert beam_weighted_brightness.shape == (2, 4, n_pixel)
    assert bool(jnp.all(beam[jnp.broadcast_to(
        az_za[:, 1:2, :] >= jnp.pi / 2, beam.shape,
    )] == 0))
    ```

    See [instrument configuration](instrument.md#primary-beam) for the other analytic profiles and interpolated beam inputs.

## 4. Add the pixel contributions to each visibility

Finally, $\mathbf F^{-1}$ applies the full three-dimensional fringe response of
each baseline and sums over the pixels. Keeping the $w$ term retains the
wide-field geometry, including directions close to the horizon. As in the
[EoR model](eor.md), the beam and fringe are combined in the code into
$\mathbf R = \mathbf F^{-1}\mathbf P$, which already includes the pixel solid
angle.

=== "Maths"

    $$
    \delta m_{{\rm FG},tfbp}
    =\underbrace{\Delta\Omega\,
    e^{+2\pi i\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)}
    P_{tfp}}_{R_{tfbp}}I_{{\rm FG},p}(\nu_f),
    \qquad
    m_{{\rm FG},tfb}=\sum_p\delta m_{{\rm FG},tfbp}.
    $$

    Baselines $\boldsymbol u_b$ are expressed in wavelengths. The positive fringe sign follows the code's `ant2 - ant1` convention, as implemented in [build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator).

=== "Code"

    [build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator) returns the beam-weighted response with shape `(time, frequency, baseline, pixel)`.

    ```python
    from bayeseor_jax.model.operators import build_sky_to_visibility_operator

    baselines_m = jnp.array([[10.0, -10.0], [0.0, 0.0], [0.0, 0.0]])
    uvw_lambda = frequencies_hz[:, None, None] / 299792458.0 * baselines_m
    R = build_sky_to_visibility_operator(
        lmn, uvw_lambda, beam, image.pixel_area_sr,
    )
    # Use intrinsic brightness because R already applies the beam.
    model_by_step = jnp.einsum("tfbp,fp->tfb", R, brightness)
    assert model_by_step.shape == (2, 4, 2)
    ```

## 5. Put the steps together

Combining these gives $\mathbf T_{\rm FG}$, whose columns describe how each
spectral amplitude in each pixel contributes to every visibility.

=== "Maths"

    Combining the steps, the contribution of every amplitude $q_{g,p}$ to the
    visibility measured by baseline $b$ at frequency $\nu_f$ and time $t$ is

    $$
    m_{{\rm FG},tfb}
    = \sum_{p}
    \Delta\Omega\,
    \exp\!\left[+2\pi i\,\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)\right]
    P_{tfp}
    \left[q_{0,p} + \sum_{g\geq1} q_{g,p}
    \left(\frac{\nu_f}{\nu_0}\right)^{-\beta_g}\right].
    $$

    Stacking over times, frequencies and baselines gives
    $\boldsymbol m_{\rm FG} = \mathbf F^{-1}\mathbf P\mathbf Q_z\boldsymbol q
    \equiv \mathbf T_{\rm FG}\boldsymbol q$. Rows are ordered by time, frequency
    and baseline, with baseline varying fastest; columns are ordered by spectral
    component, then pixel.

=== "Code"

    [build_T_fg_core](api/operators.md#bayeseor_jax.model.operators.build_T_fg_core) combines the response and spectral factors. [T_fg](api/operators.md#bayeseor_jax.model.operators.T_fg) also constructs the response directly from the supplied geometry and beam arrays.

    ```python
    import numpy as np
    from bayeseor_jax.model.operators import build_T_fg_core

    T = build_T_fg_core(R, Q)
    model = T @ q.reshape(-1)
    assert T.shape == (16, 3 * n_pixel)
    np.testing.assert_allclose(
        model, model_by_step.reshape(-1), rtol=1e-10, atol=1e-15,
    )
    ```

    For disk-backed construction, [T_fg_build_save](api/construction.md#bayeseor_jax.model.construction.operator_construction.T_fg_build_save) evaluates the same mapping in CPU time-frequency blocks and writes `T_fg.npy`. See the [construction workflow](forward-model.md) for execution and storage.

## Configuration

`fov_fg`
:   The full angular diameter of the foreground model field.

    - **Default:** `fov_eor`.
    - **Units:** degrees.
    - **Note:** bright foregrounds remain visible through the beam sidelobes, so
      this is usually larger than `fov_eor`. Use `180` to model the full sky to
      the horizon.

`beta_fg`
:   The power-law spectral indices, $\beta_g$, shared across the sky.

    - **Default:** `[2.63, 2.82]`, for Galactic synchrotron and extragalactic
      sources.
    - **Sign:** positive values give spectra falling as
      $(\nu/\nu_0)^{-\beta_g}$.
    - **Note:** together with the constant term, this sets the number of spectral
      components per pixel, $1 + N_\beta$.

`nside`
:   The HEALPix resolution, shared with the [EoR model](eor.md#configuration).

See the [configuration reference](configuration.md) and the configuration in
`examples/eor_fgs/` for a complete run.
