# Foreground model

The foreground model starts with spectral amplitudes on a spherical HEALPix grid and follows each pixel through the primary beam and full three-dimensional fringe response to predict the visibilities.

[![Foreground-only forward model: image-domain coefficients pass through Q_z, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/foreground-workflow.png#only-light)](assets/diagrams/foreground-workflow.png)
[![Foreground-only forward model: image-domain coefficients pass through Q_z, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/foreground-workflow-dark.png#only-dark)](assets/diagrams/foreground-workflow-dark.png)

Here we begin directly in image space, rather than on the Fourier grid used by the [EoR model](eor.md). Let's follow one pixel's spectrum into its contribution to the visibilities, then combine the contributions across the sky. The spherical geometry lets us include emission all the way to the horizon without a tangent-plane approximation to the foreground field.

Open the **Mathematics** or **Code and example** panels for the details of each step. The Python examples run in sequence using a small illustrative setup; use the [example scripts](examples.md) for a complete analysis.

## 1. Choose the sky pixels

Start by selecting the foreground field of view. The model includes every celestial pixel that enters this field at any point during the observation, and tracks its direction relative to the instrument at each integration.

Each selected pixel carries its own set of spectral amplitudes. Expanding the field therefore adds pixels and coefficients, while keeping the same construction.

??? note "Mathematics: the image-domain coefficients"

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

??? example "Code and example: select and track the pixels"

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

The spectral operator, **Q_z**, turns the pixel amplitudes into brightness at each observing frequency. The configured model combines a constant spectrum with power laws whose slopes are shared across the sky; their amplitudes are independent for each pixel.

We only need a small frequency-by-component matrix of spectral factors. The code applies it to every pixel without constructing a large pixel-by-pixel matrix.

??? note "Mathematics: from amplitudes to brightness"

    $$
    I_{{\rm FG},p}(\nu_f)=\sum_g Q_{fg}q_{g,p},
    \qquad
    Q_{f0}=1,\quad
    Q_{fg}=\left(\frac{\nu_f}{\nu_0}\right)^{-\beta_g}\quad(g>0).
    $$

    The full spectral operator is $\mathbf Q_z=\mathbf Q\otimes\mathbf I_{N_{\rm pix}}$ with component-major coefficient ordering. The builder uses the first observing channel as $\nu_0$.

??? example "Code and example: build the spectral factors"

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

The primary beam determines how strongly each pixel contributes at each frequency and time. Its response is zero at or below the horizon, so pixels can move through the observation without contributing when they are no longer visible.

This is the same beam-weighting step used for the cosmological field, evaluated over the wider foreground pixel selection.

??? note "Mathematics: beam-weighted brightness"

    $$
    I^{\rm beam}_{{\rm FG},tfp}
    =P_{tfp}I_{{\rm FG},p}(\nu_f).
    $$

    The weights act independently on each image pixel. There is no need to allocate a dense diagonal beam matrix.

??? example "Code and example: evaluate the beam"

    [Beam_Model.get_beam_vals](api/geometry.md#bayeseor_jax.model.instrument.beam.Beam_Model.get_beam_vals) accepts azimuth/zenith-angle coordinates and frequencies in **Hz**. Reorder its `(frequency, time, pixel)` output for the forward operators.

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

For each baseline, apply the geometric fringe phase and sum the beam-weighted sky brightness. The full three-dimensional response retains the wide-field geometry, including directions close to the horizon.

As in the EoR model, the code combines the beam and fringe into **R**. This response already includes the pixel solid angle; multiplying by it again would double-count the integration weight.

??? note "Mathematics: the contribution of one pixel"

    $$
    \delta m_{{\rm FG},tfbp}
    =\underbrace{\Delta\Omega\,
    e^{+2\pi i\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)}
    P_{tfp}}_{R_{tfbp}}I_{{\rm FG},p}(\nu_f),
    \qquad
    m_{{\rm FG},tfb}=\sum_p\delta m_{{\rm FG},tfbp}.
    $$

    Baselines $\boldsymbol u_b$ are expressed in wavelengths. The positive fringe sign follows the code's `ant2 - ant1` convention, as described in the [shared response](model.md#shared-sky-to-visibility-response).

??? example "Code and example: propagate brightness to visibilities"

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

Combining these mappings gives the foreground operator, **T_fg**. Each column describes the visibility response to one spectral amplitude in one pixel. Applying it to all the coefficients reproduces the step-by-step calculation above.

??? note "Mathematics: the complete operator"

    $$
    (T_{\rm FG})_{(tfb),(gp)}=R_{tfbp}Q_{fg},
    \qquad
    \boldsymbol m_{\rm FG}=\mathbf T_{\rm FG}\boldsymbol q,
    \qquad
    \mathbf T_{\rm FG}=\mathbf F^{-1}\mathbf P\mathbf Q_z.
    $$

    Rows are ordered by time, frequency and baseline, with baseline varying fastest. Columns are ordered by spectral component, then pixel, matching the flattened coefficient array.

??? example "Code and example: assemble and check the operator"

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

The result is the modelled foreground visibility vector. Continue to the [KL Reduced Basis](kl-reduced-basis.md) or [Inference](joint-inference.md) for the next stage.
