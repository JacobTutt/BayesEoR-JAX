# Model

[![Combined forward model: the EoR and foreground components enter their respective Fourier and image spaces, then propagate through the shared beam and fringe response into modelled data.](assets/diagrams/joint-model-workflow.png#only-light)](assets/diagrams/joint-model-workflow.png)
[![Combined forward model: the EoR and foreground components enter their respective Fourier and image spaces, then propagate through the shared beam and fringe response into modelled data.](assets/diagrams/joint-model-workflow-dark.png#only-dark)](assets/diagrams/joint-model-workflow-dark.png)

## The interferometric measurement equation

The forward model connects the sky brightness to the
[measured visibilities](visibilities.md). For a calibrated, unphased observation
and a scalar primary-beam response, the predicted visibility is

$$
m_{ij}(\nu,t)
=\int_{\Omega_{\rm vis}(t)}
P(\hat{\boldsymbol s},\nu)\,
I(\hat{\boldsymbol s},\nu,t)\,
\exp\!\left[-2\pi i\boldsymbol u_{ij}(\nu)
\cdot\hat{\boldsymbol s}\right]\,\mathrm d\Omega.
$$

Here $I=I_{21}+I_{\rm FG}$ is the sky brightness, $P$ is the primary-beam power
response, and $\hat{\boldsymbol s}=(l,m,n)$ is a direction in the instrument's
topocentric frame. The full three-dimensional phase retains the wide-field
geometry, including emission near the horizon. Earth rotation changes the
directions of celestial sources in this frame during an observation.
The fringe sign depends on the baseline orientation convention; the numerical
operator convention is specified [below](#shared-sky-to-visibility-response).

## Cosmological and foreground contributions

The cosmological and foreground models begin in different coefficient spaces,
but share the same beam-weighted sky-to-visibility mapping:

$$
\begin{aligned}
\boldsymbol m_{21}
&= \mathbf T_{21}\boldsymbol a
 = \mathbf F^{-1}\mathbf P\mathbf F'\mathbf F_z\boldsymbol a,\\
\boldsymbol m_{\rm FG}
&= \mathbf T_{\rm FG}\boldsymbol q
 = \mathbf F^{-1}\mathbf P\mathbf Q_z\boldsymbol q.
\end{aligned}
$$

Read each product from right to left. The [EoR model](eor.md) reconstructs a
brightness field from a rectilinear Fourier grid; the
[foreground model](foregrounds.md) assigns spectral amplitudes directly to
spherical image pixels. Their visibility contributions are added:

$$
\boldsymbol m=\boldsymbol m_{21}+\boldsymbol m_{\rm FG},
\qquad
\boldsymbol d=\boldsymbol m+\boldsymbol n.
$$

The physical forward models stop at $\boldsymbol m$: a predicted visibility
vector, without a noise realisation or a likelihood evaluation.

## Shared sky-to-visibility response

For sky pixel $p$, frequency channel $f$, integration $t$ and baseline $b$,
the response implemented by the code is

$$
R_{tfbp}
=
\underbrace{\Delta\Omega
\exp\!\left[+2\pi i\,\boldsymbol u_b(\nu_f)
\cdot\hat{\boldsymbol s}_p(t)\right]}_{(F^{-1})_{tfbp}}
\underbrace{P(\hat{\boldsymbol s}_p(t),\nu_f)}_{P_{tfp}}.
$$

The [image geometry](api/geometry.md#bayeseor_jax.model.observation.image_geometry.build_image_geometry)
provides each celestial HEALPix pixel's time-dependent topocentric direction
$\hat{\boldsymbol s}_p(t)=(l_p,m_p,n_p)$.
The baseline coordinate $\boldsymbol u_b(\nu_f)$ is measured in wavelengths,
and $\Delta\Omega$ is the pixel solid angle. The
[primary-beam model](api/geometry.md#bayeseor_jax.model.instrument.beam.Beam_Model)
provides power-response weights, with zero gain at or below the horizon.

The positive fringe exponent follows the baseline convention used by
[build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator).
$\mathbf F^{-1}$ denotes the Fourier mapping to visibilities, not a matrix
inverse computed numerically. The code combines $\mathbf F^{-1}\mathbf P$
into $\mathbf R$ rather than allocating a dense diagonal beam matrix.

For either component, a brightness field $I_p(\nu_f,t)$ gives

$$
m_{tfb}=\sum_p R_{tfbp}I_p(\nu_f,t).
$$

The pixel integration weight is already included in $\mathbf R$ and must not
be applied again. The numerical operators do not perform brightness-unit
conversions: coefficient, data and prior units must be consistent.

## Operator map

| Step | Operator | Implementation |
| --- | --- | --- |
| Line-of-sight Fourier modes to frequency | $\mathbf F_z$ | [build_spectral_fourier_operator](api/operators.md#bayeseor_jax.model.operators.build_spectral_fourier_operator) |
| Transverse Fourier modes to image pixels | $\mathbf F'$ | [build_transverse_fourier_operator](api/operators.md#bayeseor_jax.model.operators.build_transverse_fourier_operator) |
| Foreground spectral amplitudes to brightness | $\mathbf Q_z$ | [build_foreground_spectral_operator](api/operators.md#bayeseor_jax.model.operators.build_foreground_spectral_operator) |
| Beam-weighted image to visibilities | $\mathbf R=\mathbf F^{-1}\mathbf P$ | [build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator) |
| Cosmological coefficients to visibilities | $\mathbf T_{21}$ | [T_21](api/operators.md#bayeseor_jax.model.operators.T_21) |
| Foreground coefficients to visibilities | $\mathbf T_{\rm FG}$ | [T_fg](api/operators.md#bayeseor_jax.model.operators.T_fg) |

## Geometry, construction and reduction

Both model fields use the union of celestial pixels entering the specified
angular extent during the observation. The cosmological and foreground fields
can have different extents but use the configured HEALPix resolution.
The cosmological transverse Fourier representation is a tangent-plane model;
the subsequent interferometric response retains all three direction cosines.

The operator functions describe the mathematical mappings for supplied array
blocks. The [construction workflow](forward-model.md) handles CPU block
assignment, disk-backed storage and joint assembly separately.
Saved visibility rows are ordered by **time, frequency, baseline**, with baseline
varying fastest.

The [KL Reduced Basis](kl-reduced-basis.md) changes the representation of
foreground uncertainty, not the underlying measurement equation.

Continue to [Inference](joint-inference.md) to combine these operators with
coefficient priors and analytically marginalise their amplitudes.
