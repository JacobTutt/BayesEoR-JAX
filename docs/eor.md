# EoR model

The cosmological 21-cm field is parameterised on a rectilinear
$(u,v,\eta)$ Fourier grid. This links the model coefficients naturally to
the power spectrum while resolving the line-of-sight structure that helps
distinguish the signal from spectrally smooth foreground emission.

[![EoR-only forward model: Fourier coefficients pass through F_z, F prime, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/eor-workflow.png#only-light)](assets/diagrams/eor-workflow.png)
[![EoR-only forward model: Fourier coefficients pass through F_z, F prime, the primary beam and the three-dimensional fringe response to modelled data.](assets/diagrams/eor-workflow-dark.png#only-dark)](assets/diagrams/eor-workflow-dark.png)

*The component-only schematic ends at modelled visibilities; inference is a
separate stage. The thumbnails illustrate the spaces, not a numerical evaluation
of one shared realisation.*

The complete mapping is

$$
\boldsymbol m_{21}
=\underbrace{\mathbf F^{-1}\mathbf P\mathbf F'\mathbf F_z}_{\mathbf T_{21}}
\boldsymbol a.
$$

## 1. Define the Fourier coefficients

The coefficients $a_{he}$ are indexed by transverse pair
$(u_h,v_h)$ and line-of-sight mode $\eta_e$, the Fourier dual of frequency.
[build_fourier_geometry](api/geometry.md#bayeseor_jax.model.cosmology.fourier_geometry.build_fourier_geometry)
constructs these coordinates and their integration weights.

The default transverse spacings are set by the model's angular dimensions,
while $\Delta\eta=B^{-1}$ for bandwidth $B$. The transverse monopole
$(u,v)=(0,0)$ and the $\eta=0$ plane are excluded, leaving

$$
N_{21,\mathrm{param}}=(N_uN_v-1)(N_\eta-1).
$$

The angular grid describes the intrinsic field in a local tangent plane.
Its image pixels are subsequently propagated through the full three-dimensional
instrument response.

## 2. Map line-of-sight modes to frequency: $\mathbf F_z$

At each transverse coordinate, reconstruct the spectral dependence:

$$
S_h(\nu_f)=\sum_e(F_z)_{fe}a_{he},
\qquad
(F_z)_{fe}
=\Delta\eta\,
\exp[-2\pi i(\nu_f-\nu_{\rm ref})\eta_e].
$$

[build_spectral_fourier_operator](api/operators.md#bayeseor_jax.model.operators.build_spectral_fourier_operator)
evaluates these factors. The configured builder uses the middle observing
channel as $\nu_{\rm ref}$.

## 3. Map transverse modes to image pixels: $\mathbf F'$

The transverse transform reconstructs the brightness at each cosmological image pixel:

$$
I_{21,p}(\nu_f)
=\sum_h F'_{ph}S_h(\nu_f),
\qquad
F'_{ph}
=\Delta u\,\Delta v\,
\exp[-2\pi i(u_hl_p+v_hm_p)].
$$

[build_transverse_fourier_operator](api/operators.md#bayeseor_jax.model.operators.build_transverse_fourier_operator)
uses pixel direction cosines at the reference integration. The image geometry
then follows those fixed celestial pixels through the drift scan.

## 4. Apply the beam and fringe response: $\mathbf F^{-1}\mathbf P$

Weight the reconstructed field by the primary beam and sum its contributions to
each measured visibility:

$$
m_{21,tfb}
=\sum_p
\underbrace{(F^{-1})_{tfbp}P_{tfp}}_{R_{tfbp}}
I_{21,p}(\nu_f).
$$

The [shared response](model.md#shared-sky-to-visibility-response) includes
the pixel solid angle, chromatic beam and full three-dimensional fringe phase.
[build_sky_to_visibility_operator](api/operators.md#bayeseor_jax.model.operators.build_sky_to_visibility_operator)
constructs $\mathbf R$ on the cosmological image support.

## 5. Assemble the cosmological operator: $\mathbf T_{21}$

Combining the preceding steps gives the visibility contribution of one coefficient:

$$
(T_{21})_{(tfb),(he)}
=\left[\sum_p R_{tfbp}F'_{ph}\right](F_z)_{fe},
\qquad
\boldsymbol m_{21}=\mathbf T_{21}\boldsymbol a.
$$

[build_T_21_core](api/operators.md#bayeseor_jax.model.operators.build_T_21_core)
performs this contraction; [T_21](api/operators.md#bayeseor_jax.model.operators.T_21)
also constructs $\mathbf R$ from supplied geometry and beam arrays.
Columns are ordered by transverse mode, then $\eta$ varying fastest.

For disk-backed construction,
[T_21_build_save](api/construction.md#bayeseor_jax.model.construction.operator_construction.T_21_build_save)
builds CPU time-frequency blocks and writes **T_21.npy**.
This changes how the calculation is executed, not the operator being evaluated.

## Power-spectrum parameterisation

The Fourier coordinates map to comoving wavenumbers at a reference redshift:

$$
k_x=\frac{2\pi u}{D_M},\qquad
k_y=\frac{2\pi v}{D_M},\qquad
k_z=\frac{2\pi\eta}{|d\chi/d\nu|}.
$$

Modes are grouped into spherical bins in $k=(k_x^2+k_y^2+k_z^2)^{1/2}$.
The quantity inferred in each bin is

$$
\varphi_s=\Delta^2_{21}(k_s)=\frac{k_s^3}{2\pi^2}P_{21}(k_s).
$$

These amplitudes determine the Gaussian coefficient prior rather than the
individual coefficient values. The implementation includes the Fourier-volume
normalisation in
[PhiInverse](api/inference.md#bayeseor_jax.inference.phi.PhiInverse);
$\varphi_s$ should not be inserted as the raw coefficient variance without
that conversion. The linear coefficients are analytically marginalised in the
[inference stage](inference.md).

## Configuration

| Setting | Role |
| --- | --- |
| `fov_eor` | Cosmological model's full angular field of view, in degrees. |
| `n_u_eor`, `n_v_eor` | Transverse grid sizes, independently inferred if omitted; undersized supplied values produce warnings. |
| `du_eor`, `dv_eor` | Optional transverse spacings; otherwise derived from the model field. |
| `n_eta` | Line-of-sight grid size; defaults to `n_freq`. |
| `nside` | Shared HEALPix resolution for the image quadrature; inferred if omitted. |
| `priors` | One `[min, max]` range per populated power-spectrum bin, in $\log_{10}(\varphi_s/\mathrm{mK}^2)$. |

The `uniform_priors` setting selects bins with a linear-uniform physical power
prior while retaining logarithmic sampler coordinates. See the
[configuration reference](configuration.md) for the full settings and
the example configurations in `test_examples/eor/` and `test_examples/eor_fgs/`.

Increasing the Fourier grid or model extent increases the number of marginalised
coefficients and the memory needed for dense likelihood products.
