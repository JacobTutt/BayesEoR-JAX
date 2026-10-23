# API reference

## Setup

Parse a YAML or command-line configuration into a typed `RunConfig`, and turn it
into the likelihood, prior and metadata used for sampling.

[Setup API](setup.md)

---

## Data

Load and prepare visibilities, and describe the instrument that measured them.

$$
\boldsymbol d_{tf}
=\begin{pmatrix}\boldsymbol v_{tf}\\\boldsymbol v_{tf}^{*}\end{pmatrix}
$$

[Data API](data.md)

---

## Sky and Fourier grids

Define the sky pixels, the cosmological Fourier grid and the conversion of its
modes to wavenumbers and power-spectrum bins.

$$
\boldsymbol k = \left(\frac{2\pi u}{D_{\rm M}},\ \frac{2\pi v}{D_{\rm M}},\
\frac{2\pi\eta}{|\mathrm d\chi/\mathrm d\nu|}\right)
$$

[Sky and Fourier grids API](grids.md)

---

## Operators

Map the EoR and foreground coefficients into visibilities, as JAX arrays.

$$
\mathbf T_{21}=\mathbf F^{-1}\mathbf P\mathbf F'\mathbf F_z,
\qquad
\mathbf T_{\rm FG}=\mathbf F^{-1}\mathbf P\mathbf Q_z
$$

[Operators API](operators.md)

---

## Construction and KL reduction

Build the operators on disk in CPU blocks, compress the foregrounds into a KL
reduced basis, and precompute the likelihood products.

$$
\mathbf A=\mathbf N^{-1/2}\mathbf T\,\mathbf\Sigma^{1/2}=\mathbf U\mathbf S\mathbf V^\dagger,
\qquad
\mathbf T_r=\mathbf N^{1/2}\mathbf U_r\mathbf S_r
$$

[Construction API](construction.md)

---

## Inference

Evaluate the marginalised likelihood and MAP coefficients, and sample the
power-spectrum posterior.

$$
\mathbf\Sigma(\boldsymbol\varphi)=\mathbf T^\dagger\mathbf N^{-1}\mathbf T
+\mathbf\Psi^{-1}(\boldsymbol\varphi),
\qquad
\widehat{\boldsymbol b}=\mathbf\Sigma^{-1}\mathbf T^\dagger\mathbf N^{-1}\boldsymbol d
$$

[Inference API](inference.md)

---

## Analysis

Read weighted MultiNest chains, summarise each $k$ bin and plot the results.

[Analysis API](analysis.md)
