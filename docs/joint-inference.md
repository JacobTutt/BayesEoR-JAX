# Parameterisation and marginalisation

Given a physical model for the measured visibilities, inference connects its
coefficients to the quantities we wish to recover. BayesEoR assigns zero-mean Gaussian
priors to the foreground and cosmological coefficients, propagates their
uncertainty through analytical marginalisation, and samples the power-spectrum
amplitudes that determine the cosmological prior.

The steps below describe the joint model with fixed observational noise.
The [running guide](inference.md) covers component selection, optional noise
fitting and execution; the [analysis guide](outputs-analysis.md) describes the
resulting posterior and reconstruction products.

## 1. Combine the physical models

The [cosmological](eor.md) and [foreground](foregrounds.md) visibility
contributions form a single linear model:

$$
\boldsymbol d=\boldsymbol m+\boldsymbol n,
\qquad
\boldsymbol m
=\mathbf T_{21}\boldsymbol a+\mathbf T'_{\rm FG}\boldsymbol q'
=\underbrace{\begin{bmatrix}\mathbf T_{21}&\mathbf T'_{\rm FG}\end{bmatrix}}_{\mathbf T}
\underbrace{\begin{pmatrix}\boldsymbol a\\\boldsymbol q'\end{pmatrix}}_{\boldsymbol b}.
$$

Here, $\mathbf T'_{\rm FG}$ denotes either the original foreground operator
or its [KL reduced representation](kl-reduced-basis.md), with corresponding
coefficients $\boldsymbol q'$. This choice changes the coefficient space and
its prior, but not the form of the inference calculation. Foreground-only and
cosmological-only models retain just the relevant block.

## 2. Parameterise the coefficient prior

Taking the cosmological and foreground contributions to be independent a priori
gives the block-diagonal coefficient covariance

$$
\mathbf\Psi(\boldsymbol\varphi)
=\begin{pmatrix}
\mathbf\Phi_{21}(\boldsymbol\varphi)&\mathbf 0\\
\mathbf 0&\mathbf\Sigma'_{\rm FG}
\end{pmatrix}.
$$

For the original foreground representation,
$\mathbf\Sigma'_{\rm FG}=\mathbf\Sigma_q$ contains the physical coefficient
variances. In the KL representation, that prior weighting is already absorbed
into the reduced operator: the retained coefficients have a standard-normal
prior, $\mathbf\Sigma'_{\rm FG}=\mathbf I$.

The cosmological block is determined by the spherical-bin power-spectrum
amplitudes, $\varphi_s=\Delta^2_{21}(k_s)$. We parameterise these through

$$
\rho_s=\log_{10}\!\left(\frac{\varphi_s}{\mathrm{mK}^2}\right).
$$

The configured `priors` specify bounds in these logarithmic coordinates.
[PhiInverse](api/inference.md#bayeseor_jax.inference.phi.PhiInverse)
converts the power amplitudes to coefficient variances using the Fourier-volume
normalisation and assembles the joint prior precision. Its name `Phi` refers
to the full coefficient prior, denoted $\mathbf\Psi$ here to distinguish it
from the cosmological block.

## 3. Marginalise the individual coefficients

The target is the power spectrum, rather than a particular realisation of
every foreground and cosmological coefficient. Their uncertainty is included
by integrating over the coefficient prior:

$$
\mathcal L(\boldsymbol d\mid\boldsymbol\varphi,\mathcal M)
=\int
\mathcal L(\boldsymbol d\mid\boldsymbol b,\mathcal M)
\pi(\boldsymbol b\mid\boldsymbol\varphi,\mathcal M)
\,\mathrm d\boldsymbol b.
$$

The linear model and Gaussian prior and noise make this integral analytic.
Equivalently, the model uncertainty enters visibility space through

$$
\mathbf C(\boldsymbol\varphi)
=\mathbf N+\mathbf T\mathbf\Psi(\boldsymbol\varphi)\mathbf T^\dagger.
$$

The implementation evaluates the equivalent coefficient-space expression,
avoiding the construction of this full data-space covariance at every call.
In either form, marginalisation propagates the uncertainty and degeneracies
between the two contributions into the power-spectrum posterior.

## 4. Evaluate the marginal likelihood

For fixed operators and noise, setup caches

$$
\mathbf G=\mathbf T^\dagger\mathbf N^{-1}\mathbf T,
\qquad
\overline{\boldsymbol d}=\mathbf T^\dagger\mathbf N^{-1}\boldsymbol d,
\qquad
Q=\boldsymbol d^\dagger\mathbf N^{-1}\boldsymbol d.
$$

Each proposed power spectrum changes the prior precision, giving the system

$$
\mathbf\Sigma(\boldsymbol\varphi)
=\mathbf G+\mathbf\Psi^{-1}(\boldsymbol\varphi),
\qquad
\mathbf\Sigma(\boldsymbol\varphi)\widehat{\boldsymbol b}
=\overline{\boldsymbol d}.
$$

One Cholesky factorisation supplies both the solve and the log determinant.
For the visibility convention used by BayesEoR, the normalised log likelihood is

$$
\begin{aligned}
\ln\mathcal L
={}&-\frac12\left[
\ln\det\mathbf\Sigma+\ln\det\mathbf\Psi
-\overline{\boldsymbol d}^{\dagger}\widehat{\boldsymbol b}\right]
+c_{\rm fixed},\\
c_{\rm fixed}
={}&-\frac12\left[Q+\ln\det\mathbf N+N_{\rm d}\ln\pi\right].
\end{aligned}
$$

The data vector includes each measured visibility and its conjugate; these
are not independent complex measurements. The factors of one half follow
this convention and the real-sky coefficient constraints, as explained in
[complex storage and normalisation](inference.md#complex-storage-and-the-factors-of-one-half).
The implemented fixed-noise target omits $c_{\rm fixed}$, which cancels in
model comparisons only when the same data and noise conventions are used.
See [evidence normalisation](inference.md#evidence-normalisation) before
comparing reported evidences.

The dense factorisation is the principal repeated cost. KL reduction decreases
the size of this system while controlling the discarded foreground uncertainty
relative to the observational noise.

## 5. Infer the power spectrum and evaluate evidence

With the coefficients marginalised, the remaining posterior is

$$
P(\boldsymbol\rho\mid\boldsymbol d,\mathcal M)
=\frac{\mathcal L(\boldsymbol d\mid\boldsymbol\varphi(\boldsymbol\rho),\mathcal M)
\pi(\boldsymbol\rho\mid\mathcal M)}{\mathcal Z_{\mathcal M}},
$$

where

$$
\mathcal Z_{\mathcal M}
=\int\mathcal L(\boldsymbol d\mid\boldsymbol\varphi(\boldsymbol\rho),\mathcal M)
\pi(\boldsymbol\rho\mid\mathcal M)\,\mathrm d\boldsymbol\rho.
$$

[MultiNest](inference.md#multinest) explores this posterior while estimating
the evidence. For a foreground-only model with fixed prior and noise, no
hyperparameters remain: its fully marginalised likelihood is its evidence,
and no sampling is required.

Evidence ratios compare support for alternative models, including the
foreground-only and joint hypotheses. Applying the same comparison to
signal-free validation data can reveal whether an additional cosmological
component is instead absorbing unmodelled foreground structure.

## 6. Reconstruct conditional model visibilities

Analytical marginalisation does not prevent reconstruction. At a given power
spectrum, the solve above yields the conditional posterior mean and MAP
coefficients, with predicted visibilities

$$
\widehat{\boldsymbol b}(\boldsymbol\varphi)
=\mathbf\Sigma^{-1}(\boldsymbol\varphi)\overline{\boldsymbol d},
\qquad
\widehat{\boldsymbol m}(\boldsymbol\varphi)
=\mathbf T\widehat{\boldsymbol b}(\boldsymbol\varphi).
$$

These reconstructions support residual and component diagnostics; they do not
replace the marginalised power-spectrum posterior. Continue to
[running inference](inference.md) for configuration and execution, or
[posterior analysis](outputs-analysis.md) for inspecting the results.
