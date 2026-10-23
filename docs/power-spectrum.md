# Δ² parameterisation

A detection of the 21-cm signal concerns the power on each spatial scale, not a
particular realisation of the field. BayesEoR therefore treats the EoR
coefficients as a statistically homogeneous and isotropic Gaussian field, whose
variance is set by the dimensionless power spectrum, $\Delta^2_{21}(k)$, in a set
of spherical $k$ bins. These bin amplitudes are the hyperparameters of the model,
and the only parameters explored by the sampler.

## 1. Convert the Fourier modes to wavenumbers

Each [EoR coefficient](eor.md) belongs to a mode $(u, v, \eta)$, which corresponds
to a comoving wavenumber at the central redshift of the observation.

=== "Maths"

    $$
    k_x=\frac{2\pi u}{D_{\rm M}(z_{\rm c})},\qquad
    k_y=\frac{2\pi v}{D_{\rm M}(z_{\rm c})},\qquad
    k_z=\frac{2\pi\eta}{|\mathrm d\chi/\mathrm d\nu|},
    $$

    where $D_{\rm M}$ is the transverse comoving distance and
    $\mathrm d\chi/\mathrm d\nu$ the comoving distance per unit frequency. Each mode
    has magnitude $k=(k_x^2+k_y^2+k_z^2)^{1/2}$.

=== "Code"

    `generate_k_modes_in_physical_coordinates` returns the wavenumber magnitude of
    each mode, in the same order as the columns of $\mathbf T_{21}$. This continues
    the example on the [EoR model](eor.md) page.

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
    ```

## 2. Group the modes into spherical bins

Modes with similar $|k|$ are grouped into spherical shells, each with a single
power-spectrum amplitude. Only shells containing at least one mode are kept, and
these define the sampled parameters.

=== "Maths"

    The shell edges are geometric. With $\Delta k_\parallel = 2\pi/L_\parallel$ for a
    line-of-sight box length $L_\parallel$, the upper edges are

    $$
    k^{\rm high}_s = 2\,\Delta k_\parallel\,g^{\,s},\qquad g = 1.35,
    $$

    and mode $i$ belongs to bin $s(i)$ if
    $k^{\rm high}_{s-1} < k_i \leq k^{\rm high}_s$. Each bin is represented by the
    mean $k_s$ of its modes, and assigned the amplitude

    $$
    \varphi_s \equiv \Delta^2_{21}(k_s)=\frac{k_s^3}{2\pi^2}P_{21}(k_s).
    $$

=== "Code"

    `generate_k_cube_model_spherical_binning` assigns each mode to a bin. Setup does
    this automatically and writes the result to the output directory:

    - `k-vals.txt`: the representative $k_s$ of each bin.
    - `k-vals-bins.txt`: the bin edges.
    - `k-vals-nsamples.txt`: the number of modes in each bin.

## 3. Set the coefficient prior

Taking distinct Fourier modes to be uncorrelated, the prior on the EoR coefficients
is diagonal, with each variance set by the amplitude of its bin.

=== "Maths"

    $$
    \langle a_i a_j^*\rangle
    =\left[\mathbf\Phi(\boldsymbol\varphi)\right]_{ij}
    \propto \varphi_{s(i)}\,\delta_{ij}.
    $$

    The constant of proportionality is the Fourier-volume normalisation, which
    converts $\Delta^2_{21}(k_s)$ into the variance of a single coefficient. Each
    amplitude is sampled through

    $$
    \rho_s = \log_{10}\!\left(\frac{\varphi_s}{\mathrm{mK}^2}\right),
    $$

    with an independent uniform prior on each $\rho_s$.

=== "Code"

    [PhiInverse](api/inference.md#bayeseor_jax.inference.phi.PhiInverse) applies the
    normalisation and returns the diagonal prior precision of the joint coefficients
    for a given set of amplitudes. Its foreground block is fixed; see
    [analytical marginalisation](marginalisation.md).

    ```python
    from bayeseor_jax import run_setup

    setup = run_setup(config)
    # rho: one log10 amplitude per bin
    precision = setup.phi_inverse_class.evaluate(10.0 ** rho)
    ```

## Configuration

`priors`
:   The prior range on each power-spectrum amplitude.

    - **Format:** one `[min, max]` pair per non-empty $k$ bin, in
      $\rho_s = \log_{10}(\Delta^2_{21}/\mathrm{mK}^2)$.
    - **Required:** for any model including the EoR. Setup reports an error if the
      number of pairs does not match the number of bins.

`uniform_priors`
:   Selects bins whose prior is uniform in linear, rather than logarithmic, power.

    - **Format:** one `0` or `1` per bin.
    - **Default:** all `0`, logarithmic.
    - **Note:** the sampler still works in $\rho_s$; a Jacobian correction is added
      for the selected bins.
