# Maximum a posteriori

Marginalisation does not discard the sky coefficients. For a fixed prior, either
at a chosen power spectrum or for the fixed foreground prior alone, their posterior
is Gaussian, and its mean, which is also its maximum, follows from the same linear
system used by the [marginalised likelihood](marginalisation.md).

## Conditional coefficients and visibilities

At a given power spectrum, the solve that evaluates the likelihood also gives the
maximum a posteriori (MAP) coefficients, and hence the model visibilities of each
component.

=== "Maths"

    $$
    \widehat{\boldsymbol b}(\boldsymbol\varphi)
    =\mathbf\Sigma^{-1}(\boldsymbol\varphi)\,\overline{\boldsymbol d},
    \qquad
    \widehat{\boldsymbol m}(\boldsymbol\varphi)
    =\mathbf T\,\widehat{\boldsymbol b}(\boldsymbol\varphi)
    =\mathbf T_{21}\widehat{\boldsymbol a}
    +\mathbf T'_{\rm FG}\widehat{\boldsymbol q}'.
    $$

    The conditional posterior covariance of the coefficients is
    $\mathbf\Sigma^{-1}(\boldsymbol\varphi)$.

=== "Code"

    [MaximumAPosteriori](api/inference.md#bayeseor_jax.inference.maxaposteriori.MaximumAPosteriori)
    reuses the likelihood products from setup:

    ```python
    import numpy as np
    from bayeseor_jax import MaximumAPosteriori

    likelihood = setup.likelihood_class
    map_solver = MaximumAPosteriori(
        t_ninv_t=likelihood.t_ninv_t, dbar=likelihood.dbar,
        d_ninv_d=likelihood.d_ninv_d, n_data=likelihood.n_data,
        phi_inv_class=setup.phi_inverse_class, k_vals=setup.k_vals,
        n_power_params=likelihood.n_power_params,
    )
    b_hat, log_post = map_solver.map_estimate(dmps=dmps)   # Δ² in mK², per bin

    T = np.load("matrices/T.npy", mmap_mode="r")
    m_hat = T @ np.asarray(b_hat)
    ```

    The first $N_{21}$ entries of `b_hat` are the EoR coefficients, and the rest the
    foreground coefficients.

These reconstructions are useful for inspecting residuals,
$\boldsymbol d - \widehat{\boldsymbol m}$, and the separate component models. They
are conditional on the chosen power spectrum, and do not replace the marginalised
power-spectrum posterior.

## Foreground-only evidence

A foreground-only model has no sampled parameters: its prior is fixed, so the
marginalised likelihood is itself the Bayesian evidence, and is evaluated in a
single solve.

=== "Maths"

    $$
    \mathcal Z_{\rm FG}
    =\int\mathcal L(\boldsymbol d\mid\boldsymbol q')\,
    \pi(\boldsymbol q')\,\mathrm d\boldsymbol q',
    \qquad
    \widehat{\boldsymbol q}'
    =\mathbf\Sigma_{\rm FG}^{-1}\,\overline{\boldsymbol d}_{\rm FG},
    $$

    with $\mathbf\Sigma_{\rm FG}$ and $\overline{\boldsymbol d}_{\rm FG}$ formed from
    $\mathbf T'_{\rm FG}$ alone. This evidence is the null hypothesis against which
    the joint model is compared.

=== "Code"

    With `model: fg`, [running inference](inference.md) skips sampling and writes:

    - `summary.json`: the evidence, with `sampler: analytic` and `logZerr: 0`.
    - `map_coefficients.npy`: the MAP foreground coefficients,
      $\widehat{\boldsymbol q}'$.
