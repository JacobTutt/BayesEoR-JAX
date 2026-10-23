# Analysis

`DataContainer` reads one or more MultiNest runs with their setup metadata, and
summarises the posterior on $\Delta^2_{21}(k)$ in each bin. The plotting functions
take a `DataContainer` and return Matplotlib figures and axes. See
[posterior analysis](../outputs-analysis.md) for a guide.

```python
from bayeseor_jax import DataContainer
from bayeseor_jax.analyze import plot_power_spectra_and_posteriors

dc = DataContainer(["outputs/multinest"])
fig, (power_ax, posterior_axes) = plot_power_spectra_and_posteriors(dc)
```

## Results

::: bayeseor_jax.analyze.results.DataContainer
    options:
      heading_level: 3

## Plotting

::: bayeseor_jax.analyze.plotting.plot_power_spectra_and_posteriors
    options:
      heading_level: 3

::: bayeseor_jax.analyze.plotting.plot_power_spectra
    options:
      heading_level: 3

::: bayeseor_jax.analyze.plotting.plot_posteriors
    options:
      heading_level: 3

## Unit conversions

Convert between $P(k)$, in mK² Mpc³, and
$\Delta^2(k) = k^3 P(k)/2\pi^2$, in mK².

::: bayeseor_jax.analyze.results.ps_to_dmps
    options:
      heading_level: 3

::: bayeseor_jax.analyze.results.dmps_to_ps
    options:
      heading_level: 3
