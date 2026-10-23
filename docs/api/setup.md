# Setup

A run is described by a `RunConfig`, normally read from YAML by `BayesEoRParser`.
`run_setup` then loads the data, builds or reuses the operators, and returns a
`SetupResult` holding everything the sampler needs.

```python
from bayeseor_jax import BayesEoRParser, run_setup

config = BayesEoRParser().parse_args(["--config", "config.yaml"])
setup = run_setup(config)
```

See [construction and execution](../forward-model.md) for what setup does, and the
[configuration reference](../configuration.md) for every option.

## Main interface

::: bayeseor_jax.setup.run_setup
    options:
      heading_level: 3

::: bayeseor_jax.setup.SetupResult
    options:
      heading_level: 3

::: bayeseor_jax.configuration.BayesEoRParser
    options:
      heading_level: 3

## Configuration objects

`RunConfig` holds every setting, and `ModelConfig` the subset that defines the
physical operators. Both are frozen dataclasses: use `dataclasses.replace` to
derive a new configuration.

::: bayeseor_jax.configuration.RunConfig
    options:
      heading_level: 3

::: bayeseor_jax.configuration.ModelConfig
    options:
      heading_level: 3
