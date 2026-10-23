# Data

`get_vis_data` loads a visibility file or preprocessed vector and returns the data
vector, noise information and observation axes in the
[analysis convention](../data.md#data-vector-ordering). The instrument enters
through its baselines, `InstrumentModel`, and its primary beam, `Beam_Model`.

See the [Data guide](../visibilities.md) for the supported inputs and settings.

## Visibilities

::: bayeseor_jax.data.vis.get_vis_data
    options:
      heading_level: 3

::: bayeseor_jax.data.vis.preprocess_uvdata
    options:
      heading_level: 3

## Instrument

::: bayeseor_jax.model.instrument.baselines.InstrumentModel
    options:
      heading_level: 3

::: bayeseor_jax.model.instrument.beam.Beam_Model
    options:
      heading_level: 3

## Simulated noise

::: bayeseor_jax.data.noise.generate_gaussian_noise
    options:
      heading_level: 3
