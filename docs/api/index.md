# Python API

The reference below is generated from the source docstrings. It follows the path
from observation and model configuration through visibility operators to inference.

| Reference | Role |
| --- | --- |
| [Configuration and setup](setup.md) | Parse settings and initialise an analysis. |
| [Geometry and instrument](geometry.md) | Define image and Fourier coordinates, beam response and noise. |
| [Linear operators](operators.md) | Map cosmological and foreground coefficients into visibilities. |
| [Construction and KL reduction](construction.md) | Build disk-backed operators and likelihood products. |
| [Inference](inference.md) | Evaluate the marginal likelihood, recover coefficients and sample the posterior. |
| [Data and analysis](data-analysis.md) | Load observations and inspect inference results. |

For a guided introduction, start with the [forward model](../forward-model.md)
and [inference](../inference.md) guides.
