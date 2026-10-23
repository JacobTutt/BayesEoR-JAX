# Examples

Two self-contained examples demonstrate the full BayesEoR workflow on a laptop,
from loading simulated visibilities to recovering the 21-cm power spectrum. Both
use the same simulated HERA-like observation, a reduced version of the one
analysed in Tutt et al. (2026), and run on CPU in a few minutes.

## Overview

- [21-cm only](eor.md): infer the power spectrum of a cosmological signal on its
  own.
- [21-cm and foregrounds](eor-fgs.md): add bright foregrounds, compress them into a
  [KL reduced basis](../model/kl-reduced-basis.md) and infer the power spectrum
  jointly.

Each example has its own data, configuration and entry point in the repository's
`examples/` folder, and runs from the repository root after
[installation](../installation.md):

```bash
python -m examples.eor.run --sample
python -m examples.eor_fgs.run --sample
```

Without `--sample`, each stops after setup, having loaded the data, added noise and
built the operators.
