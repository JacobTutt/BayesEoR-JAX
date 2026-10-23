# Cosmological signal and foregrounds

From the repository root, in the installed CPU environment:

```bash
export JAX_PLATFORMS=cpu
python test_examples/eor_fgs/prepare_kl.py
python test_examples/run_analysis_example.py --config test_examples/eor_fgs/config.yaml --sample
```

The first command builds and reduces the foreground operator with native JAX.
The second performs joint inference, analytically marginalising the cosmological
and latent foreground coefficients. Only six power-spectrum amplitudes are sampled.

Both components occupy 10° fields with `Nside=64` inference grids. The input
contains the same signal as the signal-only example, plus GSM2008 and point-source
emission. Results go to `outputs/laptop_eor_fgs/`.

See [the examples guide](../../docs/examples.md) for the observation, regeneration
and limitations. This restricted-field example is not a full-sky validation.
