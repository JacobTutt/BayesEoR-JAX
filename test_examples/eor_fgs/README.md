# Cosmological signal and foregrounds

From the repository root, in the installed CPU environment:

```bash
export JAX_PLATFORMS=cpu
python test_examples/run_analysis_example.py --config test_examples/eor_fgs/config.yaml --prepare-kl --sample
```

This builds and reduces the foreground operator with native JAX, loads the
saved data and adds seeded noise through setup, then performs joint inference,
analytically marginalising the cosmological
and latent foreground coefficients. Only six power-spectrum amplitudes are sampled.

Both components occupy 10° fields with the same global `Nside=128` inference grid.
The `footprint_nside=64` setting preserves the sky boundary of the saved data;
it does not set either model's resolution. The input
contains the same signal as the signal-only example, plus GSM2008 and point-source
emission. Results go to `outputs/laptop_eor_fgs_nside128/`, including the noisy
data, noise, matrices, KL diagnostics, posterior samples, evidence and plots.

For both examples and comparison plots, run `python -m test_examples.run_laptop_examples`.
Use its `--setup-only` option to stop before sampling.

See [the examples guide](../../docs/examples.md) for the observation, regeneration
and limitations. This restricted-field example is not a full-sky validation.
