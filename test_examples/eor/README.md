# Cosmological signal only

From the repository root, in the installed CPU environment:

```bash
export JAX_PLATFORMS=cpu
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml --sample
```

Omit `--sample` for setup only. This uses all 30 signed short-baseline groups,
15 channels and 15 integrations, with a 10° cosmological field and an
`Nside=64` inference grid. The noiseless input was simulated at `Nside=256`;
setup adds seeded thermal noise.

Results go to `outputs/laptop_eor/`. See [the examples guide](../../docs/examples.md)
for the observation, input provenance, regeneration and limitations.
