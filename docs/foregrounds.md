# Foregrounds

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX represents smooth foreground brightness directly on a spherical
image-domain grid. Each spatial mode has one constant coefficient and one
coefficient for every configured power-law spectral index:

```yaml
fov_fg: 180.0
beta_fg: [2.63, 2.82]
foreground_prior_std_mk: [8690.96, 2479547.30, 1406322.41]
```

The foreground operator is

```text
T_fg = F^-1 P Q_z,
```

where `Q_z` contains the constant and power-law spectral fields, `P` is the
primary-beam response, and `F^-1` is the full three-dimensional spherical
fringe operator. The coefficient columns are ordered by spectral field and
then by image-domain spatial mode.

`fov_fg` may be substantially larger than the cosmological field because
foreground emission enters through beam sidelobes and the chromatic horizon
response. The foreground field uses a single HEALPix resolution. The saved
physical operator can subsequently be weighted by its prior and noise factors
for KL construction. `T_fg_svd_build_save` decomposes that saved weighted matrix
on GPUs and writes its noise-thresholded reduced operator as a separate stage;
see [the construction workflow](forward-model.md#distributed-foreground-svd-and-truncation).

Large operators are divided into contiguous ranges of time-frequency blocks
using the active JAX process topology, constructed on CPU, and written into a
shared NumPy memmap. This avoids materialising the full matrix on a GPU.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
