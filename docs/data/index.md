# Interferometric data

BayesEoR performs cosmological inference directly on radio-interferometric
visibilities, the raw measurements of an interferometer, rather than on images or
power spectra estimated from them.

## Visibilities

An interferometer is an array of antennas whose signals are correlated in pairs.
The visibility measured by antennas $i$ and $j$ is the time-averaged
cross-correlation of their voltages,

$$
V_{ij}(\nu,t)=\left\langle v_i(\nu,t)\,v_j^{*}(\nu,t)\right\rangle.
$$

Each pair forms a baseline, $\boldsymbol b_{ij}=\boldsymbol r_i-\boldsymbol r_j$,
whose visibility measures a single Fourier mode of the beam-weighted sky. Measured
in wavelengths, the baseline sets which mode is sampled,

$$
\boldsymbol u_{ij}(\nu)=\frac{\nu}{c}\,\boldsymbol b_{ij}.
$$

Longer baselines respond to finer angular structure, and the baseline's
orientation sets the direction of that structure. Because $\boldsymbol u_{ij}$
scales with frequency, the same baseline samples a different angular scale in
every channel, and as the Earth rotates, the sampled modes change over the
course of an observation.

## Why the instrument matters

Each visibility only samples the modes set by its baseline, beam, frequency and
time, so the forward model is only as accurate as its description of the
instrument. BayesEoR therefore needs the array layout, primary beam, observing
frequencies, times and telescope location of the data being analysed:

- [Instrumental configuration](instrument.md): array layout and primary beam.
- [Observation configuration](observation.md): frequencies, times and telescope
  location.
- [Visibility data](visibility-data.md): supported formats, units, ordering and noise.

## Generating realistic data

To simulate realistic visibilities, use
[BayesEoR-Data-Generation](https://github.com/JacobTutt/BayesEoR-Data-Generation).
From a single YAML configuration, it builds mock-EoR, diffuse (GSM) and
point-source (GLEAM-like and A-team) sky models and simulates their visibilities
with [pyuvsim](https://github.com/RadioAstronomySoftwareGroup/pyuvsim). The
resulting UVH5 files can be passed directly to BayesEoR as `data_path`.
