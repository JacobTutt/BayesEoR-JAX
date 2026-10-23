# Interferometric visibilities

BayesEoR aims to detect the cosmological 21-cm contribution from the Epoch of
Reionisation and recover its power spectrum from interferometric observations.
These measurements also contain much brighter foreground emission, with both
contributions coupled through the instrument's frequency-dependent response.
We therefore begin with the measured visibilities, the instrument that samples
them, and the inputs required to describe an observation.

## From antenna voltages to visibilities

An interferometer measures correlations between the electric fields received
by pairs of antennas. For complex voltage vectors $\boldsymbol v_i$ and
$\boldsymbol v_j$, containing the two polarisation channels, their
cross-correlation defines the visibility coherency matrix

$$
\mathbf V_{ij}(\nu,t)
=\left\langle\boldsymbol v_i(\nu,t)
\boldsymbol v_j^\dagger(\nu,t)\right\rangle.
$$

The brackets denote averaging over the integration time and frequency channel,
and $\dagger$ denotes the conjugate transpose. BayesEoR's scalar sky model uses
total-intensity visibilities derived from these correlations; the supported
polarisation selection and pseudo-Stokes combination are described under
[input preparation](data.md#selecting-and-preparing-raw-visibilities).

Antennas at positions $\boldsymbol r_i$ and $\boldsymbol r_j$ define the physical
baseline and its wavelength-scaled coordinate,

$$
\boldsymbol b_{ij}=\boldsymbol r_i-\boldsymbol r_j,
\qquad
\boldsymbol u_{ij}(\nu)=\frac{\nu}{c}\boldsymbol b_{ij}
=(u_{ij},v_{ij},w_{ij}).
$$

The baseline's length and orientation determine which angular structure it
samples. Longer baselines resolve finer structure, while baselines with the
same separation and orientation are redundant under an identical antenna
response.

## The sky-to-visibility measurement

For a calibrated, unphased observation and a scalar primary-beam response,
the predicted visibility is

$$
m_{ij}(\nu,t)
=\int_{\Omega_{\rm vis}(t)}
P(\hat{\boldsymbol s},\nu)\,
I(\hat{\boldsymbol s},\nu,t)\,
\exp\!\left[-2\pi i\boldsymbol u_{ij}(\nu)
\cdot\hat{\boldsymbol s}\right]\,\mathrm d\Omega.
$$

Here $I=I_{21}+I_{\rm FG}$ is the sky brightness, $P$ is the primary-beam power
response, and $\hat{\boldsymbol s}=(l,m,n)$ is a direction in the instrument's
topocentric frame. The full three-dimensional phase retains the wide-field
geometry, including emission near the horizon. Earth rotation changes the
directions of celestial sources in this frame during an observation.

Stacking the measurements across baselines, frequencies and times gives

$$
\boldsymbol d=\boldsymbol m_{21}+\boldsymbol m_{\rm FG}+\boldsymbol n,
$$

where $\boldsymbol n$ denotes thermal noise. The [Model](model.md) section
defines the two sky contributions, and [Inference](joint-inference.md)
describes how their uncertainties enter power-spectrum recovery.

Before constructing these models, specify the [instrument and observation](instrument.md):
baseline geometry, observing frequencies and times, telescope location and
primary beam. Then follow [input preparation](data.md) to supply the measured
or simulated visibilities in the required units and ordering.
