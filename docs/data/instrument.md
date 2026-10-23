# Instrumental configuration

The instrument enters the forward model through its array layout, which sets the
sampled baselines, and its primary beam, which weights the sky seen by each
antenna. Both must describe the instrument that produced the data.

## Array layout

BayesEoR describes the array by its sampled baselines, rather than a table of
antenna positions. Each baseline is a three-dimensional vector in metres, with
east, north and up components in the telescope's local frame. The full vector,
including the up ($w$) component, enters the forward model, so the wide-field
geometry is preserved.

Every baseline is paired with its reversed copy, $\boldsymbol b \to
-\boldsymbol b$, which carries the conjugate visibility. The baseline axis
therefore has length $2N_{\rm bls}$: the original baselines, followed by their
reversed copies in the same order.

How the baselines are supplied depends on the data format.

=== "Visibility files (.uvh5, .uvfits, .ms)"

    The baselines are read from the file with pyuvdata, sorted by antenna pair and
    paired with their reversed copies automatically. The file should contain
    calibrated, drift-scan data, with the same baselines present at every selected
    time. Phased data are unprojected before use.

    `antenna_selection`
    :   Which antenna pairs to keep.

        - **Default:** `cross`, all cross-correlations, no autocorrelations.
        - **Accepts:** any string accepted by pyuvdata's
          [`UVData.select(ant_str=...)`](https://pyuvdata.readthedocs.io/en/latest/uvdata.html),
          for example `all` or explicit pairs such as `0_1,0_2,1_2`.

    `bl_cutoff`
    :   Removes baselines longer than the cutoff.

        - **Default:** none, all baselines kept.
        - **Units:** metres.
        - **Length:** the projected length $\sqrt{u^2+v^2}$; the up component is
          not included.

    `redundant_avg`
    :   Averages groups of redundant baselines, those with the same separation and
        orientation, into one visibility each.

        - **Default:** `false`.
        - **Redundancy:** the number of baselines in each group, $R$, is stored
          and reduces its noise variance in the likelihood to $\sigma^2/R$.

    `uniform_redundancy`
    :   Keeps a redundancy of one for every baseline, even after averaging.

        - **Default:** `false`.

    `save_model`
    :   Writes the selected baselines to `save_dir` as `uvw_model.npy`,
        `redundancy_model.npy` and `antpairs.npy`.

        - **Default:** `false`.
        - **Reuse:** the files use the preprocessed format, so together with the
          vector written by `save_vis: true` they can be passed back in as a
          `.npy` input.

=== "Preprocessed vectors (.npy)"

    `.npy` visibility vectors carry no metadata, so point `inst_model` to a
    directory describing their baselines:

    ```yaml
    inst_model: inputs/instrument/
    ```

    `uvw_model.npy` (required)
    :   The baseline vectors.

        - **Shape:** $(N_t, 2N_{\rm bls}, 3)$, one set per integration.
        - **Units:** metres, not wavelengths, despite the name.
        - **Components:** east, north, up.
        - **Order:** entries $b$ and $N_{\rm bls}+b$ are the same baseline with
          opposite sign, and the order matches the data vector; see
          [visibility ordering](visibility-data.md#data-vector-ordering).

    `redundancy_model.npy` (required)
    :   The number of baselines averaged into each entry.

        - **Shape:** $(N_t, 2N_{\rm bls})$ or $(N_t, 2N_{\rm bls}, 1)$.
        - **Values:** ones for unaveraged data; a baseline and its reversed copy
          take the same value.

    `antpairs.npy` (optional)
    :   Antenna-pair labels, used for bookkeeping only.

        - **Shape:** $2N_{\rm bls}$ tuples, in baseline order.
        - **Values:** for example `(0, 1)`, with its reversed copy labelled
          `(1, 0)`.

    Each file can be a plain array, or a BayesEoR dictionary with the array under
    `data`, as written by `save_model`. A single `instrument_model.npy`
    dictionary with `uvw_model`, `redundancy_model` and optional `antpairs`
    entries can be used instead, and takes precedence if present. Dictionary files
    are loaded with pickle, so only load them from trusted sources.

    ??? example "Building the files from an array of baselines"

        ```python
        import numpy as np

        # uvw: original baselines, shape (n_t, n_bls, 3), east/north/up in metres
        uvw_model = np.concatenate([uvw, -uvw], axis=1)
        redundancy_model = np.ones(uvw_model.shape[:2])

        np.save("inputs/instrument/uvw_model.npy", uvw_model)
        np.save("inputs/instrument/redundancy_model.npy", redundancy_model)
        ```

## Primary beam

The primary beam is the power response of each antenna to the sky,
$P(\hat{\boldsymbol s},\nu)$. It is evaluated along every sky pixel's track at
each frequency and integration, and weights the sky before it is mapped into the
visibilities. A single beam is used for all antennas, and its response is zero at
and below the horizon.

Set `beam_type` to the name of an analytic profile, or to the path of a beam file.
Below, $\theta$ is the zenith angle.

=== "Analytic profiles"

    `uniform`
    :   A unit response everywhere above the horizon.

        - **Parameters:** none.
        - **Default:** this is the default `beam_type`.

    `gaussian`
    :   A Gaussian in zenith angle, $P = A\exp(-\theta^2/2\sigma^2)$.

        - **Width:** `fwhm_deg`, in degrees, for a frequency-independent width;
          or `diam`, in metres, for a chromatic width matched to the main lobe of
          an Airy beam.
        - **Peak:** `peak_amp`, the zenith response $A$. Defaults to 1.

    `airy`
    :   The power pattern of a uniformly illuminated circular dish,
        $P = [2J_1(x)/x]^2$ with $x = \pi D\nu\sin\theta/c$.

        - **Aperture:** `diam`, the dish diameter $D$ in metres; or `fwhm_deg`,
          from which an effective diameter is set at each frequency so that the
          main-lobe width is fixed.
        - **Peak:** always 1.

    `gausscosine`
    :   A Gaussian modulated by a cosine, $P = A\exp(-\theta^2/2\sigma^2)
        \cos^2(\pi f_c\theta)$, adding radial structure to the beam.

        - **Width:** `fwhm_deg` or `diam`, as for `gaussian`.
        - **Modulation:** `cosfreq`, $f_c$, in inverse radians.
        - **Peak:** `peak_amp`. Defaults to 1.

    `taperairy`
    :   An Airy pattern multiplied by a Gaussian taper, suppressing its sidelobes.

        - **Aperture:** `diam`, in metres.
        - **Taper:** `fwhm_deg`, the width of the Gaussian, in degrees.
        - **Peak:** `peak_amp`. Defaults to 1.

    `tanhairy`
    :   An Airy pattern multiplied by a smooth taper,
        $1 - s\tanh(k\theta)$, that suppresses the response far from zenith.

        - **Aperture:** `diam`, in metres.
        - **Rate:** `tanh_freq`, $k$, in inverse radians.
        - **Suppression:** `tanh_sl_red`, $s$. Far from zenith the power is scaled
          by $1-s$, so `0.99` leaves 1%.

    For example:

    ```yaml
    beam_type: gaussian
    fwhm_deg: 9.306821090681533
    ```

=== "Beam files"

    Any beamfits file readable by pyuvdata's `UVBeam` can be used, on either a
    HEALPix or an azimuth/zenith-angle grid, and as an electric-field or power
    beam. It is interpolated onto the sky pixels and observing frequencies, so its
    frequency range should cover the observation.

    ```yaml
    beam_type: inputs/beams/beam.fits
    pol: pI
    ```

    `pol`
    :   The beam polarisation to use.

        - **Default:** `xx`.
        - **Accepts:** `xx`, `yy` or `pI`. Electric-field beams are converted to
          the corresponding power or pseudo-Stokes I response; power beams must
          already contain it.
        - **Match the data:** with the default `form_pi: true`, the data are
          pseudo-Stokes I, so use `pol: pI`.

    `freq_interp_kind`
    :   How the beam is interpolated between its tabulated frequencies.

        - **Default:** `cubic`.
        - **Accepts:** any SciPy `interp1d` kind, such as `linear` or `cubic`.

!!! tip

    Beam sidelobes matter. If the beam has a non-negligible response outside the
    EoR field of view, set the foreground field of view, `fov_fg`, larger than
    `fov_eor`. See the [foreground model](../model/foregrounds.md).
