# Observation configuration

The observation sets when and where the sky was observed: the frequency channels,
the integration times and the telescope location. Together these fix where each
sky direction lies relative to the instrument and the horizon at every
integration.

For visibility files, the frequencies and times are read from the file and can be
down-selected. Preprocessed `.npy` vectors carry no metadata, so their axes are
defined in the configuration instead. The telescope location is always set in the
configuration.

## Frequencies

=== "Visibility files (.uvh5, .uvfits, .ms)"

    All channels in the file are used unless a starting option is set. Choose one
    starting option and, optionally, a width:

    `freq_idx_min`
    :   Starts at a channel index.

        - **Units:** zero-based channel index.

    `freq_min`
    :   Starts at the first channel at or above this frequency.

        - **Units:** Hz.

    `freq_center`
    :   Selects a window centred on the channel nearest this frequency.

        - **Units:** Hz.
        - **Requires:** `n_freqs_select`.

    `n_freqs_select`
    :   The number of channels to keep.

        - **Default:** all channels from the starting channel onwards.
        - **Note:** has no effect without a starting option.

    If more than one starting option is set, `freq_center` takes precedence, then
    `freq_min`. The parser still requires `n_freq`, `f0_mhz` and `df_mhz`, but
    their values are replaced by the selected channels.

=== "Preprocessed vectors (.npy)"

    The channels are defined directly, as $\nu_f = \nu_0 + f\,\Delta\nu$ for
    $f = 0, \dots, N_\nu - 1$:

    ```yaml
    n_freq: 38
    f0_mhz: 158.304048743
    df_mhz: 0.23761898685801028
    ```

    `n_freq`
    :   The number of channels, $N_\nu$.

    `f0_mhz`
    :   The frequency of the first channel, $\nu_0$.

        - **Units:** MHz.

    `df_mhz`
    :   The channel spacing, $\Delta\nu$.

        - **Units:** MHz.

## Times

=== "Visibility files (.uvh5, .uvfits, .ms)"

    All integrations in the file are used unless a starting option is set. Choose
    one starting option and, optionally, a width:

    `jd_idx_min`
    :   Starts at an integration index.

        - **Units:** zero-based index.

    `jd_min`
    :   Starts at the first integration at or after this time.

        - **Units:** Julian date.

    `jd_center`
    :   Selects a window centred on the integration nearest this time.

        - **Units:** Julian date.
        - **Requires:** `n_times_select`.

    `n_times_select`
    :   The number of integrations to keep.

        - **Default:** all integrations from the starting one onwards.
        - **Note:** has no effect without a starting option.

    If more than one starting option is set, `jd_center` takes precedence, then
    `jd_min`. The integration time is read from the file.

=== "Preprocessed vectors (.npy)"

    The integrations are defined directly, as $N_t$ evenly spaced times:

    ```yaml
    n_t: 1
    d_t: 10.0
    jd_center: 2461040.5
    ```

    `n_t`
    :   The number of integrations, $N_t$.

        - **Default:** 1.
        - **Note:** must match the time axis of the
          [instrument model](instrument.md#array-layout).

    `d_t`
    :   The spacing between integrations.

        - **Units:** seconds.
        - **Note:** required, even for a single integration.

    `jd_min` or `jd_center`
    :   Anchors the times. One is required.

        - **Units:** Julian date.
        - **`jd_min`:** the first integration.
        - **`jd_center`:** the integration at index $\lfloor N_t/2 \rfloor$.
        - **Precedence:** `jd_min`, if both are set.

## Telescope location

```yaml
telescope_latlonalt: [-30.72152777777791, 21.428305555555557, 1073.0000000093132]
```

`telescope_latlonalt`
:   The telescope's geodetic position, as `[latitude, longitude, altitude]`.

    - **Units:** degrees, degrees, metres.
    - **Required:** for every input format; it is not read from visibility files.
    - **Example:** the values above are for HERA.
