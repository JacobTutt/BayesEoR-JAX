# 21-cm only

With no foregrounds in the data, this example isolates the EoR model: a
simulated 21-cm signal, observed by a compact HERA-like array, is recovered as a
power spectrum and compared directly with the spectrum that was injected.

=== "Data"

    The observation is a short drift scan by a compact hexagonal array, of the kind
    used by HERA. A simulated 21-cm signal is observed for a few minutes over a
    narrow band, and thermal noise is added during setup.

    `Array`
    :   37 antennas on a 14.6 m hexagonal grid, keeping the 15 independent baselines
        shorter than 40 m (30 with their reversed copies).

    `Primary beam`
    :   An Airy pattern for a 14.2 m dish.

    `Band`
    :   15 channels from 158.40 to 159.77 MHz, spaced by 97.66 kHz.

    `Integrations`
    :   15, spaced by 21.5 s, centred on JD 2459999.097.

    `Sky`
    :   A Gaussian random field with independent 6.4 mK pixels at HEALPix
        `Nside=256`, within a 10° field. This has a flat dimensional power
        spectrum, $P(k) \approx 8.84\times10^4\ \mathrm{mK^2\,Mpc^3}$, which is the
        injected spectrum to recover.

    `Noise`
    :   Complex Gaussian noise with RMS 0.006464 mK sr, added with a fixed seed.

    The data are 6,750 complex visibilities, or 3,375 independent measurements once
    the conjugate copies are accounted for. They are supplied as a noiseless `.npy`
    vector with its [instrument model](../data/instrument.md#array-layout), so no
    sky simulation is needed to run the example.

    ```text
    examples/eor/
    ├── config.yaml
    ├── run.py
    └── data/
        ├── visibilities.npy             # noiseless data vector, in mK sr
        ├── uvw_model.npy                # baseline vectors
        ├── redundancy_model.npy         # baseline redundancies
        ├── injected_power_spectrum.npy  # the injected P(k)
        └── simulation.json              # full simulation record
    ```

=== "Running"

    `python -m examples.eor.run --sample` runs every step below; without
    `--sample`, it stops after setup.

    **1. Configure the run.** `config.yaml` describes the data, the observation and
    the model:

    ```yaml
    # Model and data
    model: eor                          # fit the 21-cm signal only
    # Noiseless data vector
    data_path: examples/eor/data/visibilities.npy
    inst_model: examples/eor/data       # baselines and redundancies
    # Saved operators
    array_save_directory: outputs/laptop_eor_nside128/matrices
    # Sampler outputs and metadata
    output_dir: outputs/laptop_eor_nside128

    # Observation
    n_freq: 15                          # number of channels
    f0_mhz: 158.3984375                 # first channel, MHz
    df_mhz: 0.09765625                  # channel spacing, MHz
    n_t: 15                             # number of integrations
    d_t: 21.511369943618774             # integration spacing, s
    jd_center: 2459999.097166265        # central integration, JD
    # HERA site
    telescope_latlonalt: [-30.72152777777791, 21.428305555555557, 1073.0000000093132]

    # Instrument
    beam_type: airy                     # Airy primary beam
    diam: 14.2                          # dish diameter, m

    # EoR model
    fov_eor: 10.0                       # model field diameter, degrees
    nside: 128                          # HEALPix resolution of the model
    footprint_nside: 64                 # grid on which the field boundary is chosen

    # Noise
    sigma: 0.006464                     # complex noise RMS, mK sr
    noise_seed: 742123                  # seed for the added noise

    # Inference
    # Prior range on log10(Δ²/mK²), one pair per bin
    priors: [[-1, 6], [-1, 6], [-1, 6], [-1, 6], [-1, 6], [-1, 6]]
    n_live_scale_factor: 10             # live points = 10 × parameters
    random_key: 42                      # sampler seed
    ```

    See the [Data](../data/index.md), [Model](../model/index.md) and
    [Inference](../inference/index.md) pages for each setting.

    **2. Build the model.** [`run_setup`](../model/construction.md) loads the data,
    adds the noise, and builds the [EoR operator](../model/eor.md) and likelihood
    products:

    ```python
    from bayeseor_jax import BayesEoRParser, run_setup

    config = BayesEoRParser().parse_args(
        ["--config", "examples/eor/config.yaml", "--run"]
    )
    setup = run_setup(config)
    ```

    The Fourier grid is sized automatically to $11 \times 11$ transverse and 14
    line-of-sight modes, giving 1,680 EoR coefficients in six
    [power-spectrum bins](../inference/power-spectrum.md). The coefficients are
    [marginalised analytically](../inference/marginalisation.md), so only the six
    bin amplitudes are sampled.

    **3. Sample the power spectrum.** [`run_nestedsampling`](../inference/running.md)
    explores the posterior on the six amplitudes with MultiNest, and evaluates the
    evidence:

    ```python
    from bayeseor_jax import run_nestedsampling

    summary = run_nestedsampling(
        setup,
        out_dir="outputs/laptop_eor_nside128/nested_sampling",
        random_key=config.random_key,
        n_live_scale_factor=config.n_live_scale_factor,
        center_likelihood=True,   # numerical offset, restored in the evidence
    )
    ```

    The noisy data, saved operators, chains and `summary.json` are written under
    `outputs/laptop_eor_nside128/`.

    **4. Summarise the posterior.** `save_analysis` reads the chains, summarises
    each bin and writes the figure in Results to `examples/eor/plots/`:

    ```python
    from bayeseor_jax.analyze import save_analysis

    save_analysis(
        "outputs/laptop_eor_nside128/nested_sampling",
        output_dir="examples/eor/plots",
        metadata_dir="outputs/laptop_eor_nside128",
        label="Cosmological signal only",
    )
    ```

=== "Results"

    [![Recovered 21-cm power spectrum and marginal posteriors for the signal-only example.](../assets/examples/eor.png#only-light)](../assets/examples/eor.png)
    [![Recovered 21-cm power spectrum and marginal posteriors for the signal-only example.](../assets/examples/eor-dark.png#only-dark)](../assets/examples/eor-dark.png)

    All six power-spectrum amplitudes are recovered, with the injected spectrum
    (dashed) inside the 95% credible interval of every bin. The posteriors are
    compact and well separated from the prior edges, showing that the data
    constrain each scale.

    | $k$ [Mpc⁻¹] | Injected $\Delta^2$ [mK²] | Median [mK²] | 95% interval [mK²] |
    | --- | --- | --- | --- |
    | 0.26 | 77 | 50 | 26 – 86 |
    | 0.51 | 612 | 503 | 274 – 868 |
    | 0.77 | 2,063 | 2,657 | 1,745 – 4,661 |
    | 1.03 | 4,888 | 5,606 | 3,320 – 8,619 |
    | 1.42 | 12,706 | 15,193 | 10,796 – 21,552 |
    | 1.80 | 26,192 | 26,320 | 14,244 – 46,524 |
