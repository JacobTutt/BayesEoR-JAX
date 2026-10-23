# 21-cm and foregrounds

This example adds bright Galactic and extragalactic foregrounds to the same
simulated 21-cm signal. The foregrounds are compressed into a
[KL reduced basis](../model/kl-reduced-basis.md) during setup, and the power
spectrum is inferred jointly with them, recovering the injected spectrum despite
emission orders of magnitude brighter.

=== "Data"

    The observation is identical to the [21-cm only](eor.md) example, with the same
    array, beam, band, integrations, noise and 21-cm realisation. Two foreground
    components are added within the same 10° field:

    `Diffuse emission`
    :   The Global Sky Model (GSM2008), at HEALPix `Nside=128`.

    `Point sources`
    :   A statistical population of extragalactic sources, together with the
        catalogued bright A-team sources, each at its individual position.

    All three components were propagated through the beam and full
    three-dimensional fringe response, and summed into a single noiseless data
    vector. `simulation.json` records the seeds and provenance of each component.

    ```text
    examples/eor_fgs/
    ├── config.yaml
    ├── run.py
    └── data/
        ├── visibilities.npy             # noiseless 21-cm + foreground data, in mK sr
        ├── uvw_model.npy                # baseline vectors
        ├── redundancy_model.npy         # baseline redundancies
        ├── injected_power_spectrum.npy  # the injected P(k)
        └── simulation.json              # full simulation record
    ```

=== "Running"

    `python -m examples.eor_fgs.run --sample` runs every step below; without
    `--sample`, it stops after setup, including the KL construction.

    **1. Configure the run.** `config.yaml` extends the 21-cm only configuration with
    the foreground model and the path of its reduced basis:

    ```yaml
    # Model and data
    model: eor+fg                       # fit the 21-cm signal and foregrounds jointly
    # Noiseless data vector
    data_path: examples/eor_fgs/data/visibilities.npy
    inst_model: examples/eor_fgs/data   # baselines and redundancies
    # Saved operators
    array_save_directory: outputs/laptop_eor_fgs_nside128/matrices
    # Sampler outputs and metadata
    output_dir: outputs/laptop_eor_fgs_nside128
    # KL basis, built in setup
    foreground_kl_path: outputs/laptop_eor_fgs_nside128/kl/T_fg_reduced.npy

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

    # Sky models
    fov_eor: 10.0                       # EoR field diameter, degrees
    fov_fg: 10.0                        # foreground field diameter, degrees
    beta_fg: [2.63, 2.82]               # foreground power-law indices
    nside: 128                          # HEALPix resolution of both models
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

    The foreground prior widths use their defaults; see
    [`foreground_prior_std_mk`](../inference/marginalisation.md#foreground-prior).

    **2. Build the model and reduced basis.** Setup builds the prior- and
    noise-weighted foreground operator, compresses it and assembles the joint model,
    in a single call:

    ```python
    from bayeseor_jax import BayesEoRParser, run_setup

    config = BayesEoRParser().parse_args(
        ["--config", "examples/eor_fgs/config.yaml", "--run"]
    )
    setup = run_setup(config, build_foreground_kl=True)
    ```

    The physical foreground model has 1,248 coefficients: three spectral amplitudes
    in each of 416 pixels. Truncating at $f = 0.1$ retains **179 KL modes**, with the
    discarded foreground RMS at 9.5% of the noise RMS, in about 2 s on a laptop CPU.
    With the 1,680 EoR coefficients, the likelihood marginalises 1,859 coefficients
    and samples the same six power-spectrum amplitudes. Without
    `build_foreground_kl=True`, setup loads an existing basis from
    `foreground_kl_path`.

    **3. Sample the power spectrum.** Sampling is the same as in the 21-cm only
    example:

    ```python
    from bayeseor_jax import run_nestedsampling

    summary = run_nestedsampling(
        setup,
        out_dir="outputs/laptop_eor_fgs_nside128/nested_sampling",
        random_key=config.random_key,
        n_live_scale_factor=config.n_live_scale_factor,
        center_likelihood=True,
    )
    ```

    Outputs are written under `outputs/laptop_eor_fgs_nside128/`, with the KL basis,
    its singular values and truncation diagnostics under `kl/`.

    **4. Summarise the posterior.** `save_analysis` writes the figure in Results to
    `examples/eor_fgs/plots/`:

    ```python
    from bayeseor_jax.analyze import save_analysis

    save_analysis(
        "outputs/laptop_eor_fgs_nside128/nested_sampling",
        output_dir="examples/eor_fgs/plots",
        metadata_dir="outputs/laptop_eor_fgs_nside128",
        label="Cosmological signal + foregrounds",
        uplim_inds=[True, False, False, False, False, False],
    )
    ```

=== "Results"

    [![Recovered 21-cm power spectrum and marginal posteriors with foregrounds.](../assets/examples/eor-fgs.png#only-light)](../assets/examples/eor-fgs.png)
    [![Recovered 21-cm power spectrum and marginal posteriors with foregrounds.](../assets/examples/eor-fgs-dark.png#only-dark)](../assets/examples/eor-fgs-dark.png)

    With foregrounds included, the five highest-$k$ bins are recovered as precisely
    as in the 21-cm only example. The lowest bin, $k = 0.26\ \mathrm{Mpc^{-1}}$, is
    no longer bounded from below: its posterior extends to the edge of the prior,
    since these large, spectrally smooth scales are where the signal is most
    degenerate with the foregrounds. The injected value still lies within its 95%
    interval, so the foreground uncertainty is propagated into the power spectrum
    rather than biasing it.

    The lowest-$k$ bin is shown as a 95% upper limit in the power-spectrum panel;
    the other bins show central 68% credible intervals. The right-hand panels
    display the full marginal posteriors.

    | $k$ [Mpc⁻¹] | Injected $\Delta^2$ [mK²] | Median [mK²] | 95% interval [mK²] |
    | --- | --- | --- | --- |
    | 0.26 | 77 | 5 | 0.1 – 79 |
    | 0.51 | 612 | 588 | 338 – 1,023 |
    | 0.77 | 2,063 | 2,884 | 1,863 – 4,648 |
    | 1.03 | 4,888 | 5,610 | 3,252 – 9,292 |
    | 1.42 | 12,706 | 15,500 | 10,719 – 22,208 |
    | 1.80 | 26,192 | 26,802 | 15,807 – 43,248 |
