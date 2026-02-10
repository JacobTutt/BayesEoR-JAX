"""
"""

import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Tuple, Union
from jax.experimental import sparse
import jax.numpy as jnp

from astropy import constants as c

from .dft_funcs import inverse_udft_1d_rect, inverse_udft_1d_sub, build_nudft_array
from .uv_model_space import UV_ModelSpace
from .lmn_model_space import LMN_ModelSpace
from .beam_model import Beam_Model

class BuildMatrices:
    """
    Class for handling matrix construction and arithmetic.

    Parameters
    ----------
    EoR model uv-plane parameters:
    n_u_eor : int
        Number of pixels on a side for the u axis in the model uv-plane
        of the EoR model. Defaults to None.
    du_eor : float
        Fourier mode spacing along the u axis in inverse radians of the
        EoR model uv-plane. Defaults to None.
    n_v_eor : int
        Number of pixels on a side for the v axis in the model uv-plane
        of the EoR model. Defaults to None.
    dv_eor : float
        Fourier mode spacing along the v axis in inverse radians of the
        EoR model uv-plane. Defaults to None.

    Foreground model uv-plane parameters:
    n_u_fg : int
        Number of pixels on a side for the u-axis in the FG model uv-plane.
        Defaults to None.
    du_fg : float
        Fourier mode spacing along the u axis in inverse radians of the
        FG model uv-plane. Defaults to None.
    n_v_fg : int
        Number of pixels on a side for the v-axis in the FG model uv-plane.
        Defaults to None.
    dv_fg : float
        Fourier mode spacing along the v axis in inverse radians of the
        FG model uv-plane. Defaults to None.
    n_modes_fg : int, optional
        Number of quadratic modes in the Large Spectral Scale Model (LSSM).
        Defaults to 0.
    n_powerlaws_fg : int, optional
        Number of power law coefficients which replace quadratic modes in
        the LSSM. Defaults to 0.
    beta_fg : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Can be a single spectral index, e.g.
        [2.63], or multiple spectral indices, e.g. [2.63, 2.82]. Defaults to
        [2.63, 2.82].

    
    SubHarmonic Grid (SHG) parameters:
    use_shg : bool, optional
        Use the SubHarmonic Grid (SHG) in the model uv-plane. Defaults to
        False.
    n_u_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic model
        uv-plane. Defaults to None.
    du_sh : float, optional
        Fourier mode spacing along the u axis in inverse radians of the
        subharmonic model uv-plane. Defaults to None.
    n_v_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic model
        uv-plane. Defaults to None.
    dv_sh : float, optional
        Fourier mode spacing along the v axis in inverse radians of the
        subharmonic model uv-plane. Defaults to None.
    n_modes_sh : int, optional
        Number of large spectral scale modes for each pixel in the subharmonic
        grid. Defaults to None.
    n_powerlaws_sh : int, optional
        Number of power law coefficients used in the large spectral scale model
        for each pixel in the subharmonic grid. Defaults to None.
    beta_sh : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Defaults to [2.63, 2.82].
    fit_for_sh_amps : bool, optional
        If True, fit per-frequency SHG amplitudes (full eta spectrum, includes eta=0).
        If False, fit only smooth spectral terms (eta collapsed to zero mode only).
        Defaults to False.
    
    # Frequency and line of sight Fourier mode parameters:
    n_freq : int
        Number of frequency channels. Defaults to None.
    f0_mhz : float
        Minimum frequency in megahertz. Defaults to None.
    df_mhz : float
        Frequency channel width in megahertz. Defaults to None.
    n_eta : int
        Number of Line of Sight (LoS) Fourier modes. Defaults to None.
    d_eta : float
        Fourier mode spacing along the eta (line of sight, frequency) axis in
        inverse Hz. Defaults to None.

    # Model parameters:
    fit_for_monopole : bool, optional
        Fit for (u, v) = (0, 0) (True) or exclude it from the fit (False).
        Defaults to False.

    Healpix image-domain model parameters:
    telescope_latlonalt : tuple, optional
        The latitude, longitude, and altitude of the telescope in degrees,
        degrees, and meters, respectively. Defaults to HERA location.
    jd_start : float
        Start time step of the observation in JD units.
        Defaults to 2461040.5 (2026-01-01 00:00:00 UTC).
    nside : int
        HEALPix nside parameter. Defaults to None.
    fov_eor : float
        Circular field of view in degrees for the EoR model.
    fov_fg : float
        Circular field of view in degrees for the foreground model. 
    n_t : int
        Number of time integrations. Defaults to 1.
    d_t : float, optional
        Integration time in seconds. Required if ``n_t > 1``.
    
    # Beam model parameters:
    beam_type : str, optional
        Can be either a path to a pyuvdata-compatible beam file or one of
        'uniform', 'gaussian', 'airy', 'gausscosine', 'taperairy', or 'tanhairy'.
        Defaults to 'uniform'.
    peak_amp : float, optional
        Peak amplitude of the beam. Defaults to 1.0.
    fwhm_deg : float, optional
        Full width at half maximum of the beam in degrees. Required if
        `beam_type` is 'gaussian' or 'gausscosine'.
    diam : float, optional
        Antenna (aperture) diameter in meters. Used if `beam_type` is 'airy'.
    cosfreq : float, optional
        Cosine frequency in inverse radians. Used if `beam_type` is 'gausscosine'.
    tanh_freq : float, optional
        Exponential frequency (rate parameter) in inverse radians. Used if
        `beam_type` is 'tanhairy'.
    tanh_sl_red : float, optional
        Airy sidelobe amplitude reduction as a fractional percent. Used if
        `beam_type` is 'tanhairy'.
    pol : str, optional
        Polarization string. Used only if `beam_type` is a UVBeam file path.
        Defaults to 'xx'.
    freq_interp_kind : str, optional
        Frequency interpolation kind from scipy.interpolate.interp1d. 
        Defaults to 'cubic'.

    # Data parameters:
    uvw_array_m : numpy.ndarray
        Array containing the (u(t), v(t), w(t)) coordinates of the instrument
        model with shape (nt, nbls, 3). Defaults to None.
    bl_red_array : numpy.ndarray
        Array containing the number of redundant baselines at each
        (u(t), v(t), w(t)) in the instrument model with shape (nt, nbls, 1).
        Defaults to None.
    phasor : numpy.ndarray, optional
        Array with shape (ndata,) that contains the phasor term used to phase
        visibilities after performing the nuDFT from HEALPix (l, m, f) to
        instrumentally sampled, unphased (u, v, f).  Defaults to None, i.e.
        modelling unphased visibilities. Defaults to None.

    """


    def __init__(
        self,
        n_u_eor: int,
        du_eor: float,
        n_v_eor: int,
        dv_eor: float,
        n_u_fg: int,
        du_fg: float,
        n_v_fg: int,
        dv_fg: float,
        n_modes_fg: int = 0,
        n_powerlaws_fg: int = 0,
        beta_fg: Optional[Union[float, Sequence[float], jnp.ndarray]] = None,
        use_shg: bool = False,
        n_u_sh: Optional[int] = None,
        du_sh: Optional[float] = None,
        n_v_sh: Optional[int] = None,
        dv_sh: Optional[float] = None,
        n_modes_sh: Optional[int] = None,
        n_powerlaws_sh: Optional[int] = None,
        beta_sh: Optional[Union[float, Sequence[float], jnp.ndarray]] = None,
        fit_for_sh_amps: bool = False,
        n_freq: int = 0,
        f0_mhz: Optional[float] = None,
        df_mhz: Optional[float] = None,
        n_eta: int = 0,
        d_eta: float = 0.0,
        fit_for_monopole: bool = False,
        telescope_latlonalt: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        jd_start: float = 2461040.5,
        nside: Optional[int] = None,
        fov_eor: Optional[float] = None,
        fov_fg: Optional[float] = None,
        n_t: int = 1,
        d_t: Optional[float] = None,
        beam_type: str = 'uniform',
        peak_amp: float = 1.0,
        fwhm_deg: Optional[float] = None,
        diam: Optional[float] = None,
        cosfreq: Optional[float] = None,
        tanh_freq: Optional[float] = None,
        tanh_sl_red: Optional[float] = None,
        pol: str = 'xx',
        freq_interp_kind: str = 'cubic',
        uvw_array_m: Optional[jnp.ndarray] = None,
        bl_red_array: Optional[jnp.ndarray] = None,
        phasor: Optional[jnp.ndarray] = None,
    ) -> None:
        
        # Store the input parameters as attributes of the class instance.
        # The Models UV-plane parameters for the EoR model.
        self.n_u_eor = n_u_eor
        self.du_eor = du_eor
        self.n_v_eor = n_v_eor
        self.dv_eor = dv_eor
        # Overall No. of UV space samples in the EoR model (accounting for monopole exclusion).
        # The monopole of the EoR model is always excluded
        self.n_uv_eor = (n_u_eor * n_v_eor) - 1

        # Store the foreground uv-plane parameters.
        self.n_u_fg = n_u_fg
        self.du_fg = du_fg
        self.n_v_fg = n_v_fg
        self.dv_fg = dv_fg
        # LSSM parameters for the foreground model
        self.n_modes_fg = n_modes_fg
        self.n_powerlaws_fg = n_powerlaws_fg
        self.beta_fg = beta_fg if beta_fg is not None else [2.63, 2.82]
        self.fit_for_monopole = fit_for_monopole
        # Overall No. of UV space samples in the foreground model (accounting for optional monopole exclusion).
        self.n_uv_fg = (n_u_fg * n_v_fg) - int(not self.fit_for_monopole)

        
        # Store the SubHarmonic Grid (SHG) parameters.
        self.use_shg = use_shg
        self.n_u_sh = n_u_sh
        self.du_sh = du_sh
        self.n_v_sh = n_v_sh
        self.dv_sh = dv_sh
        # LSSM parameters for the subharmonic grid
        self.n_modes_sh = n_modes_sh
        self.n_powerlaws_sh = n_powerlaws_sh
        self.fit_for_sh_amps = fit_for_sh_amps
        self.beta_sh = beta_sh if beta_sh is not None else [2.63, 2.82]
        # Overall No. of UV space samples in the subharmonic grid (accounting for monopole exclusion).
        # The monopole of the SHG is always excluded
        self.n_uv_sh = (n_u_sh * n_v_sh) - 1

        # Store the frequency and line of sight Fourier mode parameters.
        self.n_freq = n_freq
        self.f0_mhz = f0_mhz
        self.df_mhz = df_mhz    
        self.n_eta = n_eta
        self.d_eta = d_eta
        # Construct the frequency array in Hz (from MHz)
        self.freq_hz = (f0_mhz + df_mhz * jnp.arange(n_freq, dtype=jnp.float64)) * 1e6 

        # Store the Healpix image-domain model parameters.
        self.nside = nside
        self.fov_eor = fov_eor
        self.fov_fg = fov_fg
        self.telescope_latlonalt = telescope_latlonalt
        self.jd_start = jd_start
        self.n_t = n_t
        self.d_t = d_t

        # Store the beam model parameters.
        self.beam_type = beam_type
        self.peak_amp = peak_amp
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red
        self.pol = pol
        self.freq_interp_kind = freq_interp_kind

        # Store the data products and parameters.
        self.uvw_array_m = jnp.array(uvw_array_m) # Shape (nt, nbls, 3)
        self.inst_uvw_irad = self.uvw_array_m[jnp.newaxis, :, :, :]/ (float(c.to("m/s").value) / self.freq_hz[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]) # Shape (nf, nt, nbls, 3)
        self.inst_uvw_irad = jnp.swapaxes(self.inst_uvw_irad, -2, -1) # Shape (nf, nt, 3, nbls)
        self.bl_red_array = bl_red_array
        self.phasor = phasor


        # Initialised the UV_model space, LMN_ModelSpace and BeamModel classes as attributes of the BuildMatrices class.
        self.uv_model_space = UV_ModelSpace(
            n_u_eor=self.n_u_eor,
            n_v_eor=self.n_v_eor,
            n_u_fg=self.n_u_fg,
            n_v_fg=self.n_v_fg,
            n_u_sh=self.n_u_sh,
            n_v_sh=self.n_v_sh,
            du_eor=self.du_eor,
            dv_eor=self.dv_eor,
            du_fg=self.du_fg,
            dv_fg=self.dv_fg,
            du_sh=self.du_sh if self.du_sh is not None else 1.0,
            dv_sh=self.dv_sh if self.dv_sh is not None else 1.0,
            use_shg=self.use_shg,
            sh_spacing= "linear",
            exclude_monopole=not self.fit_for_monopole,
        )

        self.lmn_model_space = LMN_ModelSpace(
            fov_eor=self.fov_eor,
            fov_fg=self.fov_fg,
            nside=self.nside,
            telescope_latlonalt=self.telescope_latlonalt,
            jd_start=self.jd_start,
            n_t=self.n_t,
            d_t=self.d_t,
        )
        
        self.beam_model = Beam_Model(
            self.beam_type, 
            self.peak_amp, 
            self.fwhm_deg, 
            self.diam, 
            self.cosfreq, 
            self.tanh_freq, 
            self.tanh_sl_red, 
            self.pol, 
            self.freq_interp_kind,
        )

        self.beam_response = self.beam_model.get_beam_vals(self.lmn_model_space.lmn_fg, self.freq_hz) # Shape (n_freq, npix_fg)


    def build_Finv_Fprime(self):
        r"""
        Build the matrix product :math:`\mathbf{F}^{-1}\mathbf{F}'`.
        """

        # Jax Array of the time dependent model lmn coordinates for the EoR and FG models.
        n_pix_fg = self.lmn_model_space.n_pix_fg # Shape (npix_fg,)
        n_pix_eor = self.lmn_model_space.n_pix_eor # Shape (npix_eor,)
        model_lmn_fg_rad = self.lmn_model_space.lmn_eor # Shape (n_t, 3, n_pix_eor)
        model_lmn_eor_rad = self.lmn_model_space.lmn_fg # Shape (n_t, 3, n_pix_fg)
        eor_fg_shared_axis = self.lmn_model_space.eor_fg_shared_axis # Shape (n_pix_fg,)

        sampled_uv_vectors_eor = self.uv_model_space.uv_eor # Shape (2, N_uv_eor)
        sampled_uv_vectors_fg = self.uv_model_space.uv_fg # Shape (2, N_uv_fg)
        # Not currently supporting Subharmonicn Grid (SHG) uv-plane components in the matrix construction. 


        # Frequency-dependent (u, v, w) sampled by the instrument in units of inverse radians
        inst_uvws_irad = self.inst_uvw_irad # Shape (nf, nt, 3, nbls)

        # Construct the dense NUIDFT matrices in Fprime relating the model
        # uv-plane (u, v, f) to the image space (l, m, f) coordinates.
        # EoR model uv-plane components and matrix:
        # Flattened vectors of (u, v) in the EoR model uv-plane
        nuidft_uv_to_lm_eor = build_nudft_array(
            sampled_uv_vectors_eor,
            model_lmn_fg_rad[self.nt//2, :, self.lmn_model_space.eor_fg_shared_axis].T,
            sign=-1
        )
        # FIXME: Fprime normalization no longer needs to include nu*nv norm
        # when using the build_nudft_array function.  The nuidft_matrix_2d
        # function follows the numpy approach and uses 1/Npix for the inverse
        # transform which is why the nu*nv normalization is currently
        # included in self.Fprime_normalization_eor.
        nuidft_uv_to_lm_eor *= (
            self.Fprime_normalization_eor / (self.nu * self.nv)
        )
        
        # eta=0 / FG model uv-plane components and matrix
        model_us_fg_irad, model_vs_fg_irad = sampled_uv_vectors(
            self.nu_fg,
            self.nv_fg,
            du=self.du_fg,
            dv=self.dv_fg,
            exclude_mean=(not self.fit_for_monopole)
        )
        model_uv_fg_irad = np.vstack((model_us_fg_irad, model_vs_fg_irad))
        nuidft_uv_to_lm_fg = build_nudft_array(
            model_uv_fg_irad,
            model_lmnazza_rad[self.nt//2, :2],
            sign=-1
        )
        # FIXME: Fprime normalization no longer needs to include nu*nv norm
        # when using the build_nudft_array function.  The nuidft_matrix_2d
        # function follows the numpy approach and uses 1/Npix for the
        # inverse transform which is why the nu*nv normalization is
        # currently included in self.Fprime_normalization_fg.
        nuidft_uv_to_lm_fg *= (
            self.Fprime_normalization_fg / (self.nu_fg * self.nv_fg)
        )
        # Move the (u, v) = (0, 0) pixel to the rightmost column in
        # nuidft_uv_to_lm_fg so that the LSSM components are all clustered
        # together in the model vector (recall that (u, v) = (0, 0) acts
        # like the constant term in the LSSM).
        mp_column = nuidft_uv_to_lm_fg[:, nuv_fg//2].copy()
        mp_ind = nuv_fg//2
        nuidft_uv_to_lm_fg[:, mp_ind:-1] = nuidft_uv_to_lm_fg[:, mp_ind+1:]
        nuidft_uv_to_lm_fg[:, -1] = mp_column

        # FIXME: add support for the SHG uv-plane (issue #50)
        # if self.use_shg:
        #     nuidft_uv_to_lm_shg = ?

        # Instantiate empty containers for matrices
        Finv_Fprime = sparse.lil_matrix(Finv_Fprime_shape, dtype=complex)
        
        matrix_name = "Finv_Fprime"
        # For now, there are no prerequisites for this matrix.  But,
        # this might change in the future if we add a kwarg for writing out
        # intermediate matrices to disk (e.g. the dense blocks that comprise
        # Fprime).
        pmd = self.load_prerequisites(matrix_name)
        if self.verbose:
            start = time.time()
            print("Performing matrix algebra")
        
        for i_f in range(self.nf):
            uv_Fprime_inds_eor = slice(
                i_f*nuv_eor, (i_f + 1)*nuv_eor
            )
            uv_Fprime_inds_fg = slice(
                self.nf*nuv_eor + i_f*nuv_fg,
                self.nf*nuv_eor + (i_f + 1)*nuv_fg
            )
            # FIXME: add support for the SHG uv-plane (issue #50)
            # if self.use_shg:
            #     uv_Fprime_inds_shg = slice(
            #         self.nf*(nuv_eor + nuv_fg) + i_f*?,
            #         self.nf*(nuv_eor + nuv_fg) + (i_f + 1)*?
            #     )
            for i_t in range(self.nt):
                vis_inds = slice(
                    i_t*self.nf*self.nbls + i_f*self.nbls,
                    i_t*self.nf*self.nbls + (i_f + 1)*self.nbls
                )

                if self.drift_scan:
                    # For a drift scan observation, the beam drifts across the
                    # modelled patch of sky, so the (l, m, n) coordinates for
                    # a given HEALPix pixel change with time.
                    lmn_rad = model_lmnazza_rad[i_t, :3]
                    az_rad = model_lmnazza_rad[i_t, 3]
                    za_rad = model_lmnazza_rad[i_t, 4]
                else:
                    # For modelling phased observations, the beam is always
                    # phased to the center of the model FoV.  The (l, m, n)
                    # coordinates in this case are independent of time.
                    lmn_rad = model_lmnazza_rad[self.nt//2, :3]
                    az_rad = model_lmnazza_rad[self.nt//2, 3]
                    za_rad = model_lmnazza_rad[self.nt//2, 4]

                # NUDFT matrix for image space (l, m, n) to measurement
                # space (u, v, w) sampled by the instrument
                nudft_lmn_to_vis = build_nudft_array(
                    lmn_rad,
                    inst_uvw_irad[i_f, 0],
                    sign=+1
                )
                nudft_lmn_to_vis *= self.Finv_normalisation
                # Beam amplitude at each (l, m, n)
                beam_vals = self.hpx.get_beam_vals(
                    az_rad, za_rad, freq=self.freqs_hertz[i_f]
                )
                # Equivalent to np.dot(nudft_lmn_to_vis, np.diag(beam_vals))
                Finv_block = nudft_lmn_to_vis * beam_vals[None, :]

                # FIXME: add support for modelling phased visibilities via
                # the phasor vector.

                Finv_Fprime[vis_inds, uv_Fprime_inds_eor] = np.dot(
                    Finv_block[:, self.hpx.eor_to_fg_pix],
                    nuidft_uv_to_lm_eor
                )
                Finv_Fprime[vis_inds, uv_Fprime_inds_fg] = np.dot(
                    Finv_block,
                    nuidft_uv_to_lm_fg
                )
                # FIXME: add support for the SHG uv-plane (issue #50)
                # if self.use_shg:
                #     Finv_Fprime[vis_inds, uv_Fprime_inds_shg] = ?

        if self.verbose:
            print(f"Time taken: {time.time() - start}")
        self.output_data(Finv_Fprime, matrix_name)




    

        












