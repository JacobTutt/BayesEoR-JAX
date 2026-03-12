"""
High-level orchestration for constructing model matrices and associated kernels.

This module wires together UV/LMN model spaces, beam evaluation, and NUDFT
operators to build the dense Finv/Fprime blocks used downstream.
"""
import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jax.experimental import sparse
from jax.experimental.pjit import pjit
from functools import partial

from astropy.constants import c

from .dft_funcs import (
    inverse_udft_1d_rect,
    build_nudft_matrix,
    reorder_fuv_to_uvf,
)
from .lssm_model_space import build_lssm_basis
from .uv_model_space import UV_ModelSpace
from .lmn_model_space import LMN_ModelSpace
from .beam_model import Beam_Model
from .jax_grid import MeshManager

class BuildMatrices:
    """
    Orchestrate model-space construction and build NUDFT-based matrices.

    This class centralizes:
    - UV model grids for EoR/FG.
    - LMN image-domain coordinates across time.
    - Beam evaluation across time/frequency.
    - NUDFT kernels and the Finv/Fprime blocks.

    Parameters
    ----------
    EoR model uv-plane parameters
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

    Foreground model uv-plane parameters
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

    Frequency and line of sight Fourier mode parameters
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

    Model parameters
    fit_for_monopole : bool, optional
        Fit for (u, v) = (0, 0) (True) or exclude it from the fit (False).
        Defaults to False.

    HEALPix image-domain model parameters
    telescope_latlonalt : tuple, optional
        The latitude, longitude, and altitude of the telescope in degrees,
        degrees, and meters, respectively. Defaults to HERA location.
    jd_center : float
        Central time of the observation in JD units.
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
    
    Beam model parameters
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

    Data parameters
    uvw_array_m : numpy.ndarray
        Array containing the (u(t), v(t), w(t)) coordinates of the instrument
        model with shape (nt, nbls, 3). Defaults to None.
    bl_red_array : numpy.ndarray
        Array containing the number of redundant baselines at each
        (u(t), v(t), w(t)) in the instrument model with shape (nt, nbls, 1).
        Defaults to None.
    sigma : float, optional
        RMS noise per visibility before redundancy weighting. Required if
        `effective_noise` is None.
    effective_noise : numpy.ndarray, optional
        Optional array of per-visibility noise standard deviations. If
        provided, it overrides `sigma` and redundancy weighting.
    array_save_directory : str, optional
        Path to the directory where sparse matrices will be saved.
        Defaults to './matrices/'.
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
        n_freq: int = 0,
        f0_mhz: Optional[float] = None,
        df_mhz: Optional[float] = None,
        n_eta: int = 0,
        d_eta: float = 0.0,
        fit_for_monopole: bool = False,
        telescope_latlonalt: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        jd_center: float = 2461040.5,
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
        sigma: Optional[float] = None,
        effective_noise: Optional[jnp.ndarray] = None,
        array_save_directory: str = "./matrices/",
    ) -> None:
        
        # Store EoR UV-plane parameters.
        self.n_u_eor = n_u_eor
        self.du_eor = du_eor
        self.n_v_eor = n_v_eor
        self.dv_eor = dv_eor
        # Total EoR uv modes (monopole always excluded).
        self.n_uv_eor = (n_u_eor * n_v_eor) - 1

        # Store foreground UV-plane parameters and LSSM settings.
        self.n_u_fg = n_u_fg
        self.du_fg = du_fg
        self.n_v_fg = n_v_fg
        self.dv_fg = dv_fg
        self.n_modes_fg = n_modes_fg
        self.n_powerlaws_fg = n_powerlaws_fg
        self.beta_fg = beta_fg if beta_fg is not None else [2.63, 2.82]
        self.fit_for_monopole = fit_for_monopole
        # Total FG uv modes (optional monopole exclusion).
        self.n_uv_fg = (n_u_fg * n_v_fg) - int(not self.fit_for_monopole)

        # Store frequency/eta grid parameters.
        self.n_freq = n_freq
        self.f0_mhz = f0_mhz
        self.df_mhz = df_mhz    
        self.n_eta = n_eta
        self.d_eta = d_eta
        # Frequency array in Hz (from MHz).
        self.freq_hz = (f0_mhz + df_mhz * jnp.arange(n_freq, dtype=jnp.float64)) * 1e6 
        c_ms = float(c.to("m/s").value)

        # Store HEALPix image-domain parameters.
        self.nside = nside
        self.fov_eor = fov_eor
        self.fov_fg = fov_fg
        self.telescope_latlonalt = telescope_latlonalt
        self.jd_center = jd_center
        self.n_t = n_t
        self.d_t = d_t


        # Store beam model parameters.
        self.beam_type = beam_type
        self.peak_amp = peak_amp
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red
        self.pol = pol
        self.freq_interp_kind = freq_interp_kind

        # Instrument data: convert uvw from meters to inverse radians for each frequency.
        uvw_array_m = jnp.array(uvw_array_m)  # (n_t, n_bls, 3)
        self.nbls = int(uvw_array_m.shape[1])
        # Only the first time-step uvw is used; time dependence is handled by beam rotation.
        uvw_array_m = uvw_array_m[0].T  # (3, n_bls)
        # Convert meters -> inverse radians by dividing by wavelength per frequency.
        wavelengths = c_ms / self.freq_hz
        self.inst_uvw_irad = uvw_array_m[jnp.newaxis, :, :] / wavelengths[:, jnp.newaxis, jnp.newaxis]

        self.bl_red_array = bl_red_array
        self.sigma = sigma
        self.effective_noise = effective_noise
        self.array_save_directory = Path(array_save_directory)
        self.array_save_directory.mkdir(parents=True, exist_ok=True)

        # UV model space (EoR + FG rectilinear grids).
        self.uv_model_space = UV_ModelSpace(
            n_u_eor=self.n_u_eor,
            n_v_eor=self.n_v_eor,
            n_u_fg=self.n_u_fg,
            n_v_fg=self.n_v_fg,
            du_eor=self.du_eor,
            dv_eor=self.dv_eor,
            du_fg=self.du_fg,
            dv_fg=self.dv_fg,
            exclude_monopole=not self.fit_for_monopole,
        )

        # LMN model space: FG pixel set and shared EoR mask (computed at init).
        self.lmn_model_space = LMN_ModelSpace(
            fov_eor=self.fov_eor,
            fov_fg=self.fov_fg,
            nside=self.nside,
            telescope_latlonalt=self.telescope_latlonalt,
            jd_center=self.jd_center,
            n_t=self.n_t,
            d_t=self.d_t,
        )

        # Beam model: evaluate beam response for FG pixels across time/frequency.
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
        self.beam_response = self.beam_model.get_beam_vals(
            self.lmn_model_space.az_za_fg, self.freq_hz
            )  # Shape (n_freq, n_t, n_pix_fg)

        # Initialize sharding utilities.
        self.mm = MeshManager()
        self.sharding_grid = self.mm.get_sharding(rank=4)
        self.sharding_vec = self.mm.get_sharding(rank=1)
        self.sharding_mat = self.mm.get_sharding(rank=2)
        self._build_t_directly_single_device_fn = None
        self._build_t_directly_multi_device_fn = None


    # ---------------------------
    # Construction of Finv/Fprime
    # ---------------------------
    def _uv_to_lm_eor(self):
        """
        Construct the EoR NUDFT kernel (model UV -> image LM/RaDec).
        """
        # 2D NUDFT for UV -> LM at the central time step (EoR pixel set).
        lmn_center = self.lmn_model_space.lmn_fg[self.lmn_model_space.time_ind_center, :2]
        lmn_center = lmn_center[:, self.lmn_model_space.eor_fg_shared_axis]
        nuidft_uv_to_lm_eor = build_nudft_matrix(
            self.uv_model_space.uv_eor,  # (2, n_uv_eor)
            lmn_center,  # (2, n_pix_eor)
            sign=-1
        )  # (n_pix_eor, n_uv_eor)
        # Pixel area normalization in UV.
        nuidft_uv_to_lm_eor *= (self.du_eor * self.dv_eor)

        return nuidft_uv_to_lm_eor

    def _uv_to_lm_fg(self):
        """
        Construct the FG NUDFT kernel (model UV -> image LM/RaDec).

        If monopole is included, the (u, v) = (0, 0) column is moved to the
        rightmost position to keep it adjacent to LSSM modes downstream in the
        `UV_ModelSpace` ordering.
        """
        # 2D NUDFT for UV -> LM at the central time step (FG pixel set).
        nuidft_uv_to_lm_fg = build_nudft_matrix(
            self.uv_model_space.uv_fg,  # (2, n_uv_fg)
            self.lmn_model_space.lmn_fg[self.lmn_model_space.time_ind_center, :2],  # (2, n_pix_fg)
            sign=-1
        )  # (n_pix_fg, n_uv_fg)

        # Pixel area normalization in UV.
        nuidft_uv_to_lm_fg *= (self.du_fg * self.dv_fg)
        
        return nuidft_uv_to_lm_fg
    
    def _build_Finv_Fprime_block(self, nuidft_uv_to_lm_eor, nuidft_uv_to_lm_fg, i_f, i_t):
        """
        Build the (i_f, i_t) block of Finv/Fprime.

        This maps model uv coordinates at frequency i_f into LM space at time i_t,
        applies the beam, and projects back into the model uv grids.
        """
        # LMN coordinates for time step i_t (FG pixel set).
        lmn_fg_it = self.lmn_model_space.lmn_fg[i_t, :, :]  # (3, n_pix_fg)
        # UVW coordinates for frequency i_f.
        uvw_irad_if = self.inst_uvw_irad[i_f, :, :]  # (3, n_bls)
        # NUDFT: LMN -> UVW at this (i_f, i_t).
        nudft_lmn_to_uvw = build_nudft_matrix(
            lmn_fg_it,  # (3, n_pix_fg)
            uvw_irad_if,  # (3, n_bls)
            sign=+1,
        )
        # Pixel area normalization in image domain.
        nudft_lmn_to_uvw *= self.lmn_model_space.pixel_area_sr  # (n_bls, n_pix_fg)

        # Beam response for this time/frequency.
        beam_it = self.beam_response[i_f, i_t, :]  # (n_pix_fg,)

        # Apply beam weighting in LMN.
        beam_nudft_lmn_to_uvw = nudft_lmn_to_uvw * beam_it[jnp.newaxis, :]  # (n_bls, n_pix_fg)

        # Project into FG and EoR uv grids.
        Finv_Fprime_block_fg = jnp.dot(beam_nudft_lmn_to_uvw, nuidft_uv_to_lm_fg)  # (n_bls, n_uv_fg)
        Finv_Fprime_block_eor = jnp.dot(
            beam_nudft_lmn_to_uvw[:, self.lmn_model_space.eor_fg_shared_axis],
            nuidft_uv_to_lm_eor,
        )  # (n_bls, n_uv_eor)

        return Finv_Fprime_block_eor, Finv_Fprime_block_fg

    def _build_finv_fprime_blocks(self):
        """
        Build dense per-frequency/per-time Finv_Fprime blocks.

        Notes
        -----
        This function is written as local vectorized batching over a flattened
        (frequency, time) index grid. When called inside a ``pjit`` context,
        cross-device distribution is handled by ``pjit``/mesh sharding, while
        this ``vmap`` handles per-device batching.

        Returns
        -------
        full_eor, full_fg : tuple[jnp.ndarray, jnp.ndarray]
            Tensors with shapes:
            - full_eor: (n_freq, n_t, n_bls, n_uv_eor)
            - full_fg:  (n_freq, n_t, n_bls, n_uv_fg)
        """
        # Precompute NUDFT kernels (UV -> LM).
        nuidft_uv_to_lm_eor = self._uv_to_lm_eor()
        nuidft_uv_to_lm_fg = self._uv_to_lm_fg()

        # Vectorized construction over a flattened (frequency, time) grid.
        t_idx = jnp.arange(self.n_t, dtype=jnp.int32)
        f_idx = jnp.arange(self.n_freq, dtype=jnp.int32)
        f_grid = jnp.repeat(f_idx, self.n_t)
        t_grid = jnp.tile(t_idx, self.n_freq)

        def block_fn(i_f, i_t):
            return self._build_Finv_Fprime_block(
                nuidft_uv_to_lm_eor, nuidft_uv_to_lm_fg, i_f, i_t
            )

        flat_eor, flat_fg = jax.vmap(block_fn, in_axes=(0, 0), out_axes=(0, 0))(f_grid, t_grid)
        
        full_eor = flat_eor.reshape(self.n_freq, self.n_t, self.nbls, self.n_uv_eor)
        full_fg = flat_fg.reshape(self.n_freq, self.n_t, self.nbls, self.n_uv_fg)

        return full_eor, full_fg


    @partial(jax.jit, static_argnums=(0,))
    def build_Finv_Fprime(self) -> jnp.ndarray:
        """
        Build Finv/Fprime blocks across frequency and time.

        Uses sharded NUDFT kernels and nested scans to reduce memory usage.

        Returns
        -------
        finv_fprime : jnp.ndarray
            Dense matrix with shape (n_vis, n_params), where
            n_vis = n_freq * n_t * n_bls and
            n_params = n_freq * (n_uv_eor + n_uv_fg).
        """
        full_eor, full_fg = self._build_finv_fprime_blocks()

        # Preserve legacy column layout from the sparse implementation:
        # [all EoR frequency blocks][all FG frequency blocks].
        freq_eye = jnp.eye(self.n_freq, dtype=full_eor.dtype)[:, None, None, :, None]  # (f,1,1,f,1)
        eor_blocked = full_eor[:, :, :, None, :] * freq_eye  # (f,t,b,f,n_uv_eor)
        fg_blocked = full_fg[:, :, :, None, :] * freq_eye  # (f,t,b,f,n_uv_fg)

        # Legacy row ordering: time slowest, then frequency, then baseline.
        eor_blocked = jnp.transpose(eor_blocked, (1, 0, 2, 3, 4)).reshape(
            self.n_t * self.n_freq * self.nbls,
            self.n_freq * self.n_uv_eor,
        )
        fg_blocked = jnp.transpose(fg_blocked, (1, 0, 2, 3, 4)).reshape(
            self.n_t * self.n_freq * self.nbls,
            self.n_freq * self.n_uv_fg,
        )
        return jnp.concatenate([eor_blocked, fg_blocked], axis=1)


    # ----------------------------
    # Construction of Fz (eta -> f)
    # ----------------------------
    def build_eta_to_freq_idft_eor(self) -> jnp.ndarray:
        r"""
        Build the EoR IDFT block for eta -> frequency.

        Returns a block-diagonal matrix with one IDFT block per (u, v) pixel.
        Excludes eta=0 (belongs to the FG model).

        Returns
        -------
        idft_eor : jnp.ndarray
            Block-diagonal dense matrix of shape
            (n_uv_eor * n_freq, n_uv_eor * (n_eta - 1)).
        """
        idft_array_1d = inverse_udft_1d_rect(
            self.n_freq,
            self.n_eta,
            include_eta0=False
        )
        # Normalise by eta spacing.
        idft_array_1d *= self.d_eta

        return jnp.kron(jnp.eye(self.n_uv_eor, dtype=idft_array_1d.dtype), idft_array_1d)

    def build_eta_to_freq_idft_fg(self) -> jnp.ndarray:
        r"""
        Build the FG IDFT block for eta -> frequency.

        Includes eta=0 terms, optional LSSM modes, and an optional monopole block.

        Returns
        -------
        idft_fg : jnp.ndarray
            Block-diagonal dense matrix for the FG model with shape
            (n_uv_fg * n_freq, n_eta0 + n_lssm + n_mp), where:
            n_eta0 = (n_uv_fg - int(fit_for_monopole)) * 1
            n_lssm = (n_uv_fg - int(fit_for_monopole)) * n_modes_fg
            n_mp = fit_for_monopole * (n_eta + n_modes_fg)
        """

        # eta=0 column mapped into frequency space.
        eta0_to_frequency = jnp.ones((self.n_freq, 1), dtype=jnp.complex128) * self.d_eta * self.n_eta

        # One eta=0 block per FG (u, v), excluding monopole if fitted separately.
        neta0_blocks = self.n_uv_fg - int(self.fit_for_monopole)

        idft_array_1d_fg = jnp.kron(
            jnp.eye(neta0_blocks, dtype=eta0_to_frequency.dtype),
            eta0_to_frequency,
        )

        if self.n_modes_fg > 0:
            lssm_basis_vecs = build_lssm_basis(
                n_freq=self.n_freq,
                n_modes=self.n_modes_fg,
                n_powerlaws=self.n_powerlaws_fg,
                f0_mhz=self.f0_mhz,
                df_mhz=self.df_mhz,
                beta=self.beta_fg,
            )
            # Normalise LSSM basis.
            lssm_basis_vecs *= self.d_eta * self.n_eta

            # Repeat for each non-monopole FG (u, v).
            lssm_array = jnp.kron(
                jnp.eye(neta0_blocks, dtype=lssm_basis_vecs.dtype),
                lssm_basis_vecs,
            )
            idft_array_1d_fg = jnp.hstack([idft_array_1d_fg, lssm_array])
            
        if self.fit_for_monopole:
            idft_array_1d_mp = inverse_udft_1d_rect(
                n_freq=self.n_freq,
                n_eta=self.n_eta,
                n_modes=self.n_modes_fg,
                n_powerlaws=self.n_powerlaws_fg,
                f0_mhz=self.f0_mhz,
                df_mhz=self.df_mhz,
                beta=self.beta_fg,
            )
            # Monopole block includes eta=0 and all LSSM modes.
            idft_array_1d_mp *= self.d_eta
            idft_array_1d_mp = idft_array_1d_mp.at[:, self.n_eta // 2].multiply(self.n_eta)
            idft_array_1d_mp = idft_array_1d_mp.at[:, self.n_eta:].multiply(self.n_eta)
            top = jnp.hstack(
                [
                    idft_array_1d_fg,
                    jnp.zeros((idft_array_1d_fg.shape[0], idft_array_1d_mp.shape[1]), dtype=idft_array_1d_fg.dtype),
                ]
            )
            bottom = jnp.hstack(
                [
                    jnp.zeros((idft_array_1d_mp.shape[0], idft_array_1d_fg.shape[1]), dtype=idft_array_1d_mp.dtype),
                    idft_array_1d_mp,
                ]
            )
            idft_array_1d_fg = jnp.vstack([top, bottom])

        return idft_array_1d_fg

    def reorder_fuv_to_uvf_eor(self) -> jnp.ndarray:
        r"""
        Build the vis-ordered -> chan-ordered gridding matrix for EoR.

        This maps UV-major ordering to frequency-major ordering.

        Returns
        -------
        gridding_matrix : jnp.ndarray
            Dense permutation matrix of shape (n_uv_eor * n_freq, n_uv_eor * n_freq).
        """
        gridding_matrix = reorder_fuv_to_uvf(self.n_uv_eor, self.n_freq)

        return gridding_matrix

    
    def reorder_fuv_to_uvf_fg(self) -> jnp.ndarray:
        r"""
        Build the vis-ordered -> chan-ordered gridding matrix for FG.

        This maps UV-major ordering to frequency-major ordering.

        Returns
        -------
        gridding_matrix : jnp.ndarray
            Dense permutation matrix of shape (n_uv_fg * n_freq, n_uv_fg * n_freq).
        """

        gridding_matrix = reorder_fuv_to_uvf(self.n_uv_fg, self.n_freq)

        return gridding_matrix      

    def build_Fz(self) -> jnp.ndarray:
        r"""
        Build the full Fz operator (EoR + FG).

        Fz is a block-diagonal operator that maps eta-space vectors
        (vis-ordered) to frequency-space vectors (chan-ordered). The
        returned operator is stored as a dense matrix.

        Returns
        -------
        Fz : jnp.ndarray
            Dense matrix of shape (n_freq * (n_uv_eor + n_uv_fg), n_params_eta),
            where n_params_eta = n_uv_eor * (n_eta - 1) + n_eta0 + n_lssm + n_mp.
        """
        # Build EoR/FG IDFT blocks and gridding matrices directly, then
        # map from vis-ordered eta space to chan-ordered frequency space.
        idft_eor = self.build_eta_to_freq_idft_eor()
        idft_fg = self.build_eta_to_freq_idft_fg()
        grid_eor = self.reorder_fuv_to_uvf_eor()
        grid_fg = self.reorder_fuv_to_uvf_fg()

        Fz_eor = grid_eor @ idft_eor
        Fz_fg = grid_fg @ idft_fg
        top = jnp.hstack(
            [
                Fz_eor,
                jnp.zeros((Fz_eor.shape[0], Fz_fg.shape[1]), dtype=Fz_eor.dtype),
            ]
        )
        bottom = jnp.hstack(
            [
                jnp.zeros((Fz_fg.shape[0], Fz_eor.shape[1]), dtype=Fz_fg.dtype),
                Fz_fg,
            ]
        )
        return jnp.vstack([top, bottom])

    # --------------------------------------
    # Construction of T = Finv @ Fprime @ Fz
    # --------------------------------------

    def build_T(self) -> jnp.ndarray:
        r"""
        Build the full T operator for EoR + FG.

        T = (F^{-1} F') @ Fz

        Components
        ----------
        F^{-1} (Finv):
            Maps image-space (l, m, n) to measured visibilities (u, v, w)
            for each time/frequency.
        F' (Fprime):
            Maps model UV parameters to image-space (l, m) at the
            reference time step.
        Fz:
            Maps eta-space vectors (vis-ordered) to frequency-space
            vectors (chan-ordered).

        Returns
        -------
        Finv_Fprime_Fz : jnp.ndarray
            Dense matrix with shape
            (n_vis, n_params_eta), where
            n_vis = n_freq * n_t * n_bls and
            n_params_eta = n_uv_eor * (n_eta - 1) + n_eta0 + n_lssm + n_mp.
        """
        Finv_Fprime = self.build_Finv_Fprime()
        Fz = self.build_Fz()

        return Finv_Fprime @ Fz

    def _build_T_directly_impl(self) -> jnp.ndarray:
        r"""
        Construct ``T = (F^{-1}F')F_z`` directly, without explicitly building ``F_z``.

        Intuition
        ---------
        ``F^{-1}F'`` gives, for each observing frequency ``f``, a dense UV response
        block over ``(t, bl, uv)``:

        - ``full_eor[f, t, bl, u_eor]``
        - ``full_fg[f,  t, bl, u_fg]``

        ``F_z`` then only applies 1D spectral basis expansions per UV mode:

        - EoR: eta basis with eta=0 removed
        - FG non-monopole: ``[eta0 | LSSM]``
        - FG monopole (optional): ``[full eta | LSSM]`` monopole block

        So instead of forming huge block-diagonal/permutation matrices and
        multiplying, we apply those basis vectors directly via broadcasted
        products, then reshape/concatenate into the same final column ordering
        as ``build_T``.

        Returns
        -------
        T : jnp.ndarray
            Dense matrix with shape
            (n_vis, n_params_eta), where
            n_vis = n_freq * n_t * n_bls and
            n_params_eta = n_uv_eor * (n_eta - 1) + n_eta0 + n_lssm + n_mp.
        """
        full_eor, full_fg = self._build_finv_fprime_blocks()

        # ------------------------
        # EoR contribution to T
        # ------------------------
        # Build frequency->eta basis for EoR (eta=0 excluded).
        eor_basis = inverse_udft_1d_rect(
            self.n_freq,
            self.n_eta,
            include_eta0=False,
        )
        eor_basis = eor_basis * self.d_eta  # (n_freq, n_eta-1)

        # Apply basis per-frequency:
        # full_eor[f,t,bl,u] * eor_basis[f,eta] -> t_eor[f,t,bl,u,eta]
        t_eor = full_eor[..., None] * eor_basis[:, None, None, None, :]
        # Flatten to legacy row order: time slowest, then frequency, then baseline.
        t_eor = jnp.transpose(t_eor, (1, 0, 2, 3, 4)).reshape(
            self.n_t * self.n_freq * self.nbls,
            self.n_uv_eor * (self.n_eta - 1),
        )

        # -----------------------
        # FG contribution to T
        # -----------------------
        n_non_mp = self.n_uv_fg - int(self.fit_for_monopole)
        fg_non_mp = full_fg[..., :n_non_mp] if self.fit_for_monopole else full_fg

        n_rows = self.n_t * self.n_freq * self.nbls

        # FG eta0 columns for each non-monopole UV mode.
        eta0_basis = (
            jnp.ones((self.n_freq, 1), dtype=full_fg.dtype)
            * self.d_eta
            * self.n_eta
        )  # (n_freq, 1)
        t_fg_eta0 = fg_non_mp * eta0_basis[:, None, None, :]
        t_fg_eta0 = jnp.transpose(t_fg_eta0, (1, 0, 2, 3)).reshape(
            n_rows,
            n_non_mp,
        )

        # FG LSSM columns for each non-monopole UV mode.
        if self.n_modes_fg > 0:
            lssm_basis = build_lssm_basis(
                n_freq=self.n_freq,
                n_modes=self.n_modes_fg,
                n_powerlaws=self.n_powerlaws_fg,
                f0_mhz=self.f0_mhz,
                df_mhz=self.df_mhz,
                beta=self.beta_fg,
            )
            lssm_basis = lssm_basis * self.d_eta * self.n_eta  # (n_freq, n_modes_fg)
            t_fg_lssm = fg_non_mp[..., None] * lssm_basis[:, None, None, None, :]
            t_fg_lssm = jnp.transpose(t_fg_lssm, (1, 0, 2, 3, 4)).reshape(
                n_rows,
                n_non_mp * self.n_modes_fg,
            )
        else:
            t_fg_lssm = jnp.zeros((n_rows, 0), dtype=full_fg.dtype)

        # FG monopole block (eta + LSSM), if enabled.
        if self.fit_for_monopole:
            mp_basis = inverse_udft_1d_rect(
                n_freq=self.n_freq,
                n_eta=self.n_eta,
                n_modes=self.n_modes_fg,
                n_powerlaws=self.n_powerlaws_fg,
                f0_mhz=self.f0_mhz,
                df_mhz=self.df_mhz,
                beta=self.beta_fg,
            )
            mp_basis = mp_basis * self.d_eta
            mp_basis = mp_basis.at[:, self.n_eta // 2].multiply(self.n_eta)
            mp_basis = mp_basis.at[:, self.n_eta:].multiply(self.n_eta)

            fg_mp = full_fg[..., -1]  # monopole mode is last FG uv column
            t_fg_mp = fg_mp[..., None] * mp_basis[:, None, None, :]
            t_fg_mp = jnp.transpose(t_fg_mp, (1, 0, 2, 3)).reshape(
                n_rows,
                mp_basis.shape[1],
            )
        else:
            t_fg_mp = jnp.zeros((n_rows, 0), dtype=full_fg.dtype)

        # Legacy FG column ordering.
        t_fg = jnp.concatenate([t_fg_eta0, t_fg_lssm, t_fg_mp], axis=1)

        # Final column ordering matches legacy Fz block layout:
        # [EoR columns][FG columns].
        return jnp.concatenate([t_eor, t_fg], axis=1)

    def build_T_directly(self) -> jnp.ndarray:
        """
        Build T using an automatically selected JAX executor.

        - single-device runs: compile/cache a ``jax.jit`` executor
        - multi-device runs:  compile/cache a ``pjit`` executor and run in mesh context
        """
        if self.mm.n_dev > 1:
            if self._build_t_directly_multi_device_fn is None:
                self._build_t_directly_multi_device_fn = pjit(
                    BuildMatrices._build_T_directly_impl,
                    out_shardings=self.sharding_mat,
                    static_argnums=(0,),
                )
            with self.mm.mesh:
                return self._build_t_directly_multi_device_fn(self)

        if self._build_t_directly_single_device_fn is None:
            self._build_t_directly_single_device_fn = jax.jit(
                BuildMatrices._build_T_directly_impl,
                static_argnums=(0,),
            )
        return self._build_t_directly_single_device_fn(self)

    # ------------------------
    # Noise covariance matrices
    # ------------------------
    def _sigma_squared_array(self) -> jnp.ndarray:
        """
        Build the per-visibility noise variance vector for the current settings.

        Only the (include_instrumental_effects=True, drift_scan=True) path is supported.
        """
        if self.effective_noise is not None:
            return jnp.abs(self.effective_noise) ** 2

        if self.sigma is None:
            raise ValueError("sigma must be provided when effective_noise is None.")

        # Expect bl_red_array shaped (n_t, n_bls) or (n_t, n_bls, 1).
        red = jnp.asarray(self.bl_red_array)
        if red.ndim == 3 and red.shape[-1] == 1:
            red = red[..., 0]
        if red.shape[0] != self.n_t or red.shape[1] != self.nbls:
            raise ValueError("bl_red_array must have shape (n_t, n_bls) or (n_t, n_bls, 1).")

        # Expand redundancy over frequency: (n_t, n_freq, n_bls) then flatten.
        red_tf = jnp.repeat(red[:, jnp.newaxis, :], self.n_freq, axis=1)
        red_tf = red_tf.reshape(-1)

        sigma_redundancy = self.sigma / jnp.sqrt(red_tf)
        return (sigma_redundancy ** 2).astype(jnp.complex128)

    def build_Ninv(self) -> jnp.ndarray:
        """
        Build the inverse noise covariance matrix N^{-1}.

        Returns
        -------
        Ninv : jnp.ndarray
            Dense diagonal matrix of shape (n_vis, n_vis), where
            n_vis = n_freq * n_t * n_bls.
        """
        sigma_sq = self._sigma_squared_array()
        return jnp.diag(1.0 / sigma_sq)

    def build_N(self) -> jnp.ndarray:
        """
        Build the noise covariance matrix N.

        Returns
        -------
        N : jnp.ndarray
            Dense diagonal matrix of shape (n_vis, n_vis), where
            n_vis = n_freq * n_t * n_bls.
        """
        sigma_sq = self._sigma_squared_array()
        return jnp.diag(sigma_sq)    

    # ------------------
    # Build T^H N^{-1} T
    # ------------------

    def build_T_Ninv_T(self, T, Ninv: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build T^H N^{-1} T.

        Parameters
        ----------
        T : jnp.ndarray
            Forward operator with shape (n_vis, n_params_eta).
        Ninv : jnp.ndarray
            Inverse noise covariance with shape (n_vis, n_vis).

        Returns
        -------
        T_Ninv_T : jnp.ndarray
            Dense matrix with shape (n_params_eta, n_params_eta), where 
            n_params_eta = n_uv_eor * (n_eta - 1) + n_eta0 + n_lssm + n_mp.
        """
        
        T = jnp.asarray(T)
        Ninv_T = jnp.asarray(Ninv) @ T

        # T^H @ (N^{-1} T) in dense space for robust complex-conjugate handling.
        return jnp.conj(T).T @ Ninv_T

    # --------------------
    # Matrix save/load API
    # --------------------
    def _matrix_path(self, matrix_name: str) -> Path:
        """Return the .npz path for a stored sparse matrix."""
        return self.array_save_directory / f"{matrix_name}.npz"

    def _dense_matrix_path(self, matrix_name: str) -> Path:
        """Return the .npy path for a stored dense matrix."""
        return self.array_save_directory / f"{matrix_name}.npy"

    def check_if_matrix_exists(self, matrix_name: str) -> bool:
        """Return True if a saved matrix exists on disk."""
        return self._matrix_path(matrix_name).exists() or self._dense_matrix_path(matrix_name).exists()

    def save_sparse_matrix(self, matrix_name: str, matrix: sparse.BCOO) -> None:
        """Save a BCOO matrix to disk as data/indices/shape npz."""
        path = self._matrix_path(matrix_name)
        np.savez(
            path,
            data=np.array(matrix.data),
            indices=np.array(matrix.indices),
            shape=np.array(matrix.shape, dtype=np.int64),
        )

    def save_dense_matrix(self, matrix_name: str, matrix: jnp.ndarray) -> None:
        """Save a dense matrix to disk as .npy."""
        path = self._dense_matrix_path(matrix_name)
        np.save(path, np.array(matrix))

    def save_matrix(self, matrix_name: str, matrix) -> None:
        """Save a matrix, sparse as .npz or dense as .npy."""
        if isinstance(matrix, sparse.BCOO):
            self.save_sparse_matrix(matrix_name, matrix)
        else:
            self.save_dense_matrix(matrix_name, matrix)

    def load_sparse_matrix(self, matrix_name: str) -> sparse.BCOO:
        """Load a BCOO matrix from disk."""
        path = self._matrix_path(matrix_name)
        with np.load(path) as npz:
            data = jnp.array(npz["data"])
            indices = jnp.array(npz["indices"])
            shape = tuple(npz["shape"].tolist())
        return sparse.BCOO((data, indices), shape=shape, unique_indices=True)

    def load_dense_matrix(self, matrix_name: str) -> jnp.ndarray:
        """Load a dense matrix from disk."""
        path = self._dense_matrix_path(matrix_name)
        return jnp.array(np.load(path))

    def load_matrix(self, matrix_name: str):
        """Load a matrix (sparse or dense) from disk."""
        matrix = None
        if self._matrix_path(matrix_name).exists():
            matrix = self.load_sparse_matrix(matrix_name)
        elif self._dense_matrix_path(matrix_name).exists():
            matrix = self.load_dense_matrix(matrix_name)
        else:
            raise FileNotFoundError(f"No saved matrix found for '{matrix_name}'.")
        return matrix

    def build_matrix_if_not_here(self, matrix_name: str) -> None:
        r"""
        Build and save a matrix if it does not already exist on disk.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix to build and save.
        """
        if not self.check_if_matrix_exists(matrix_name):
            matrix = self._build_matrix_by_name(matrix_name)
            self.save_matrix(matrix_name, matrix)

    def _build_matrix_by_name(self, matrix_name: str):
        """Build a supported matrix by name."""
        matrix = None
        if matrix_name == "Fz":
            matrix = self.build_Fz()
        elif matrix_name == "Finv_Fprime":
            matrix = self.build_Finv_Fprime()
        elif matrix_name == "T":
            matrix = self.build_T_directly()
        elif matrix_name == "Ninv":
            matrix = self.build_Ninv()
        elif matrix_name == "N":
            matrix = self.build_N()
        elif matrix_name == "T_Ninv_T":
            T = self.load_matrix("T")
            Ninv = self.load_matrix("Ninv")
            try:
                matrix = self.build_T_Ninv_T(T, Ninv)
            finally:
                # Drop large intermediates as soon as possible.
                del T
                del Ninv
        else:
            raise ValueError(f"Unsupported matrix name '{matrix_name}'.")
        return matrix

    def build_minimum_matrix_stack(self) -> None:
        """
        Build and save T, Ninv, then T_Ninv_T in that order.

        This uses a disk-first workflow to avoid keeping large matrices in
        memory longer than required.
        """
        # Ensure prerequisites exist on disk.
        self.build_matrix_if_not_here("T")
        self.build_matrix_if_not_here("Ninv")

        # Build dense T_Ninv_T only if absent.
        if not self.check_if_matrix_exists("T_Ninv_T"):
            T_Ninv_T = self._build_matrix_by_name("T_Ninv_T")
            try:
                self.save_matrix("T_Ninv_T", T_Ninv_T)
            finally:
                del T_Ninv_T
    
    
