"""
High-level orchestration for constructing model matrices and associated kernels.

This module wires together UV/LMN model spaces, beam evaluation, and NUDFT
operators to build the dense Finv/Fprime blocks used downstream.
"""
import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

from astropy import constants as c

from .dft_funcs import (
    inverse_udft_1d_rect,
    build_nudft_array,
    reorder_fuv_to_uvf,
    sparse_block_diag_repeat,
    sparse_block_diag_two,
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
        self.jd_start = jd_start
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
        # Only the first time-step uvw is used; time dependence is handled by beam rotation.
        uvw_array_m = uvw_array_m[0].T  # (3, n_bls)
        # Convert meters -> inverse radians by dividing by wavelength per frequency.
        wavelengths = c_ms / self.freq_hz
        self.inst_uvw_irad = uvw_array_m[jnp.newaxis, :, :] / wavelengths[:, jnp.newaxis, jnp.newaxis]

        self.bl_red_array = bl_red_array

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
            jd_center=self.jd_start,
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
            self.lmn_model_space.lmn_fg, self.freq_hz
            )  # Shape (n_freq, n_t, n_pix_fg)

        # Initialize sharding utilities.
        self.mm = MeshManager()
        self.sharding_grid = self.mm.get_sharding(rank=4, shard_uv=True)
        self.sharding_vec = self.mm.get_sharding(rank=1)
        self.sharding_mat = self.mm.get_sharding(rank=2)


    # ---------------------------
    # Construction of Finv/Fprime
    # ---------------------------
    def _uv_to_lm_eor(self):
        """
        Construct the EoR NUDFT kernel (model UV -> image LM/RaDec).
        """
        # 2D NUDFT for UV -> LM at the central time step (EoR pixel set).
        nuidft_uv_to_lm_eor = build_nudft_array(
            self.uv_model_space.uv_eor,  # (2, n_uv_eor)
            self.lmn_model_space.lmn_fg[
                self.lmn_model_space.time_ind_center, :2, self.lmn_model_space.eor_fg_shared_axis
            ],  # (2, n_pix_eor)
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
        nuidft_uv_to_lm_fg = build_nudft_array(
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
        nudft_lmn_to_uvw = build_nudft_array(
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


    @partial(jax.jit, static_argnums=(0,))
    def build_Finv_Fprime(self):
        """
        Build Finv/Fprime blocks across frequency and time.

        Uses sharded NUDFT kernels and nested scans to reduce memory usage.

        Returns
        -------
        finv_fprime : sparse.BCOO
            Sparse matrix with shape (n_vis, n_params), where
            n_vis = n_freq * n_t * n_bls and
            n_params = n_freq * (n_uv_eor + n_uv_fg).
        """
        # Precompute NUDFT kernels (UV -> LM) inside the JIT, then constrain sharding.
        nuidft_uv_to_lm_eor = jax.lax.with_sharding_constraint(self._uv_to_lm_eor(), self.sharding_mat)
        nuidft_uv_to_lm_fg  = jax.lax.with_sharding_constraint(self._uv_to_lm_fg(),  self.sharding_mat)


        # Nested scans: loop over frequency, then time, to build blocks.
        def frequency_loop(carry, i_f):
            def time_loop(t_carry, i_t):
                block_eor, block_fg = self._build_Finv_Fprime_block(
                    nuidft_uv_to_lm_eor, nuidft_uv_to_lm_fg, i_f, i_t
                )
                return t_carry, (block_eor, block_fg)
            _, (e_blocks, f_blocks) = lax.scan(time_loop, None, jnp.arange(self.n_t))
            return carry, (e_blocks, f_blocks)

        # Execute full scan across frequencies.
        _, (full_eor, full_fg) = lax.scan(frequency_loop, None, jnp.arange(self.n_freq))

        # Shapes: 
        # full_eor (n_freq, n_t, n_bls, n_uv_eor)
        # full_fg (n_freq, n_t, n_bls, n_uv_fg)

        # Constrain sharding of the dense blocks to avoid host/device reshuffles.
        full_eor = jax.lax.with_sharding_constraint(full_eor, self.sharding_grid)
        full_fg = jax.lax.with_sharding_constraint(full_fg, self.sharding_grid)

        # Build a final BCOO matrix by constructing global row/col indices.
        n_vis = self.n_freq * self.n_t * self.nbls
        n_params = self.n_freq * (self.n_uv_eor + self.n_uv_fg)

        f = jnp.arange(self.n_freq, dtype=jnp.int32)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        t = jnp.arange(self.n_t, dtype=jnp.int32)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        b = jnp.arange(self.nbls, dtype=jnp.int32)[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]

        # Shared base rows: (n_freq, n_t, n_bls, 1)
        # Match legacy ordering: time is the slowest axis, then frequency, then baseline.
        base_rows = ((t * self.n_freq + f) * self.nbls + b)

        # EoR block indices
        uv_eor = jnp.arange(self.n_uv_eor, dtype=jnp.int32)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        cols_eor = (f * self.n_uv_eor + uv_eor)
        data_eor = full_eor.reshape(-1)
        rows_eor_flat = jnp.broadcast_to(base_rows, full_eor.shape).reshape(-1)
        cols_eor_flat = jnp.broadcast_to(cols_eor, full_eor.shape).reshape(-1)
        indices_eor = jnp.stack([rows_eor_flat, cols_eor_flat], axis=1)

        # FG block indices (column offset after all EoR params)
        uv_fg = jnp.arange(self.n_uv_fg, dtype=jnp.int32)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        cols_fg = (self.n_freq * self.n_uv_eor) + (f * self.n_uv_fg + uv_fg)
        data_fg = full_fg.reshape(-1)
        rows_fg_flat = jnp.broadcast_to(base_rows, full_fg.shape).reshape(-1)
        cols_fg_flat = jnp.broadcast_to(cols_fg, full_fg.shape).reshape(-1)
        indices_fg = jnp.stack([rows_fg_flat, cols_fg_flat], axis=1)

        data = jnp.concatenate([data_eor, data_fg], axis=0)
        indices = jnp.concatenate([indices_eor, indices_fg], axis=0)

        # Constrain sharding of the BCOO payload as well.
        data = jax.lax.with_sharding_constraint(data, self.sharding_vec)
        indices = jax.lax.with_sharding_constraint(indices, self.sharding_mat)

        return sparse.BCOO((data, indices), shape=(n_vis, n_params), unique_indices=True)


    # ----------------------------
    # Construction of Fz (eta -> f)
    # ----------------------------
    def build_eta_to_freq_idft_eor(self):
        r"""
        Build the EoR IDFT block for eta -> frequency.

        Returns a block-diagonal matrix with one IDFT block per (u, v) pixel.
        Excludes eta=0 (belongs to the FG model).
        """
        idft_array_1d = inverse_udft_1d_rect(
            self.n_freq,
            self.n_eta,
            include_eta0=False
        )
        # Normalise by eta spacing.
        idft_array_1d *= self.d_eta

        multi_vis_idft_array_1d = sparse_block_diag_repeat(idft_array_1d, self.n_uv_eor)

        return multi_vis_idft_array_1d

    def build_eta_to_freq_idft_fg(self):
        r"""
        Build the FG IDFT block for eta -> frequency.

        Includes eta=0 terms, optional LSSM modes, and an optional monopole block.
        """

        # eta=0 column mapped into frequency space.
        eta0_to_frequency = jnp.ones((self.n_freq, 1), dtype=jnp.complex128) * self.d_eta * self.n_eta

        # One eta=0 block per FG (u, v), excluding monopole if fitted separately.
        neta0_blocks = self.n_uv_fg - int(self.fit_for_monopole)

        idft_array_1d_fg = sparse_block_diag_repeat(eta0_to_frequency, neta0_blocks)

        if self.n_modes_fg > 0:
            lssm_basis_vecs = build_lssm_basis(
                self.n_freq, nq=self.n_modes_fg, npl=self.n_powerlaws_fg, f_min=self.f0_mhz,
                df=self.df_mhz, beta=self.beta_fg
            )
            # Normalise LSSM basis.
            lssm_basis_vecs *= self.d_eta * self.n_eta

            # Repeat for each non-monopole FG (u, v).
            lssm_array = sparse_block_diag_repeat(lssm_basis_vecs, neta0_blocks)
            idft_array_1d_fg = sparse.bcoo_concatenate(
                [idft_array_1d_fg, lssm_array], dimension=1
            )
            
        if self.fit_for_monopole:
            idft_array_1d_mp = inverse_udft_1d_rect(
                self.n_freq, self.n_eta, nq=self.n_modes_fg, npl=self.n_powerlaws_fg,
                f_min=self.f0_mhz, df=self.df_mhz, beta=self.beta_fg
            )
            # Monopole block includes eta=0 and all LSSM modes.
            idft_array_1d_mp *= self.d_eta
            idft_array_1d_mp[:, self.n_eta//2] *= self.n_eta
            idft_array_1d_mp[:, self.n_eta:] *= self.n_eta
            idft_array_1d_fg = sparse_block_diag_two(idft_array_1d_fg, idft_array_1d_mp)

        return idft_array_1d_fg

    def reorder_fuv_to_uvf_eor(self):
        r"""
        Build the vis-ordered -> chan-ordered gridding matrix for EoR.

        This maps UV-major ordering to frequency-major ordering.
        """
        gridding_matrix = reorder_fuv_to_uvf(self.n_uv_eor, self.n_freq)

        return gridding_matrix

    
    def reorder_fuv_to_uvf_fg(self):
        r"""
        Build the vis-ordered -> chan-ordered gridding matrix for FG.

        This maps UV-major ordering to frequency-major ordering.
        """

        gridding_matrix = reorder_fuv_to_uvf(self.n_uv_fg, self.n_freq)

        return gridding_matrix      

    def build_Fz(self):
        r"""
        Build the full Fz operator (EoR + FG).

        Fz is a block-diagonal operator that maps eta-space vectors
        (vis-ordered) to frequency-space vectors (chan-ordered). The
        returned operator is stored as a sparse BCOO matrix.
        """
        # Build EoR/FG IDFT blocks and gridding matrices directly, then
        # map from vis-ordered eta space to chan-ordered frequency space.
        idft_eor = self.build_eta_to_freq_idft_eor()
        idft_fg = self.build_eta_to_freq_idft_fg()
        grid_eor = self.reorder_fuv_to_uvf_eor()
        grid_fg = self.reorder_fuv_to_uvf_fg()

        # Use sparse matmul in JAX.
        grid_eor = sparse.BCOO.fromdense(grid_eor)
        grid_fg = sparse.BCOO.fromdense(grid_fg)
        Fz_eor = sparse.bcoo_dot_general(
            grid_eor,
            idft_eor,
            dimension_numbers=(((1,), (0,)), ((), ()))
        )
        Fz_fg = sparse.bcoo_dot_general(
            grid_fg,
            idft_fg,
            dimension_numbers=(((1,), (0,)), ((), ()))
        )
        Fz = sparse_block_diag_two(Fz_eor, Fz_fg)

        return Fz

    def build_T(self):
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
        T : sparse.BCOO
            Sparse matrix with shape
            (ndata, (nu*nv - 1)*(neta - 1)
             + (nq + 1)*(nu_fg*nv_fg - 1)
             + fit_for_monopole*(neta + nq)).
        """
        Finv_Fprime = self.build_Finv_Fprime()
        Fz = self.build_Fz()

        # Sparse matmul in JAX: (n_vis, n_params) @ (n_params, n_params_eta)
        T = sparse.bcoo_dot_general(
            Finv_Fprime,
            Fz,
            dimension_numbers=(((1,), (0,)), ((), ()))
        )

        return T
    
