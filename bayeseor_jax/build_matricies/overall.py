"""
High-level orchestration for constructing model matrices and associated kernels.

This module combines the physical model spaces, beam response, and NUDFT
kernels into the dense operators used by inference.
"""
import time

import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P

from astropy.constants import c

from .dft_funcs import (
    inverse_udft_1d_rect,
    build_nudft_matrix,
)
from .lssm_model_space import build_lssm_basis
from .uv_model_space import UV_ModelSpace
from .lmn_model_space import LMN_ModelSpace
from .beam_model import Beam_Model
from .forward_model import (
    build_T_eor_core,
    build_T_fg_core,
    build_sky_to_visibility_core,
)
from .build_utils import (
    _dense_matrix_path as _dense_matrix_path_impl,
    _log_build,
    _prepare_memmap_output as _prepare_memmap_output_impl,
    _slice_bounds as _slice_bounds_impl,
    check_if_matrix_exists as check_if_matrix_exists_impl,
    load_matrix as load_matrix_impl,
    padded_visibility_partition as padded_visibility_partition_impl,
    save_matrix as save_matrix_impl,
)
from .jax_grid import MeshManager


class BuildMatrices:
    """Build the physical 21-cm and spherical foreground operators.

    The cosmological contribution is represented on a rectilinear
    ``(u, v, eta)`` grid, while smooth foreground brightness is represented in
    the spherical image domain. Both are propagated through the same primary
    beam and full three-dimensional fringe response. Large operators are built
    in process-local CPU row blocks and written to shared ``.npy`` memmaps.
    """

    def __init__(
        self,
        n_u_eor: int,
        du_eor: float,
        n_v_eor: int,
        dv_eor: float,
        fg_core_fov: Optional[float] = None,
        fg_outer_nside: Optional[int] = None,
        n_modes_fg: int = 0,
        n_powerlaws_fg: int = 0,
        beta_fg: Optional[Union[float, Sequence[float], jnp.ndarray]] = (2.63, 2.82),
        n_freq: int = 0,
        f0_mhz: Optional[float] = None,
        df_mhz: Optional[float] = None,
        n_eta: int = 0,
        d_eta: float = 0.0,
        image_domain_trim: bool = False,
        sky_mask_time_mode: str = "union",
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
        t_ninv_t_block_cols: int = 16384,
        t_ninv_t_right_block_batch: int = 4,
        clobber: bool = False,
    ) -> None:
        
        # Store EoR UV-plane parameters.
        self.n_u_eor = n_u_eor
        self.du_eor = du_eor
        self.n_v_eor = n_v_eor
        self.dv_eor = dv_eor
        # Total EoR uv modes (monopole always excluded).
        self.n_uv_eor = (n_u_eor * n_v_eor) - 1

        # Store foreground spectral-model settings.
        self.n_modes_fg = n_modes_fg
        self.fg_core_fov = fg_core_fov
        self.fg_outer_nside = fg_outer_nside
        self.n_powerlaws_fg = n_powerlaws_fg
        if beta_fg is None:
            self.beta_fg = [2.63, 2.82]
        elif np.isscalar(beta_fg):
            self.beta_fg = [float(beta_fg)]
        else:
            self.beta_fg = list(beta_fg)
        self.image_domain_trim = bool(image_domain_trim)
        self.sky_mask_time_mode = str(sky_mask_time_mode)

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
        self.t_ninv_t_block_cols = int(t_ninv_t_block_cols)
        self.t_ninv_t_right_block_batch = int(t_ninv_t_right_block_batch)
        self.clobber = bool(clobber)
        if self.t_ninv_t_block_cols < 1:
            raise ValueError("t_ninv_t_block_cols must be >= 1")
        if self.t_ninv_t_right_block_batch < 1:
            raise ValueError("t_ninv_t_right_block_batch must be >= 1")

        # The cosmological model retains its rectilinear transverse grid.
        self.uv_model_space = UV_ModelSpace(
            n_u_eor=self.n_u_eor,
            n_v_eor=self.n_v_eor,
            du_eor=self.du_eor,
            dv_eor=self.dv_eor,
        )

        # LMN model space: FG pixel set and shared EoR mask (computed at init).
        self.lmn_model_space = LMN_ModelSpace(
            fov_eor=self.fov_eor,
            fov_fg=self.fov_fg,
            fg_core_fov=self.fg_core_fov,
            fg_outer_nside=self.fg_outer_nside,
            nside=self.nside,
            sky_mask_time_mode=self.sky_mask_time_mode,
            telescope_latlonalt=self.telescope_latlonalt,
            jd_center=self.jd_center,
            n_t=self.n_t,
            d_t=self.d_t,
        )
        self.n_fg_pixels_trimmed = 0
        if self.image_domain_trim:
            self.n_fg_pixels_trimmed = int(
                self.lmn_model_space.trim_fg_outer_pixels_for_total_param_divisibility(
                    n_modes_fg=int(self.n_modes_fg),
                    n_uv_eor=int(self.n_uv_eor),
                    n_eta=int(self.n_eta),
                    divisor=4,
                )
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
        self.n_fg_spatial_modes = int(self.lmn_model_space.n_fg_groups)

    def _aggregate_fg_image_response(self, fg_response: jnp.ndarray) -> jnp.ndarray:
        """
        Aggregate the high-resolution FG pixel axis into multiresolution groups.

        Parameters
        ----------
        fg_response : jnp.ndarray
            Array whose final axis is the high-resolution FG pixel axis.

        Returns
        -------
        jnp.ndarray
            Array with the same leading axes and grouped FG spatial modes on
            the final axis.

        Notes
        -----
        If a grouped spatial mode ``g`` contains high-resolution child pixels
        ``p`` then the grouped image response is

        ``F_grouped[..., g] = sum_p F_hi[..., p]``.

        The corresponding image-domain FG coefficient therefore acts as a
        single shared brightness/amplitude for every child in that group. The
        grouped image-domain prior in ``PhiInverse`` retains the same base
        brightness precision for every group, treating neighbouring children
        as strongly correlated.
        """
        if not self.lmn_model_space.use_multires_fg:
            return fg_response
        labels = jnp.asarray(np.asarray(self.lmn_model_space.fg_group_labels))
        n_groups = int(self.lmn_model_space.n_fg_groups)
        grouped = jnp.zeros(
            fg_response.shape[:-1] + (n_groups,),
            dtype=fg_response.dtype,
        )
        # Summing the child columns means one grouped coefficient acts as a
        # shared brightness/amplitude for every high-resolution child pixel in
        # that group.
        return grouped.at[..., labels].add(fg_response)


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

    def _foreground_spectral_basis(self) -> jnp.ndarray:
        """Return the image-domain foreground basis ``Q_z``."""
        constant = jnp.ones((self.n_freq, 1), dtype=jnp.complex128)
        if self.n_modes_fg == 0:
            return constant
        smooth = build_lssm_basis(
            n_freq=self.n_freq,
            n_modes=self.n_modes_fg,
            n_powerlaws=self.n_powerlaws_fg,
            f0_mhz=self.f0_mhz,
            df_mhz=self.df_mhz,
            beta=self.beta_fg,
        )
        return jnp.concatenate([constant, smooth], axis=1)

    def _build_forward_operator_row_block(
        self,
        i_t: int,
        i_f: int,
        component: str,
        eor_transverse_basis: jnp.ndarray,
        eor_spectral_basis: jnp.ndarray,
        foreground_spectral_basis: jnp.ndarray,
    ) -> jnp.ndarray:
        """Build one ``(time, frequency)`` row block on the active device."""
        if component not in {"all", "eor", "foreground"}:
            raise ValueError(
                "component must be one of {'all', 'eor', 'foreground'}"
            )

        response = build_sky_to_visibility_core(
            np.asarray(self.lmn_model_space.lmn_fg[i_t : i_t + 1]),
            np.asarray(self.inst_uvw_irad[i_f : i_f + 1]),
            np.asarray(
                self.beam_response[i_f : i_f + 1, i_t : i_t + 1]
            ).transpose(1, 0, 2),
            self.lmn_model_space.pixel_area_sr,
        )

        components = []
        if component in {"all", "eor"}:
            eor_response = response[
                ..., self.lmn_model_space.eor_fg_shared_axis
            ]
            components.append(
                build_T_eor_core(
                    eor_response,
                    eor_transverse_basis,
                    eor_spectral_basis[i_f : i_f + 1],
                )
            )

        if component in {"all", "foreground"}:
            foreground_response = self._aggregate_fg_image_response(response)
            components.append(
                build_T_fg_core(
                    foreground_response,
                    foreground_spectral_basis[i_f : i_f + 1],
                )
            )

        return components[0] if len(components) == 1 else jnp.concatenate(
            components,
            axis=1,
        )

    def _forward_operator_width(self, component: str) -> int:
        """Return the number of columns in a selected forward operator."""
        eor_width = self.n_uv_eor * (self.n_eta - 1)
        foreground_width = self.n_fg_spatial_modes * (
            1 + self.n_modes_fg
        )
        if component == "eor":
            return int(eor_width)
        if component == "foreground":
            return int(foreground_width)
        if component == "all":
            return int(eor_width + foreground_width)
        raise ValueError(
            "component must be one of {'all', 'eor', 'foreground'}"
        )

    def _build_forward_operator(
        self,
        *,
        component: str,
        matrix_name: str,
    ) -> Path:
        """Build a forward operator in distributed CPU row blocks.

        JAX processes receive contiguous time ranges according to
        ``jax.process_index()`` and ``jax.process_count()``. Each process builds
        one frequency block at a time on its local CPU and writes only its
        disjoint rows into a shared ``.npy`` memmap.
        """
        path = _dense_matrix_path_impl(self, matrix_name)
        partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        n_rows = int(self.n_t * self.n_freq * self.nbls)
        n_cols = self._forward_operator_width(component)
        output = _prepare_memmap_output_impl(
            self,
            partial_path,
            shape=(n_rows, n_cols),
            dtype=np.dtype(np.complex128),
            sync_name=f"{matrix_name}-forward-operator-ready",
        )

        cpu_devices = jax.local_devices(backend="cpu")
        if not cpu_devices:
            raise RuntimeError("No local JAX CPU device is available.")
        cpu_device = cpu_devices[0]
        with jax.default_device(cpu_device):
            if component in {"all", "eor"}:
                eor_transverse_basis = jax.device_put(
                    np.asarray(self._uv_to_lm_eor()),
                    cpu_device,
                )
                eor_spectral_basis = jax.device_put(
                    np.asarray(
                        inverse_udft_1d_rect(
                            self.n_freq,
                            self.n_eta,
                            include_eta0=False,
                        )
                        * self.d_eta
                    ),
                    cpu_device,
                )
            else:
                eor_transverse_basis = jnp.zeros((0, 0), dtype=jnp.complex128)
                eor_spectral_basis = jnp.zeros((0, 0), dtype=jnp.complex128)
            foreground_spectral_basis = (
                jax.device_put(
                    np.asarray(self._foreground_spectral_basis()),
                    cpu_device,
                )
                if component in {"all", "foreground"}
                else jnp.zeros((0, 0), dtype=jnp.complex128)
            )

            process_index = int(jax.process_index())
            process_count = int(jax.process_count())
            base_times, extra_times = divmod(int(self.n_t), process_count)
            local_n_times = base_times + int(process_index < extra_times)
            time_start = process_index * base_times + min(
                process_index,
                extra_times,
            )
            time_stop = time_start + local_n_times

            _log_build(
                self,
                f"{matrix_name}: process {process_index}/{process_count} "
                f"building times [{time_start}, {time_stop}) on CPU",
            )
            for i_t in range(time_start, time_stop):
                for i_f in range(self.n_freq):
                    block = self._build_forward_operator_row_block(
                        i_t,
                        i_f,
                        component,
                        eor_transverse_basis,
                        eor_spectral_basis,
                        foreground_spectral_basis,
                    ).block_until_ready()
                    row_start = (i_t * self.n_freq + i_f) * self.nbls
                    output[row_start : row_start + self.nbls] = np.asarray(
                        block
                    )
            output.flush()

        del output
        distributed = (
            getattr(jax, "distributed", None) is not None
            and jax.distributed.is_initialized()
        )
        if distributed:
            multihost_utils.sync_global_devices(
                f"{matrix_name}-forward-operator-done"
            )
        if jax.process_index() == 0:
            partial_path.replace(path)
            partial_path.with_suffix(
                partial_path.suffix + ".ready"
            ).unlink(missing_ok=True)
        if distributed:
            multihost_utils.sync_global_devices(
                f"{matrix_name}-forward-operator-published"
            )
        return path

    def build_T_eor(self, *, matrix_name: str = "T_eor") -> Path:
        """Build and save :math:`T_{21}=F^{-1}P F'F_z`."""
        return self._build_forward_operator(
            component="eor",
            matrix_name=matrix_name,
        )

    def build_T_fg(self, *, matrix_name: str = "T_fg") -> Path:
        """Build and save :math:`T_{\rm FG}=F^{-1}P Q_z`."""
        return self._build_forward_operator(
            component="foreground",
            matrix_name=matrix_name,
        )

    def build_T(self, *, matrix_name: str = "T") -> Path:
        """Build and save the concatenated 21-cm and foreground operators."""
        return self._build_forward_operator(
            component="all",
            matrix_name=matrix_name,
        )


    # ------------------------
    # Direct construction of T 
    # ------------------------

    def _build_T_directly_impl(self) -> jnp.ndarray:
        r"""
        Construct ``T = (F^{-1}F')F_z`` directly, without explicitly building ``F_z``.

        Intuition
        ---------
        ``F^{-1}F'`` gives, for each observing frequency ``f``, a dense spatial
        response block over ``(t, bl, spatial_mode)``.

        The EoR block always uses UV-space spatial modes:

        - ``full_eor[f, t, bl, u_eor]``

        The non-monopole FG block can use either:

        - UV-space modes ``full_fg[f, t, bl, u_fg]``, or
        - image-space modes ``finv_blocks[f, t, bl, pix_fg]``

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
        response = build_sky_to_visibility_core(
            self.lmn_model_space.lmn_fg,
            self.inst_uvw_irad,
            jnp.transpose(self.beam_response, (1, 0, 2)),
            self.lmn_model_space.pixel_area_sr,
        )
        t_eor = build_T_eor_core(
            response[..., self.lmn_model_space.eor_fg_shared_axis],
            self._uv_to_lm_eor(),
            inverse_udft_1d_rect(
                self.n_freq,
                self.n_eta,
                include_eta0=False,
            )
            * self.d_eta,
        )
        t_fg = build_T_fg_core(
            self._aggregate_fg_image_response(response),
            self._foreground_spectral_basis(),
        )
        return jnp.concatenate([t_eor, t_fg], axis=1)

    def build_T_directly(self) -> jnp.ndarray:
        """
        Build T using an automatically selected JAX executor.

        - single-device runs: use a ``jax.jit`` executor
        - multi-device runs:  use a ``jax.jit`` executor with output sharding
          and run in mesh context
        """
        multi_device = self.mm.n_dev > 1
        jit_kwargs = {"static_argnums": (0,)}
        if multi_device:
            jit_kwargs["out_shardings"] = self.sharding_mat
        build_t_fn = jax.jit(BuildMatrices._build_T_directly_impl, **jit_kwargs)

        if multi_device:
            with self.mm.mesh:
                return build_t_fn(self)
        return build_t_fn(self)
    
    # -----------------------------------------------
    # Direct construction of T (stepwise UV chunking)
    # -----------------------------------------------

    def save_T_stepwise_row_memmap(
        self,
        *,
        matrix_name: str = "T",
    ) -> None:
        """Compatibility wrapper for the distributed forward-operator builder."""
        self.build_T(matrix_name=matrix_name)
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
        Build the inverse noise covariance diagonal (vector form).

        Returns
        -------
        Ninv : jnp.ndarray
            1D vector of shape (n_vis,) containing diag(N^{-1}), where
            n_vis = n_freq * n_t * n_bls.
        """
        sigma_sq = self._sigma_squared_array()
        return 1.0 / sigma_sq

    def build_N(self) -> jnp.ndarray:
        """
        Build the noise covariance diagonal (vector form).

        Returns
        -------
        N : jnp.ndarray
            1D vector of shape (n_vis,) containing diag(N), where
            n_vis = n_freq * n_t * n_bls.
        """
        sigma_sq = self._sigma_squared_array()
        return sigma_sq

    # -----------------------------
    # Build T^H N^{-1} T (directly)
    # -----------------------------
    
    def build_T_Ninv_T(self, T, Ninv: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build T^H N^{-1} T.

        Parameters
        ----------
        T : jnp.ndarray
            Forward operator with shape (n_vis, n_params_eta).
        Ninv : jnp.ndarray
            Inverse noise covariance diagonal in vector form with
            shape (n_vis,), holding diag(N^{-1}).

        Returns
        -------
        T_Ninv_T : jnp.ndarray
            Dense matrix with shape (n_params_eta, n_params_eta), where 
            n_params_eta = n_uv_eor * (n_eta - 1) + n_eta0 + n_lssm + n_mp.
        """
        
        Ninv_T = Ninv[:, None] * T

        # T^H @ (N^{-1} T) in dense space for robust complex-conjugate handling.
        return jnp.conj(T).T @ Ninv_T
    
    # --------------------------------------------------
    # Build T^H N^{-1} T (blockwise, cached T and Ninv)
    # --------------------------------------------------

    @staticmethod
    @partial(
        pjit,
        in_shardings=(P(("host", "dev"), None), P(("host", "dev")), None, None),
        out_shardings=P(("host", "dev"), None),
        static_argnums=(4, 5),
    )
    def _build_T_Ninv_T_cached_block_batch(
        T: jnp.ndarray,
        Ninv: jnp.ndarray,
        i0: int,
        j0: int,
        left_width: int,
        right_width: int,
    ) -> jnp.ndarray:
        r"""
        Build one block batch of ``T^H N^{-1} T`` from cached sharded inputs.

        Parameters
        ----------
        T : jnp.ndarray
            Globally sharded design matrix with shape ``(n_vis, n_params)``.
        Ninv : jnp.ndarray
            Globally sharded inverse-noise vector with shape ``(n_vis,)``.
        i0, j0 : int
            Global starting parameter indices for the left and right blocks.
        left_width, right_width : int
            Widths of the left and right parameter blocks.

        Returns
        -------
        jnp.ndarray
            Dense block batch with shape ``(left_width, right_width)`` and row
            sharding over the combined host/device mesh.

        """
        n_vis = T.shape[0]
        # Slice the cached global `T` into the current left/right parameter
        # blocks. The visibility axis stays fully distributed; only the
        # parameter columns change from call to call.
        T_left = jax.lax.dynamic_slice(T, (0, i0), (n_vis, left_width))
        T_right = jax.lax.dynamic_slice(T, (0, j0), (n_vis, right_width))
        # Each device contributes its local visibility rows to the same
        # Gram-block contraction, and JAX combines those partial products
        # according to the output sharding.
        return jnp.conj(T_left).T @ (Ninv[:, None] * T_right)


    def save_T_Ninv_T_blockwise_memmap(
        self,
        *,
        matrix_name: str = "T_Ninv_T",
        t_matrix_name: str = "T",
        ninv_matrix_name: str = "Ninv",
        block_cols: Optional[int] = None,
        right_block_batch: Optional[int] = None,
    ) -> None:
        r"""
        Build and save ``T^H N^{-1} T`` from saved ``T`` and ``Ninv`` arrays.

        Parameters
        ----------
        matrix_name : str, optional
            Output dense matrix name. Defaults to ``"T_Ninv_T"``.
        t_matrix_name : str, optional
            Saved design-matrix name to read from disk. Defaults to ``"T"``.
        ninv_matrix_name : str, optional
            Saved inverse-noise vector name to read from disk. Defaults to
            ``"Ninv"``.
        block_cols : int, optional
            Parameter-space block width. If omitted, the instance default
            ``t_ninv_t_block_cols`` is used.
        right_block_batch : int, optional
            Number of right-hand parameter blocks evaluated together per local
            kernel launch. If omitted, the instance default
            ``t_ninv_t_right_block_batch`` is used.
        """
        t_path = _dense_matrix_path_impl(self, t_matrix_name)
        ninv_path = _dense_matrix_path_impl(self, ninv_matrix_name)
        out_path = _dense_matrix_path_impl(self, matrix_name)
        if not t_path.exists():
            raise FileNotFoundError(f"Saved matrix not found: {t_path}")
        if not ninv_path.exists():
            raise FileNotFoundError(f"Saved matrix not found: {ninv_path}")

        block_cols = max(
            1,
            int(self.t_ninv_t_block_cols if block_cols is None else block_cols),
        )
        right_block_batch = max(
            1,
            int(self.t_ninv_t_right_block_batch if right_block_batch is None else right_block_batch),
        )

        T_mm = np.load(t_path, mmap_mode="r")
        Ninv_mm = np.load(ninv_path, mmap_mode="r")
        n_vis, n_params = (int(T_mm.shape[0]), int(T_mm.shape[1]))
        if tuple(Ninv_mm.shape) != (n_vis,):
            raise ValueError(
                f"{ninv_matrix_name} has shape {tuple(Ninv_mm.shape)} but expected {(n_vis,)}"
            )

        process_count = int(jax.process_count())
        local_device_count = int(jax.local_device_count())
        process_index = int(jax.process_index())
        row0, row1, rows_per_process = padded_visibility_partition_impl(
            n_vis=n_vis,
            process_count=process_count,
            local_device_count=local_device_count,
            process_index=process_index,
        )

        # Pre-create the dense on-disk output once so every process can attach
        # to the same memmap and write only the blocks it owns.
        mm = _prepare_memmap_output_impl(
            self,
            out_path,
            shape=(n_params, n_params),
            dtype=np.dtype(np.complex128),
            sync_name=f"{matrix_name}-blockwise-ready",
        )

        devices = np.asarray(jax.devices(), dtype=object)
        if devices.size != process_count * local_device_count:
            raise ValueError(
                "jax.devices() size must equal process_count * local_device_count for cached T_Ninv_T"
            )
        row_shards = int(devices.size)
        n_params_padded = int(((n_params + row_shards - 1) // row_shards) * row_shards)
        pad_cols = int(n_params_padded - n_params)

        # Cache the host-local visibility stripe once. When the parameter count
        # is not divisible by the global row-sharding factor we pad only these
        # in-memory copies with trailing zero columns. The output memmap keeps
        # the true `(n_params, n_params)` shape and every block write is clipped
        # back to the valid physical rows/columns below.
        local_T = np.asarray(T_mm[row0:row1, :])
        pad_rows = int(rows_per_process - local_T.shape[0])
        if pad_rows:
            local_T = np.pad(local_T, ((0, pad_rows), (0, 0)), mode="constant")
        if pad_cols:
            local_T = np.pad(local_T, ((0, 0), (0, pad_cols)), mode="constant")
        local_Ninv = np.asarray(Ninv_mm[row0:row1])
        if pad_rows:
            local_Ninv = np.pad(local_Ninv, (0, pad_rows), mode="constant")

        mesh = Mesh(devices.reshape(process_count, local_device_count), ("host", "dev"))

        total_param_blocks = int((n_params + block_cols - 1) // block_cols)
        with mesh:
            # Convert each process-local host stripe into one global sharded
            # array. After this point, the block kernel can reuse the same
            # cached distributed `T` and `Ninv` for every output block.
            T = multihost_utils.host_local_array_to_global_array(
                local_T, mesh, P(("host", "dev"), None)
            )
            Ninv = multihost_utils.host_local_array_to_global_array(
                local_Ninv, mesh, P(("host", "dev"))
            )
            T.block_until_ready()
            Ninv.block_until_ready()

            for block_i in range(total_param_blocks):
                # Each outer-loop step builds one row of parameter-space blocks
                # in the normal matrix. That row is then flushed to disk before
                # advancing so long runs always leave durable progress behind.
                i0 = int(block_i * block_cols)
                i1 = int(min(i0 + block_cols, n_params))
                valid_left_width = i1 - i0
                left_width = int(
                    min(
                        ((valid_left_width + row_shards - 1) // row_shards) * row_shards,
                        n_params_padded - i0,
                    )
                )

                if process_index == 0:
                    _log_build(
                        self,
                        f"cached T_Ninv_T processing parameter block row {block_i + 1}/{total_param_blocks}",
                    )

                # All devices cooperate on the same block row. Reusing the
                # cached sharded `T` avoids rereading the 100 GB design matrix
                # inside the hot loop; only the finished block batch is copied
                # back to host and written to disk.
                for block_j_start in range(block_i, total_param_blocks, right_block_batch):
                    # Evaluate several adjacent right-hand parameter blocks in
                    # one kernel call so the GPUs see a larger GEMM and spend
                    # less time on launch overhead.
                    j0 = int(block_j_start * block_cols)
                    j1 = int(min(j0 + right_block_batch * block_cols, n_params))
                    valid_right_width = j1 - j0
                    right_width = valid_right_width

                    gram_batch = self._build_T_Ninv_T_cached_block_batch(
                        T,
                        Ninv,
                        i0,
                        j0,
                        left_width,
                        right_width,
                    )
                    gram_batch.block_until_ready()

                    for shard in gram_batch.addressable_shards:
                        # Each process only receives the output shards that are
                        # addressable from its local devices. We materialize
                        # those small shards on host and write them into the
                        # correct dense-matrix slices.
                        row_slice, col_slice = shard.index
                        block_host = np.asarray(shard.data)
                        row_start, row_stop = _slice_bounds_impl(row_slice, left_width)
                        col_start, col_stop = _slice_bounds_impl(col_slice, right_width)
                        row_stop_valid = min(row_stop, valid_left_width)
                        col_stop_valid = min(col_stop, valid_right_width)
                        if row_stop_valid <= row_start or col_stop_valid <= col_start:
                            continue

                        keep_rows = row_stop_valid - row_start
                        keep_cols = col_stop_valid - col_start
                        block_host = block_host[:keep_rows, :keep_cols]

                        rr0 = i0 + row_start
                        rr1 = i0 + row_stop_valid
                        cc0 = j0 + col_start
                        cc1 = j0 + col_stop_valid

                        mm[rr0:rr1, cc0:cc1] = block_host
                        # The normal matrix is Hermitian, so every strict
                        # upper-triangle block determines its mirrored partner.
                        # Mirroring here avoids a separate post-processing pass.
                        if j0 != i0 or valid_right_width != valid_left_width:
                            mm[cc0:cc1, rr0:rr1] = block_host.conj().T

                # Flush after each completed parameter-block row so the dense
                # memmap always reflects the durable progress made so far.
                mm.flush()

            if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
                multihost_utils.sync_global_devices(f"{matrix_name}-blockwise-done")


    # --------------------------------------------
    # Build T^H N^{-1} d and d^H N^{-1} (directly)
    # --------------------------------------------

    def build_dbar(
        self,
        T: jnp.ndarray,
        Ninv: jnp.ndarray,
        vis_noisy: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""
        Build ``dbar = T^H N^{-1} d`` directly from dense inputs.

        Parameters
        ----------
        T : jnp.ndarray
            Forward operator with shape ``(n_vis, n_params_eta)``.
        Ninv : jnp.ndarray
            Inverse noise covariance diagonal in vector form with shape
            ``(n_vis,)``, holding ``diag(N^{-1})``.
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.

        Returns
        -------
        jnp.ndarray
            Compressed data vector with shape ``(n_params_eta,)``.
        """
        ninv_d = Ninv * vis_noisy
        # This is the same dense contraction that later appears inside the
        # marginalized quadratic term, just compressed once up front.
        return jnp.conj(T).T @ ninv_d

    def build_d_ninv_d(
        self,
        Ninv: jnp.ndarray,
        vis_noisy: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""
        Build the scalar data term ``d^H N^{-1} d`` directly.

        Parameters
        ----------
        Ninv : jnp.ndarray
            Inverse noise covariance diagonal in vector form with shape
            ``(n_vis,)``.
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.

        Returns
        -------
        jnp.ndarray
            Real scalar ``d^H N^{-1} d``.
        """
        ninv_d = Ninv * vis_noisy
        return jnp.real(jnp.vdot(vis_noisy, ninv_d))
    
    # ------------------------------------------------------
    # Build T^H N^{-1} d (cached T and Ninv, blockwise dbar)
    # ------------------------------------------------------
    

    def save_dbar_from_saved_T(
        self,
        vis_noisy: jnp.ndarray,
        *,
        matrix_name: str = "dbar",
        t_matrix_name: str = "T",
        ninv_matrix_name: str = "Ninv",
    ) -> None:
        r"""
        Build and save ``dbar = T^H N^{-1} d`` from saved ``T`` and ``Ninv``.

        Parameters
        ----------
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.
        matrix_name : str, optional
            Output vector name. Defaults to ``"dbar"``.
        t_matrix_name : str, optional
            Saved design-matrix name to read from disk. Defaults to ``"T"``.
        ninv_matrix_name : str, optional
            Saved inverse-noise vector name to read from disk. Defaults to
            ``"Ninv"``.

        Notes
        -----
        This follows the same cached disk-first pattern as the saved
        ``T_Ninv_T`` path, but the contraction is simpler: once the globally
        sharded ``T`` and row-sharded ``N^{-1} d`` are materialized, one dense
        distributed matvec produces the full compressed data vector.
        """
        t_path = _dense_matrix_path_impl(self, t_matrix_name)
        ninv_path = _dense_matrix_path_impl(self, ninv_matrix_name)
        out_path = _dense_matrix_path_impl(self, matrix_name)
        if not t_path.exists():
            raise FileNotFoundError(f"Saved matrix not found: {t_path}")
        if not ninv_path.exists():
            raise FileNotFoundError(f"Saved matrix not found: {ninv_path}")

        T_mm = np.load(t_path, mmap_mode="r")
        Ninv_mm = np.load(ninv_path, mmap_mode="r")
        n_vis, n_params = (int(T_mm.shape[0]), int(T_mm.shape[1]))
        if tuple(Ninv_mm.shape) != (n_vis,):
            raise ValueError(
                f"{ninv_matrix_name} has shape {tuple(Ninv_mm.shape)} but expected {(n_vis,)}"
            )

        vis_noisy_np = np.asarray(vis_noisy, dtype=np.complex128).reshape(-1)
        if vis_noisy_np.shape != (n_vis,):
            raise ValueError(
                f"vis_noisy has shape {vis_noisy_np.shape} but expected {(n_vis,)}"
            )

        process_count = int(jax.process_count())
        process_index = int(jax.process_index())

        if int(jax.device_count()) > 1:
            local_device_count = int(jax.local_device_count())
            row0, row1, rows_per_process = padded_visibility_partition_impl(
                n_vis=n_vis,
                process_count=process_count,
                local_device_count=local_device_count,
                process_index=process_index,
            )
            local_ninv_d = np.asarray(Ninv_mm[row0:row1]) * vis_noisy_np[row0:row1]
            pad_rows = int(rows_per_process - local_ninv_d.shape[0])
            if pad_rows:
                local_ninv_d = np.pad(local_ninv_d, (0, pad_rows), mode="constant")

            devices = np.asarray(jax.devices(), dtype=object)
            if devices.size != process_count * local_device_count:
                raise ValueError(
                    "jax.devices() size must equal process_count * local_device_count for cached dbar"
                )

            mesh = Mesh(devices.reshape(process_count, local_device_count), ("host", "dev"))

            @partial(
                pjit,
                in_shardings=(P(("host", "dev"), None), P(("host", "dev"))),
                out_shardings=P(None),
            )
            def _build_dbar_cached(
                T: jnp.ndarray, ninv_d: jnp.ndarray
            ) -> jnp.ndarray:
                return jnp.conj(T).T @ ninv_d

            local_T = np.asarray(T_mm[row0:row1, :], dtype=np.complex128)
            if pad_rows:
                local_T = np.pad(local_T, ((0, pad_rows), (0, 0)), mode="constant")
            with mesh:
                T = multihost_utils.host_local_array_to_global_array(
                    local_T, mesh, P(("host", "dev"), None)
                )
                ninv_d = multihost_utils.host_local_array_to_global_array(
                    local_ninv_d, mesh, P(("host", "dev"))
                )
                dbar = _build_dbar_cached(T, ninv_d)
                dbar.block_until_ready()
        else:
            T = jnp.asarray(np.asarray(T_mm, dtype=np.complex128))
            ninv_d = jnp.asarray(np.asarray(Ninv_mm) * vis_noisy_np, dtype=jnp.complex128)
            dbar = jnp.conj(T).T @ ninv_d

        if jax.process_index() == 0:
            np.save(out_path, np.asarray(dbar))

        if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
            multihost_utils.sync_global_devices(f"{matrix_name}-save-done")

    def save_d_ninv_d(
        self,
        vis_noisy: jnp.ndarray,
        *,
        matrix_name: str = "d_ninv_d",
        ninv_matrix_name: str = "Ninv",
    ) -> None:
        r"""
        Build and save ``d^H N^{-1} d`` from saved ``Ninv`` and ``d``.

        Parameters
        ----------
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.
        matrix_name : str, optional
            Output scalar name. Defaults to ``"d_ninv_d"``.
        ninv_matrix_name : str, optional
            Saved inverse-noise vector name to read from disk. Defaults to
            ``"Ninv"``.

        Notes
        -----
        This contraction is only a visibility-space dot product, so unlike
        ``T`` and ``T_Ninv_T`` it does not need a separate blockwise builder.
        """
        out_path = _dense_matrix_path_impl(self, matrix_name)
        Ninv = load_matrix_impl(self, ninv_matrix_name)
        vis_noisy = jnp.asarray(vis_noisy, dtype=jnp.complex128).reshape(-1)
        d_ninv_d = self.build_d_ninv_d(Ninv, vis_noisy)

        if jax.process_index() == 0:
            np.save(out_path, np.asarray(d_ninv_d))

        if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
            multihost_utils.sync_global_devices(f"{matrix_name}-save-done")

    # --------------------
    # Matrix save/load API
    # --------------------

    def _invalidate_saved_outputs(self, matrix_names: Sequence[str]) -> None:
        """
        Delete saved outputs so subsequent build calls recompute them.

        Parameters
        ----------
        matrix_names : sequence of str
            Logical saved-array names, e.g. ``["T", "T_Ninv_T"]``.
        """
        if not matrix_names:
            return

        removed = False
        if jax.process_index() == 0:
            for matrix_name in matrix_names:
                path = _dense_matrix_path_impl(self, matrix_name)
                if path.exists():
                    path.unlink()
                    removed = True

        if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
            multihost_utils.sync_global_devices("build-matrix-clobber")

        if removed:
            _log_build(self, f"clobbered saved outputs: {', '.join(matrix_names)}")

    def build_matrix_if_not_here(
        self,
        matrix_name: str,
        *,
        invalidate_existing: bool | None = None,
    ) -> None:
        r"""
        Build and save a matrix if it does not already exist on disk.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix to build and save.
        invalidate_existing : bool, optional
            When ``True``, remove any existing saved products invalidated by
            ``matrix_name`` before checking whether a rebuild is required.
            Defaults to ``self.clobber``.
        """
        if invalidate_existing is None:
            invalidate_existing = self.clobber

        if invalidate_existing:
            invalidation_map = {
                "T": ("T", "T_Ninv_T", "dbar"),
                "Ninv": ("Ninv", "T_Ninv_T", "dbar", "d_ninv_d"),
                "T_Ninv_T": ("T_Ninv_T",),
                "dbar": ("dbar",),
                "d_ninv_d": ("d_ninv_d",),
            }
            self._invalidate_saved_outputs(invalidation_map.get(matrix_name, (matrix_name,)))

        if check_if_matrix_exists_impl(self, matrix_name):
            return
        _log_build(self, f"building and saving {matrix_name}")
        t_start = time.perf_counter()
        if matrix_name == "T":
            self.save_T_stepwise_row_memmap(matrix_name=matrix_name)
        elif matrix_name == "T_Ninv_T":
            self.build_matrix_if_not_here("T", invalidate_existing=False)
            self.build_matrix_if_not_here("Ninv", invalidate_existing=False)
            self.save_T_Ninv_T_blockwise_memmap(matrix_name=matrix_name)
        else:
            matrix = self._build_matrix_by_name(matrix_name)
            save_matrix_impl(self, matrix_name, matrix)
        elapsed = time.perf_counter() - t_start
        _log_build(self, f"built and saved {matrix_name} in {elapsed:.3f} s")

    def _build_matrix_by_name(self, matrix_name: str):
        """Build a supported matrix by name."""
        _log_build(self, f"building {matrix_name}")
        t_start = time.perf_counter()
        matrix = None
        # if matrix_name == "Fz":
        #     matrix = self.build_Fz()
        # elif matrix_name == "Finv_Fprime":
        #     matrix = self.build_Finv_Fprime()
        if matrix_name == "T":
            matrix = self.build_T_directly()
        elif matrix_name == "Ninv":
            matrix = self.build_Ninv()
        elif matrix_name == "N":
            matrix = self.build_N()
        elif matrix_name == "T_Ninv_T":
            T = load_matrix_impl(self, "T")
            Ninv = load_matrix_impl(self, "Ninv")
            try:
                matrix = self.build_T_Ninv_T(T, Ninv)
            finally:
                # Drop large intermediates as soon as possible.
                del T
                del Ninv
        else:
            raise ValueError(f"Unsupported matrix name '{matrix_name}'.")
        elapsed = time.perf_counter() - t_start
        _log_build(self, f"built {matrix_name} in {elapsed:.3f} s")
        return matrix

    def build_minimum_matrix_stack(self) -> None:
        """
        Build and save T, Ninv, then T_Ninv_T in that order.

        This uses a disk-first workflow to avoid keeping large matrices in
        memory longer than required.
        """
        if self.clobber:
            self._invalidate_saved_outputs(("T", "Ninv", "T_Ninv_T", "dbar", "d_ninv_d"))

        # Ensure T exists on disk.
        self.build_matrix_if_not_here("T", invalidate_existing=False)
        # Ensure Ninv exists on disk.
        self.build_matrix_if_not_here("Ninv", invalidate_existing=False)

        # Build and save T_Ninv_T through the blockwise disk-first path.
        self.build_matrix_if_not_here("T_Ninv_T", invalidate_existing=False)

    def build_minimum_likelihood_stack(self, vis_noisy: jnp.ndarray) -> int:
        """
        Build the minimum saved stack needed by the marginalized likelihood.

        This extends the data-independent matrix stack with the two
        data-dependent compressed terms:

        - ``dbar = T^H N^{-1} d``
        - ``d_ninv_d = d^H N^{-1} d``

        Parameters
        ----------
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.

        Returns
        -------
        int
            Number of model parameters, i.e. the length of ``dbar``.
        """
        self.build_minimum_matrix_stack()
        if self.clobber:
            self._invalidate_saved_outputs(("dbar", "d_ninv_d"))
        if not check_if_matrix_exists_impl(self, "dbar"):
            _log_build(self, "building and saving dbar")
            t_start = time.perf_counter()
            self.save_dbar_from_saved_T(vis_noisy, matrix_name="dbar")
            elapsed = time.perf_counter() - t_start
            _log_build(self, f"built and saved dbar in {elapsed:.3f} s")

        if not check_if_matrix_exists_impl(self, "d_ninv_d"):
            _log_build(self, "building and saving d_ninv_d")
            t_start = time.perf_counter()
            self.save_d_ninv_d(vis_noisy, matrix_name="d_ninv_d")
            elapsed = time.perf_counter() - t_start
            _log_build(self, f"built and saved d_ninv_d in {elapsed:.3f} s")

        t_matrix = np.load(_dense_matrix_path_impl(self, "T"), mmap_mode="r")
        return int(t_matrix.shape[1])
