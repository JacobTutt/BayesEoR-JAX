"""
Construct and save the mappings from sky coefficients to measured visibilities.

Image and Fourier geometries define the coefficient coordinates; beam and
forward operators propagate their brightness through the instrument. This
module schedules that construction in blocks and stores both the forward
matrix and its noise-weighted products for repeated likelihood evaluation.
"""
import time

import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jax.experimental import multihost_utils
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P

from astropy.constants import c

from ..cosmology.fourier_geometry import FourierGeometry, build_fourier_geometry
from ..observation.image_geometry import ImageGeometry, build_image_geometry
from ..instrument.beam import Beam_Model
from ..operators import (
    build_T_21_core,
    build_T_fg_core,
    build_foreground_spectral_operator,
    build_sky_to_visibility_operator,
    build_spectral_fourier_operator,
    build_transverse_fourier_operator,
)
from .storage import (
    _dense_matrix_path as _dense_matrix_path_impl,
    _log_build,
    _prepare_memmap_output as _prepare_memmap_output_impl,
    _slice_bounds as _slice_bounds_impl,
    check_if_matrix_exists as check_if_matrix_exists_impl,
    load_matrix as load_matrix_impl,
    save_matrix as save_matrix_impl,
)

from .distribution import padded_visibility_partition as padded_visibility_partition_impl


class BuildMatrices:
    r"""Coordinate construction of the cosmological and foreground visibility models.

    Parameters
    ----------
    n_u_eor : int
        Number of cosmological u modes before monopole exclusion.
    du_eor : float
        Cosmological u spacing in inverse radians.
    n_v_eor : int
        Number of cosmological v modes before monopole exclusion.
    dv_eor : float
        Cosmological v spacing in inverse radians.
    beta_fg : float or sequence of float
        Positive foreground spectral-index magnitudes; a constant component is
        included separately.
    n_freq : int
        Number of observing-frequency channels.
    f0_mhz : float
        First observing frequency and foreground reference frequency in MHz.
    df_mhz : float
        Separation of frequency channels in MHz.
    n_eta : int
        Number of line-of-sight Fourier modes before excluding eta=0.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).
    sky_mask_time_mode : str
        Pixel selection: 'union' across integrations or 'center' at the
        reference time.
    telescope_latlonalt : tuple of float
        Geodetic latitude and longitude in degrees, followed by altitude in
        metres.
    jd_center : float
        Reference integration time as a UTC Julian date.
    nside : int
        HEALPix resolution of the equal-area celestial pixel grid.
    fov_eor : float
        Full angular diameter of the cosmological model field in degrees.
    fov_fg : float or None
        Full angular diameter of the foreground model field in degrees; None
        uses fov_eor.
    n_t : int
        Number of observing-time samples.
    d_t : float
        Separation of integration centres in seconds.
    image_geometry : ImageGeometry or None
        Precomputed pixel directions and selections to reuse; None constructs
        them from the field and observing settings.
    fourier_geometry : FourierGeometry or None
        Precomputed (u, v, eta) coordinates and spacings to reuse; None
        constructs them from the grid settings.
    beam_type : str or None
        Analytic beam name or beamfits file path. None selects a uniform beam.
    peak_amp : float
        Peak power gain for Gaussian-based beam profiles; defaults to 1.
    fwhm_deg : float or None
        Configured beam full width at half maximum in degrees, when used instead
        of an aperture-derived width.
    diam : float or None
        Physical or effective aperture diameter in metres for Airy or Gaussian
        beam profiles.
    cosfreq : float or None
        Angular modulation frequency in inverse radians for the gausscosine
        profile.
    tanh_freq : float or None
        Angular taper rate in inverse radians for the tanhairy profile.
    tanh_sl_red : float or None
        Fractional far-off-axis power suppression for the tanhairy profile.
    pol : str
        Polarisation selected from a file-backed beam, for example xx or pI.
    freq_interp_kind : str
        Frequency interpolation method used by UVBeam; defaults to cubic.
    uvw_array_m : array-like
        Baseline coordinates in metres, shape (n_t, n_baseline, 3). The first
        integration supplies the fixed instrument-frame baselines.
    bl_red_array : array-like or None
        Redundancy per baseline and integration, shape (n_t, n_baseline) or
        (n_t, n_baseline, 1); used when effective_noise is absent.
    sigma : float or None
        Noise standard deviation before redundancy weighting, in the same units
        as the visibility data.
    effective_noise : array-like or None
        Per-visibility noise amplitudes in flattened T row order. Squared
        magnitudes override the sigma/redundancy variance model.
    array_save_directory : str or pathlib.Path
        Directory holding the cached forward matrix and likelihood products.
    t_ninv_t_block_cols : int
        Number of coefficient columns per block in the saved noise-weighted Gram
        matrix.
    t_ninv_t_right_block_batch : int
        Number of right-hand column blocks contracted together for the Gram
        matrix.
    clobber : bool
        Whether stack-building methods invalidate and rebuild existing saved
        products.

    Notes
    -----
    The cosmological contribution is represented on a rectilinear
    ``(u, v, eta)`` grid, while smooth foreground brightness is represented in
    the spherical image domain. Both are propagated through the same primary
    beam and full three-dimensional fringe response. Large operators are built
    in process-local CPU row blocks and written to shared ``.npy`` memmaps.

    ``build_T_21`` and ``build_T_fg`` save the separate coefficient-to-visibility
    mappings; ``build_T`` concatenates their columns in that order. The
    likelihood-stack methods then save ``T^H N^{-1} T``, ``T^H N^{-1} d``,
    and ``d^H N^{-1} d``. Geometry and beam values are prepared at construction;
    the large visibility matrices are created only when a build method runs.
    """

    def __init__(
        self,
        n_u_eor: int,
        du_eor: float,
        n_v_eor: int,
        dv_eor: float,
        beta_fg: Optional[Union[float, Sequence[float], jnp.ndarray]] = (2.63, 2.82),
        n_freq: int = 0,
        f0_mhz: Optional[float] = None,
        df_mhz: Optional[float] = None,
        n_eta: int = 0,
        d_eta: float = 0.0,
        sky_mask_time_mode: str = "union",
        telescope_latlonalt: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        jd_center: float = 2461040.5,
        nside: Optional[int] = None,
        fov_eor: Optional[float] = None,
        fov_fg: Optional[float] = None,
        n_t: int = 1,
        d_t: Optional[float] = None,
        image_geometry: Optional[ImageGeometry] = None,
        fourier_geometry: Optional[FourierGeometry] = None,
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
        r"""Prepare coordinates, Fourier factors, and beam weights for an observation.

        Parameters
        ----------
        n_u_eor : int
            Number of cosmological u modes before monopole exclusion.
        du_eor : float
            Cosmological u spacing in inverse radians.
        n_v_eor : int
            Number of cosmological v modes before monopole exclusion.
        dv_eor : float
            Cosmological v spacing in inverse radians.
        beta_fg : float or sequence of float
            Positive foreground spectral-index magnitudes; a constant component is
            included separately.
        n_freq : int
            Number of observing-frequency channels.
        f0_mhz : float
            First observing frequency and foreground reference frequency in MHz.
        df_mhz : float
            Separation of frequency channels in MHz.
        n_eta : int
            Number of line-of-sight Fourier modes before excluding eta=0.
        d_eta : float
            Positive eta spacing and integration weight in inverse Hz (seconds).
        sky_mask_time_mode : str
            Pixel selection: 'union' across integrations or 'center' at the
            reference time.
        telescope_latlonalt : tuple of float
            Geodetic latitude and longitude in degrees, followed by altitude in
            metres.
        jd_center : float
            Reference integration time as a UTC Julian date.
        nside : int
            HEALPix resolution of the equal-area celestial pixel grid.
        fov_eor : float
            Full angular diameter of the cosmological model field in degrees.
        fov_fg : float or None
            Full angular diameter of the foreground model field in degrees; None
            uses fov_eor.
        n_t : int
            Number of observing-time samples.
        d_t : float
            Separation of integration centres in seconds.
        image_geometry : ImageGeometry or None
            Precomputed pixel directions and selections to reuse; None constructs
            them from the field and observing settings.
        fourier_geometry : FourierGeometry or None
            Precomputed (u, v, eta) coordinates and spacings to reuse; None
            constructs them from the grid settings.
        beam_type : str or None
            Analytic beam name or beamfits file path. None selects a uniform beam.
        peak_amp : float
            Peak power gain for Gaussian-based beam profiles; defaults to 1.
        fwhm_deg : float or None
            Configured beam full width at half maximum in degrees, when used instead
            of an aperture-derived width.
        diam : float or None
            Physical or effective aperture diameter in metres for Airy or Gaussian
            beam profiles.
        cosfreq : float or None
            Angular modulation frequency in inverse radians for the gausscosine
            profile.
        tanh_freq : float or None
            Angular taper rate in inverse radians for the tanhairy profile.
        tanh_sl_red : float or None
            Fractional far-off-axis power suppression for the tanhairy profile.
        pol : str
            Polarisation selected from a file-backed beam, for example xx or pI.
        freq_interp_kind : str
            Frequency interpolation method used by UVBeam; defaults to cubic.
        uvw_array_m : array-like
            Baseline coordinates in metres, shape (n_t, n_baseline, 3). The first
            integration supplies the fixed instrument-frame baselines.
        bl_red_array : array-like or None
            Redundancy per baseline and integration, shape (n_t, n_baseline) or
            (n_t, n_baseline, 1); used when effective_noise is absent.
        sigma : float or None
            Noise standard deviation before redundancy weighting, in the same units
            as the visibility data.
        effective_noise : array-like or None
            Per-visibility noise amplitudes in flattened T row order. Squared
            magnitudes override the sigma/redundancy variance model.
        array_save_directory : str or pathlib.Path
            Directory holding the cached forward matrix and likelihood products.
        t_ninv_t_block_cols : int
            Number of coefficient columns per block in the saved noise-weighted Gram
            matrix.
        t_ninv_t_right_block_batch : int
            Number of right-hand column blocks contracted together for the Gram
            matrix.
        clobber : bool
            Whether stack-building methods invalidate and rebuild existing saved
            products.

        Returns
        -------
        None
            Initialises the instance without returning a value.

        Notes
        -----
        Grid sizes and spacings define the cosmological (u, v, eta) model;
        ``fov_eor`` and ``fov_fg`` select HEALPix fields with diameters in
        degrees. Supplied geometry objects are reused. Frequencies are
        configured in MHz, integration spacing in seconds, and baselines
        in metres. Baselines from the first integration define the fixed
        instrument frame; time dependence enters the sky directions and beam.

        ``beta_fg`` contains positive spectral-index magnitudes, converted
        to negative power-law exponents when constructing foreground spectra.
        The save directory, clobber setting, and column-block sizes control
        later persistence and distributed products; no full T is built here.
        """
        # Store foreground spectral-model settings.
        if beta_fg is None:
            self.beta_fg = [2.63, 2.82]
        elif np.isscalar(beta_fg):
            self.beta_fg = [float(beta_fg)]
        else:
            self.beta_fg = list(beta_fg)
        self.n_fg_components = 1 + len(self.beta_fg)
        self.sky_mask_time_mode = str(sky_mask_time_mode)

        # Store frequency/eta grid parameters.
        self.n_freq = n_freq
        self.f0_mhz = f0_mhz
        self.df_mhz = df_mhz    
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
        # Baselines are fixed; sky directions and beam weights carry time dependence.
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

        # Reuse setup geometry when available; direct callers may construct it here.
        self.image_geometry = image_geometry or build_image_geometry(
            fov_eor=self.fov_eor,
            fov_fg=self.fov_fg,
            nside=self.nside,
            sky_mask_time_mode=self.sky_mask_time_mode,
            telescope_latlonalt=self.telescope_latlonalt,
            jd_center=self.jd_center,
            n_t=self.n_t,
            d_t=self.d_t,
        )
        self.fourier_geometry = fourier_geometry or build_fourier_geometry(
            n_u=n_u_eor,
            n_v=n_v_eor,
            n_eta=n_eta,
            du=du_eor,
            dv=dv_eor,
            d_eta=d_eta,
        )
        lm_21 = self.image_geometry.lmn_fg[
            self.image_geometry.time_ind_center,
            :2,
        ]
        lm_21 = lm_21[:, self.image_geometry.eor_fg_shared_axis]
        self.transverse_fourier_operator = build_transverse_fourier_operator(
            self.fourier_geometry.uv,
            lm_21,
            uv_cell_area=self.fourier_geometry.du * self.fourier_geometry.dv,
        )
        frequency_offsets_hz = self.freq_hz - self.freq_hz[n_freq // 2]
        self.spectral_fourier_operator = build_spectral_fourier_operator(
            self.fourier_geometry.eta,
            frequency_offsets_hz,
            d_eta=self.fourier_geometry.d_eta,
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
            self.image_geometry.az_za_fg, self.freq_hz
            )  # Shape (n_freq, n_t, n_pix_fg)

        self.n_fg_spatial_modes = int(self.image_geometry.n_pix_fg)

    def _foreground_spectral_operator(self) -> jnp.ndarray:
        r"""Evaluate the foreground spectral factors at the selected frequencies.

        Returns
        -------
        Q : jax.Array
            Constant and power-law spectral factors, shape (n_freq,
            n_fg_components).

        Notes
        -----
        Uses the configured instance state; no explicit input arguments.

        Return shape ``(n_freq, n_fg_components)`` with a constant column
        followed by power laws referenced to ``f0_mhz``. The same factors
        convert the amplitude coefficients of every foreground pixel into
        its brightness spectrum.
        """
        return build_foreground_spectral_operator(
            self.freq_hz / 1e6,
            self.f0_mhz,
            spectral_indices=tuple(-beta for beta in self.beta_fg),
        )

    def _build_forward_operator_row_block(
        self,
        i_t: int,
        i_f: int,
        component: str,
        transverse_fourier_operator: jnp.ndarray,
        spectral_fourier_operator: jnp.ndarray,
        foreground_spectral_operator: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Propagate sky coefficients to all baselines at one time and frequency.

        Parameters
        ----------
        i_t : int
            Integration index selecting the pixel directions and beam weights.
        i_f : int
            Frequency-channel index selecting the fringe and spectral responses.
        component : str
            Component to construct: 'eor', 'foreground', or 'all' for the joint
            operator.
        transverse_fourier_operator : jax.Array
            F-prime matrix, shape (n_cosmological_pixel, n_uv_mode), mapping
            transverse modes to the selected image pixels.
        spectral_fourier_operator : jax.Array
            F_z matrix, shape (n_freq, n_eta_mode), mapping eta modes to the
            observing channels.
        foreground_spectral_operator : jax.Array
            Q spectral factors, shape (n_freq, n_fg_components), shared by all
            foreground pixels.

        Returns
        -------
        T_block : jax.Array
            Selected component response, shape (n_baseline, n_coefficient), at the
            specified time and frequency.

        Notes
        -----
        Evaluate the beam-weighted 3D fringe response for the foreground
        HEALPix pixels. For the cosmological component, select its smaller
        pixel set before applying the transverse and spectral Fourier factors.
        Return ``(n_baseline, n_coefficient)``; ``component='all'`` places
        cosmological columns before foreground columns. No file is written.
        """
        if component not in {"all", "eor", "foreground"}:
            raise ValueError(
                "component must be one of {'all', 'eor', 'foreground'}"
            )

        response = build_sky_to_visibility_operator(
            np.asarray(self.image_geometry.lmn_fg[i_t : i_t + 1]),
            np.asarray(self.inst_uvw_irad[i_f : i_f + 1]),
            np.asarray(
                self.beam_response[i_f : i_f + 1, i_t : i_t + 1]
            ).transpose(1, 0, 2),
            self.image_geometry.pixel_area_sr,
        )

        components = []
        if component in {"all", "eor"}:
            eor_response = response[
                ..., self.image_geometry.eor_fg_shared_axis
            ]
            components.append(
                build_T_21_core(
                    eor_response,
                    transverse_fourier_operator,
                    spectral_fourier_operator[i_f : i_f + 1],
                )
            )

        if component in {"all", "foreground"}:
            components.append(
                build_T_fg_core(
                    response,
                    foreground_spectral_operator[i_f : i_f + 1],
                )
            )

        return components[0] if len(components) == 1 else jnp.concatenate(
            components,
            axis=1,
        )

    def _forward_operator_width(self, component: str) -> int:
        r"""Count the coefficients mapped to visibilities by the requested component.

        Parameters
        ----------
        component : str
            Component to construct: 'eor', 'foreground', or 'all' for the joint
            operator.

        Returns
        -------
        n_coefficient : int
            Number of columns in the requested component or joint operator.

        Notes
        -----
        The cosmological count is ``n_uv_mode * n_eta_mode`` and the
        foreground count is ``n_pixel * n_spectral_component``. Their sum
        determines the column dimension of the joint operator on disk.
        """
        eor_width = (
            self.fourier_geometry.uv.shape[1]
            * self.fourier_geometry.eta.shape[0]
        )
        foreground_width = self.n_fg_spatial_modes * self.n_fg_components
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
        r"""Build and save a coefficient-to-visibility matrix one row block at a time.

        Parameters
        ----------
        component : str
            Component to construct: 'eor', 'foreground', or 'all' for the joint
            operator.
        matrix_name : str
            Output array name without the .npy suffix, relative to
            array_save_directory.

        Returns
        -------
        path : pathlib.Path
            Path to the completed .npy coefficient-to-visibility matrix.

        Notes
        -----
        JAX processes receive contiguous time ranges according to
        ``jax.process_index()`` and ``jax.process_count()``. Each process builds
        one frequency block at a time on its local CPU and writes only its
        disjoint rows into a shared ``.npy`` memmap.
        Rows follow time, frequency, baseline order. The temporary output is
        renamed to the requested matrix path after all processes finish;
        return that path rather than loading the completed matrix into memory.
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
                transverse_fourier_operator = jax.device_put(
                    np.asarray(self.transverse_fourier_operator),
                    cpu_device,
                )
                spectral_fourier_operator = jax.device_put(
                    np.asarray(self.spectral_fourier_operator),
                    cpu_device,
                )
            else:
                transverse_fourier_operator = jnp.zeros(
                    (0, 0), dtype=jnp.complex128
                )
                spectral_fourier_operator = jnp.zeros(
                    (0, 0), dtype=jnp.complex128
                )
            foreground_spectral_operator = (
                jax.device_put(
                    np.asarray(self._foreground_spectral_operator()),
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
                        transverse_fourier_operator,
                        spectral_fourier_operator,
                        foreground_spectral_operator,
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

    def build_T_21(self, *, matrix_name: str = "T_21") -> Path:
        r"""Save the mapping from cosmological Fourier amplitudes to visibilities.

        Parameters
        ----------
        matrix_name : str
            Output array name without the .npy suffix, relative to
            array_save_directory.

        Returns
        -------
        path : pathlib.Path
            Saved T_21 matrix, shape (n_t*n_freq*n_baseline, n_uv_mode*n_eta_mode).

        Notes
        -----
        Construct :math:`T_{21}=F^{-1}P F'F_z` in distributed CPU blocks.
        Return the saved ``.npy`` path. Columns follow (uv, eta) order, with
        eta varying fastest, and rows follow time, frequency, baseline order.
        """
        return self._build_forward_operator(
            component="eor",
            matrix_name=matrix_name,
        )

    def build_T_fg(self, *, matrix_name: str = "T_fg") -> Path:
        r"""Save the mapping from foreground HEALPix amplitudes to visibilities.

        Parameters
        ----------
        matrix_name : str
            Output array name without the .npy suffix, relative to
            array_save_directory.

        Returns
        -------
        path : pathlib.Path
            Saved T_fg matrix, shape (n_t*n_freq*n_baseline,
            n_fg_components*n_pix_fg).

        Notes
        -----
        Construct :math:`T_{\rm FG}=F^{-1}P Q_z` in distributed CPU blocks.
        Return the saved ``.npy`` path. Columns follow spectral component,
        then pixel order; rows follow time, frequency, baseline order.
        """
        return self._build_forward_operator(
            component="foreground",
            matrix_name=matrix_name,
        )

    def build_T(self, *, matrix_name: str = "T") -> Path:
        r"""Save the joint visibility mapping ``T = [T_21 T_fg]``.

        Parameters
        ----------
        matrix_name : str
            Output array name without the .npy suffix, relative to
            array_save_directory.

        Returns
        -------
        path : pathlib.Path
            Saved joint matrix with cosmological columns followed by foreground
            columns.

        Notes
        -----
        A coefficient vector containing cosmological Fourier amplitudes
        followed by foreground image amplitudes maps to model visibilities
        through ``m = T b``. Return the completed matrix's ``.npy`` path.
        """
        return self._build_forward_operator(
            component="all",
            matrix_name=matrix_name,
        )

    # ------------------------
    # Noise covariance matrices
    # ------------------------
    def _sigma_squared_array(self) -> jnp.ndarray:
        r"""Assign a noise variance to each measured complex visibility.

        Returns
        -------
        variance : jax.Array
            Per-visibility variances, shape (n_t*n_freq*n_baseline,), in squared
            visibility units.

        Notes
        -----
        Uses the configured instance state; no explicit input arguments.

        With an effective-noise array, return its squared magnitudes.
        Otherwise use ``sigma**2 / redundancy`` for each baseline and time,
        repeated over frequency. Return a flattened vector in the same
        time, frequency, baseline order as the rows of T.
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
        r"""Return the inverse-variance weights applied to visibility rows.

        Returns
        -------
        Ninv : jnp.ndarray
            1D vector of shape (n_vis,) containing diag(N^{-1}), where
            n_vis = n_freq * n_t * n_bls.

        Notes
        -----
        Uses the configured instance state; no explicit input arguments.

        Each weight is ``1 / variance``. The independent-noise assumption
        allows these diagonal entries to be stored as a vector and applied
        by elementwise multiplication, without allocating a dense covariance.
        """
        sigma_sq = self._sigma_squared_array()
        return 1.0 / sigma_sq

    # --------------------------------------------------
    # Build T^H N^{-1} T (blockwise, cached T and Ninv)
    # --------------------------------------------------

    @staticmethod
    @partial(
        jax.jit,
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
        r"""Contract visibility responses into one noise-weighted coefficient block.

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

        Notes
        -----
        Evaluate ``T_left^H @ (Ninv[:, None] * T_right)``: each entry sums
        the overlap between two coefficient responses over measured
        visibilities, weighted by their inverse noise variance.
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
        r"""Save the noise-weighted overlaps between all model-coefficient responses.

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

        Returns
        -------
        None
            The noise-weighted coefficient overlap matrix is saved to disk.

        Notes
        -----
        Read saved T and inverse-noise weights, contract their visibility
        rows in coefficient-column blocks, and write ``T^H N^{-1} T`` to
        disk. This data-independent matrix supplies the likelihood precision
        together with the prior precision. The completed output is published
        only after all blocks have been written.
        """
        t_path = _dense_matrix_path_impl(self, t_matrix_name)
        ninv_path = _dense_matrix_path_impl(self, ninv_matrix_name)
        out_path = _dense_matrix_path_impl(self, matrix_name)
        partial_path = out_path.with_name(
            f"{out_path.stem}.partial{out_path.suffix}"
        )
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
            partial_path,
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
        with jax.set_mesh(mesh):
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

        del mm
        distributed = (
            getattr(jax, "distributed", None) is not None
            and jax.distributed.is_initialized()
        )
        if distributed:
            multihost_utils.sync_global_devices(f"{matrix_name}-blockwise-done")
        if process_index == 0:
            partial_path.replace(out_path)
            partial_path.with_suffix(
                partial_path.suffix + ".ready"
            ).unlink(missing_ok=True)
        if distributed:
            multihost_utils.sync_global_devices(
                f"{matrix_name}-blockwise-published"
            )


    def build_d_ninv_d(
        self,
        Ninv: jnp.ndarray,
        vis_noisy: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Sum the inverse-variance-weighted squared magnitudes of the visibilities.

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

        Notes
        -----
        Return ``d^H N^{-1} d = sum_i |d_i|**2 / variance_i``, the scalar
        data term in the Gaussian likelihood. This function computes the
        scalar only; ``save_d_ninv_d`` handles persistence.
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
        r"""Project the observed visibilities onto the noise-weighted model responses.

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

        Returns
        -------
        None
            The projected data vector is saved to disk.

        Notes
        -----
        Save ``dbar = T^H N^{-1} d``, with one complex entry per coefficient
        in T's column order. Together with the coefficient precision, this
        vector determines the conditional linear fit and the quadratic term
        in the analytically marginalised likelihood.

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
                jax.jit,
                in_shardings=(P(("host", "dev"), None), P(("host", "dev"))),
                out_shardings=P(None),
            )
            def _build_dbar_cached(
                T: jnp.ndarray, ninv_d: jnp.ndarray
            ) -> jnp.ndarray:
                r"""Sum distributed visibility projections into a replicated coefficient vector.

                Parameters
                ----------
                T : jax.Array
                    Coefficient-to-visibility matrix, shape (n_vis, n_params), with
                    visibility rows distributed across devices.
                ninv_d : jax.Array
                    Inverse-variance-weighted data Ninv * d, shape (n_vis,), in T row order.

                Returns
                -------
                dbar : jax.Array
                    Replicated coefficient projection T^H N^{-1} d, shape (n_params,).
                """
                return jnp.conj(T).T @ ninv_d

            local_T = np.asarray(T_mm[row0:row1, :], dtype=np.complex128)
            if pad_rows:
                local_T = np.pad(local_T, ((0, pad_rows), (0, 0)), mode="constant")
            with jax.set_mesh(mesh):
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
        r"""Save the scalar noise-weighted squared magnitude of the observed data.

        Parameters
        ----------
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.
        matrix_name : str, optional
            Output scalar name. Defaults to ``"d_ninv_d"``.
        ninv_matrix_name : str, optional
            Saved inverse-noise vector name to read from disk. Defaults to
            ``"Ninv"``.

        Returns
        -------
        None
            The noise-weighted data scalar is saved to disk.

        Notes
        -----
        Load the inverse-variance vector, evaluate ``d^H N^{-1} d``, and
        write the scalar on process zero. It is independent of the sky-model
        coefficients and can be reused throughout likelihood evaluation.

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
        r"""Remove named cached arrays before rebuilding their dependent products.

        Parameters
        ----------
        matrix_names : sequence of str
            Logical saved-array names, e.g. ``["T", "T_Ninv_T"]``.

        Returns
        -------
        None
            The named cached files are removed if present.

        Notes
        -----
        Process zero deletes the files and distributed processes synchronise
        before proceeding. The caller supplies the dependency list, for example
        invalidating ``T_Ninv_T`` and ``dbar`` when T changes.
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
        r"""Ensure a requested forward-model or noise product is available on disk.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix to build and save.
        invalidate_existing : bool, optional
            When ``True``, remove any existing saved products invalidated by
            ``matrix_name`` before checking whether a rebuild is required.
            Defaults to ``self.clobber``.

        Returns
        -------
        None
            The requested array and any missing prerequisites are saved to disk.

        Notes
        -----
        Supports ``T``, ``Ninv``, and ``T_Ninv_T`` and builds missing
        prerequisites recursively. Existing files are reused by name unless
        invalidation is requested; their physical configuration is not checked.
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
            self.build_T(matrix_name=matrix_name)
        elif matrix_name == "T_Ninv_T":
            self.build_matrix_if_not_here("T", invalidate_existing=False)
            self.build_matrix_if_not_here("Ninv", invalidate_existing=False)
            self.save_T_Ninv_T_blockwise_memmap(matrix_name=matrix_name)
        elif matrix_name == "Ninv":
            save_matrix_impl(self, matrix_name, self.build_Ninv())
        else:
            raise ValueError(f"Unsupported matrix name '{matrix_name}'.")
        elapsed = time.perf_counter() - t_start
        _log_build(self, f"built and saved {matrix_name} in {elapsed:.3f} s")

    def build_minimum_matrix_stack(self) -> None:
        r"""Prepare the data-independent arrays used by the marginalised likelihood.

        Returns
        -------
        None
            The data-independent likelihood arrays are saved to disk.

        Notes
        -----
        Uses the configured instance state; no explicit input arguments.

        Save the coefficient-to-visibility operator T, inverse-variance vector
        Ninv, and coefficient overlap matrix ``T^H N^{-1} T`` in that order.

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
        r"""Prepare the saved model and data projections for repeated likelihood evaluation.

        Parameters
        ----------
        vis_noisy : jnp.ndarray
            Complex noisy visibility vector ``d`` with shape ``(n_vis,)``.

        Returns
        -------
        int
            Number of model parameters, i.e. the length of ``dbar``.

        Notes
        -----
        This extends the data-independent matrix stack with the two
        data-dependent compressed terms:

        - ``dbar = T^H N^{-1} d``
        - ``d_ninv_d = d^H N^{-1} d``

        Existing files are reused unless ``clobber`` is set. A changed dataset
        therefore requires invalidating its saved projections or a new directory.
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
