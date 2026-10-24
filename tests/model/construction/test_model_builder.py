"""Tests for ``bayeseor_jax.model.construction.model_builder``."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from astropy.constants import c

from bayeseor_jax.configuration import ModelConfig, RunConfig
from bayeseor_jax.io import load_numpy_dict
from bayeseor_jax.model.construction.model_builder import BuildMatrices
from bayeseor_jax.model.construction.operator_construction import (
    T_21_build_save,
    T_fg_build_save,
)
from bayeseor_jax.model.cosmology.fourier_geometry import build_fourier_geometry
from bayeseor_jax.model.observation.image_geometry import build_image_geometry
from bayeseor_jax.model.operators import T_21, T_fg, T_fg_weighted

N_TIME = 2
FREQS_HZ = 150.0e6 + 100.0e3 * np.arange(4)


@pytest.fixture(scope="module")
def image_geometry(hera_latlonalt):
    """Image geometry with a foreground field wider than the EoR field."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_image_geometry(
            fov_eor=20.0, fov_fg=30.0, nside=16, n_t=N_TIME, d_t=10.0,
            jd_center=2459999.1, telescope_latlonalt=hera_latlonalt,
        )


@pytest.fixture(scope="module")
def fourier_geometry():
    """A 3 x 3 x 4 cosmological Fourier grid."""
    return build_fourier_geometry(n_u=3, n_v=3, n_eta=4, du=3.0, dv=3.0, d_eta=1.0 / (4 * 100.0e3))


@pytest.fixture(scope="module")
def uvw(tiny_npy_dir):
    """Fixture baselines for the first two integrations, in metres."""
    return np.asarray(load_numpy_dict(tiny_npy_dir / "uvw_model.npy"))[:N_TIME]


@pytest.fixture
def make_builder(tmp_path, image_geometry, fourier_geometry, uvw):
    """Factory for a ``BuildMatrices`` with the given model settings."""

    def make(**overrides) -> BuildMatrices:
        config = ModelConfig(beam_type="airy", diam=14.0, array_save_directory=tmp_path / "m")
        config = replace(config, **overrides)
        return BuildMatrices(config, frequencies_hz=FREQS_HZ, uvw_array_m=uvw,
                             image_geometry=image_geometry, fourier_geometry=fourier_geometry)

    return make


def _beam_for(builder, mask=None):
    """Return the builder's beam in (time, frequency, pixel) order."""
    beam = np.asarray(builder.beam_response).transpose(1, 0, 2)
    return beam if mask is None else beam[:, :, mask]


class TestInit:
    def test_resolved_dimensions(self, make_builder, uvw, image_geometry):
        """Dimensions come from the resolved frequencies, baselines and geometry."""
        builder = make_builder()
        assert (builder.n_freq, builder.n_t, builder.nbls) == (4, N_TIME, uvw.shape[1])
        assert builder.f0_mhz == pytest.approx(150.0)
        assert builder.n_fg_spatial_modes == image_geometry.n_pix_fg

    def test_baselines_in_wavelengths(self, make_builder, uvw):
        """``inst_uvw_irad`` is the first integration's baselines divided by wavelength."""
        builder = make_builder()
        wavelengths = c.to_value("m/s") / FREQS_HZ
        expected = uvw[0].T[None, :, :] / wavelengths[:, None, None]
        np.testing.assert_allclose(builder.inst_uvw_irad, expected, rtol=1e-12)

    @pytest.mark.parametrize(
        ("beta_fg", "expected"),
        [(None, list(ModelConfig().beta_fg)), (2.5, [2.5]), ((2.0, 2.5, 3.0), [2.0, 2.5, 3.0])],
    )
    def test_spectral_indices(self, make_builder, beta_fg, expected):
        """``beta_fg`` accepts None, a scalar or a sequence, setting the component count."""
        builder = make_builder(beta_fg=beta_fg)
        assert builder.beta_fg == expected
        assert builder.n_fg_components == 1 + len(expected)
        assert builder.foreground_spectral_operator.shape == (4, 1 + len(expected))

    def test_creates_save_directory(self, make_builder, tmp_path):
        """The array save directory is created."""
        make_builder()
        assert (tmp_path / "m").is_dir()

    def test_beam_zero_below_horizon(self, make_builder, image_geometry):
        """Beam weights vanish where the zenith angle is at or beyond the horizon."""
        builder = make_builder()
        za = np.asarray(image_geometry.az_za_fg)[:, 1, :]
        below = np.broadcast_to(za >= np.pi / 2, np.asarray(builder.beam_response).shape)
        assert np.all(np.asarray(builder.beam_response)[below] == 0)

    def test_foreground_only_drops_fourier_factors(self, make_builder):
        """A foreground-only model ignores the Fourier grid."""
        builder = make_builder(model="fg")
        assert builder.fourier_geometry is None
        assert not hasattr(builder, "transverse_fourier_operator")

    def test_eor_only_has_no_foreground_spectra(self, make_builder):
        """An EoR-only model does not build foreground spectral factors."""
        builder = make_builder(model="eor")
        assert not hasattr(builder, "foreground_spectral_operator")

    def test_accepts_run_config(self, tmp_path, image_geometry, fourier_geometry, uvw):
        """A ``RunConfig`` is accepted as a ``ModelConfig``."""
        config = RunConfig(beam_type="airy", diam=14.0, array_save_directory=tmp_path / "m")
        builder = BuildMatrices(config, frequencies_hz=FREQS_HZ, uvw_array_m=uvw,
                                image_geometry=image_geometry, fourier_geometry=fourier_geometry)
        assert builder.model == "eor+fg"

    def test_rejects_non_config(self, image_geometry, uvw):
        """A non-configuration object raises TypeError."""
        with pytest.raises(TypeError, match="ModelConfig"):
            BuildMatrices({"model": "fg"}, frequencies_hz=FREQS_HZ, uvw_array_m=uvw,
                          image_geometry=image_geometry)

    def test_rejects_unknown_model(self, make_builder):
        """An unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="model must be"):
            make_builder(model="eor+fg+beam")

    def test_cosmology_requires_fourier_geometry(self, tmp_path, image_geometry, uvw):
        """A model with the EoR requires a Fourier grid."""
        config = ModelConfig(model="eor", array_save_directory=tmp_path)
        with pytest.raises(ValueError, match="Fourier geometry"):
            BuildMatrices(config, frequencies_hz=FREQS_HZ, uvw_array_m=uvw, image_geometry=image_geometry)

    @pytest.mark.parametrize(
        "frequencies",
        [np.array([]), -FREQS_HZ, FREQS_HZ[:, None], np.array([1.5e8, np.nan]), FREQS_HZ.astype(complex)],
    )
    def test_rejects_invalid_frequencies(self, tmp_path, image_geometry, fourier_geometry, uvw, frequencies):
        """Empty, negative, non-vector, nonfinite or complex frequencies raise ValueError."""
        config = ModelConfig(array_save_directory=tmp_path)
        with pytest.raises(ValueError, match="frequencies_hz"):
            BuildMatrices(config, frequencies_hz=frequencies, uvw_array_m=uvw,
                          image_geometry=image_geometry, fourier_geometry=fourier_geometry)

    @pytest.mark.parametrize(
        "change",
        [lambda u: u[0], lambda u: u[..., :2], lambda u: u[:1], lambda u: u[:, :0], lambda u: u * np.nan],
    )
    def test_rejects_invalid_baselines(self, tmp_path, image_geometry, fourier_geometry, uvw, change):
        """Baselines with the wrong shape, time axis or nonfinite values raise ValueError."""
        config = ModelConfig(array_save_directory=tmp_path)
        with pytest.raises(ValueError, match="uvw_array_m"):
            BuildMatrices(config, frequencies_hz=FREQS_HZ, uvw_array_m=change(uvw),
                          image_geometry=image_geometry, fourier_geometry=fourier_geometry)


class TestT21BuildSave:
    def test_matches_in_memory_operator(self, make_builder, image_geometry):
        """The saved operator equals ``T_21`` on the EoR pixels with the builder's factors."""
        builder = make_builder()
        mask = np.asarray(image_geometry.eor_fg_shared_axis)
        path = builder.T_21_build_save()
        expected = T_21(np.asarray(image_geometry.lmn_fg)[:, :, mask], builder.inst_uvw_irad,
                        _beam_for(builder, mask), image_geometry.pixel_area_sr,
                        builder.transverse_fourier_operator, builder.spectral_fourier_operator)
        assert path.name == "T_21.npy"
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)

    def test_matches_module_writer(self, make_builder, image_geometry, tmp_path):
        """The method writes the same matrix as the module-level writer."""
        builder = make_builder()
        mask = np.asarray(image_geometry.eor_fg_shared_axis)
        expected = T_21_build_save(
            np.asarray(image_geometry.lmn_fg)[:, :, mask], builder.inst_uvw_irad, _beam_for(builder, mask),
            image_geometry.pixel_area_sr, builder.transverse_fourier_operator,
            builder.spectral_fourier_operator, output_path=tmp_path / "ref.npy",
        )
        np.testing.assert_array_equal(np.load(builder.T_21_build_save()), np.load(expected))

    def test_uses_only_eor_pixels(self, make_builder, image_geometry):
        """The EoR operator covers fewer pixels than the wider foreground field."""
        assert np.sum(image_geometry.eor_fg_shared_axis) < image_geometry.n_pix_fg
        builder = make_builder()
        assert builder.transverse_fourier_operator.shape[0] == int(np.sum(image_geometry.eor_fg_shared_axis))

    def test_custom_matrix_name(self, make_builder):
        """``matrix_name`` sets the output file stem."""
        assert make_builder().T_21_build_save(matrix_name="custom").name == "custom.npy"

    def test_foreground_only_model_raises(self, make_builder):
        """A foreground-only model has no EoR operator."""
        with pytest.raises(ValueError, match="no cosmological operator"):
            make_builder(model="fg").T_21_build_save()


class TestTfgBuildSave:
    def test_matches_in_memory_operator(self, make_builder, image_geometry):
        """The saved operator equals ``T_fg`` on all foreground pixels."""
        builder = make_builder()
        path = builder.T_fg_build_save()
        expected = T_fg(image_geometry.lmn_fg, builder.inst_uvw_irad, _beam_for(builder),
                        image_geometry.pixel_area_sr, builder.foreground_spectral_operator)
        assert path.name == "T_fg.npy"
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)

    def test_matches_module_writer(self, make_builder, image_geometry, tmp_path):
        """The method writes the same matrix as the module-level writer."""
        builder = make_builder()
        expected = T_fg_build_save(image_geometry.lmn_fg, builder.inst_uvw_irad, _beam_for(builder),
                                   image_geometry.pixel_area_sr, builder.foreground_spectral_operator,
                                   output_path=tmp_path / "ref.npy")
        np.testing.assert_array_equal(np.load(builder.T_fg_build_save()), np.load(expected))

    def test_eor_only_model_raises(self, make_builder):
        """An EoR-only model has no foreground operator."""
        with pytest.raises(ValueError, match="no foreground operator"):
            make_builder(model="eor").T_fg_build_save()


class TestTfgWeightedBuildSave:
    def test_matches_in_memory_operator(self, make_builder, image_geometry, rng):
        """The saved weighted operator equals ``T_fg_weighted`` with the same weights."""
        builder = make_builder()
        n_vis = builder.n_t * builder.n_freq * builder.nbls
        prior = rng.uniform(0.5, 2.0, size=builder.n_fg_components * builder.n_fg_spatial_modes)
        whitener = rng.uniform(0.5, 2.0, size=n_vis)
        path = builder.T_fg_weighted_build_save(prior_sqrt=prior, noise_whitener=whitener)
        expected = T_fg_weighted(image_geometry.lmn_fg, builder.inst_uvw_irad, _beam_for(builder),
                                 image_geometry.pixel_area_sr, builder.foreground_spectral_operator,
                                 prior_sqrt=prior, noise_whitener=whitener)
        assert path.name == "A_fg.npy"
        np.testing.assert_allclose(np.load(path), expected, rtol=1e-12, atol=1e-14)

    def test_eor_only_model_raises(self, make_builder):
        """An EoR-only model has no weighted foreground operator."""
        with pytest.raises(ValueError, match="no foreground operator"):
            make_builder(model="eor").T_fg_weighted_build_save(prior_sqrt=np.ones(1), noise_whitener=np.ones(1))
