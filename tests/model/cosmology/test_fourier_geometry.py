"""Tests for ``bayeseor_jax.model.cosmology.fourier_geometry``."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from bayeseor_jax.model.cosmology.fourier_geometry import (
    FourierGeometry,
    _build_eta_modes,
    _build_uv_modes,
    _resolve_model_grid_size,
    _resolve_uv_grid_from_extents,
    _resolve_uv_spacing_from_image_extents,
    build_fourier_geometry,
)
from bayeseor_jax.model.instrument.beam import Beam_Model

C_MS = 299792458.0


class TestFourierGeometry:
    def test_is_frozen(self):
        """FourierGeometry instances cannot be modified after construction."""
        geometry = FourierGeometry(uv=np.zeros((2, 1)), eta=np.zeros(1), du=1.0, dv=1.0, d_eta=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            geometry.du = 2.0


class TestBuildUvModes:
    @pytest.mark.parametrize("n_u, n_v", [(3, 3), (4, 3), (3, 5), (1, 3), (2, 2)])
    def test_matches_centred_grid_without_monopole(self, n_u, n_v):
        """Modes are the centred grid, u fastest, with only (0, 0) removed."""
        du, dv = 0.5, 2.0
        u = (np.arange(n_u) - n_u // 2) * du
        v = (np.arange(n_v) - n_v // 2) * dv
        expected = [(ui, vi) for vi in v for ui in u if not (ui == 0 and vi == 0)]
        result = np.asarray(_build_uv_modes(n_u, n_v, du, dv))
        assert result.shape == (2, n_u * n_v - 1)
        np.testing.assert_allclose(result.T, np.array(expected))

    def test_monopole_absent(self):
        """No retained pair is the transverse monopole."""
        uv = np.asarray(_build_uv_modes(5, 5, 1.0, 1.0))
        assert not np.any((uv[0] == 0) & (uv[1] == 0))

    def test_u_varies_fastest(self):
        """Consecutive modes within a v row differ only in u."""
        uv = np.asarray(_build_uv_modes(3, 3, 1.0, 1.0))
        np.testing.assert_allclose(uv[:, :3], [[-1, 0, 1], [-1, -1, -1]])


class TestBuildEtaModes:
    @pytest.mark.parametrize("n_eta", [2, 3, 4, 7])
    def test_centred_without_zero(self, n_eta):
        """η modes are centred on zero, with η = 0 removed."""
        d_eta = 1e-6
        full = (np.arange(n_eta) - n_eta // 2) * d_eta
        expected = full[full != 0]
        result = np.asarray(_build_eta_modes(n_eta, d_eta))
        assert result.shape == (n_eta - 1,)
        np.testing.assert_allclose(result, expected)

    def test_single_mode_leaves_nothing(self):
        """With one η mode, only η = 0 exists and it is removed."""
        assert _build_eta_modes(1, 1.0).shape == (0,)


class TestBuildFourierGeometry:
    def test_combines_uv_and_eta(self):
        """The geometry stores the uv modes, η modes and spacings."""
        geometry = build_fourier_geometry(n_u=3, n_v=5, n_eta=4, du=0.5, dv=0.25, d_eta=1e-6)
        np.testing.assert_allclose(geometry.uv, _build_uv_modes(3, 5, 0.5, 0.25))
        np.testing.assert_allclose(geometry.eta, _build_eta_modes(4, 1e-6))
        assert (geometry.du, geometry.dv, geometry.d_eta) == (0.5, 0.25, 1e-6)

    def test_parameter_count(self):
        """The grid has (N_u N_v − 1)(N_η − 1) coefficients."""
        geometry = build_fourier_geometry(n_u=5, n_v=3, n_eta=6, du=1.0, dv=1.0, d_eta=1.0)
        assert geometry.uv.shape[1] * geometry.eta.shape[0] == (5 * 3 - 1) * (6 - 1)

    def test_spacings_are_floats(self):
        """Integer spacings are stored as Python floats."""
        geometry = build_fourier_geometry(n_u=3, n_v=3, n_eta=3, du=1, dv=2, d_eta=3)
        assert all(isinstance(x, float) for x in (geometry.du, geometry.dv, geometry.d_eta))

    @pytest.mark.parametrize("field", ["n_u", "n_v", "n_eta"])
    def test_rejects_non_positive_dimensions(self, field):
        """A grid dimension below one raises ValueError."""
        kwargs = dict(n_u=3, n_v=3, n_eta=3, du=1.0, dv=1.0, d_eta=1.0)
        kwargs[field] = 0
        with pytest.raises(ValueError, match="dimensions"):
            build_fourier_geometry(**kwargs)

    @pytest.mark.parametrize("field", ["du", "dv", "d_eta"])
    def test_rejects_non_positive_spacings(self, field):
        """A non-positive spacing raises ValueError."""
        kwargs = dict(n_u=3, n_v=3, n_eta=3, du=1.0, dv=1.0, d_eta=1.0)
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match="spacings"):
            build_fourier_geometry(**kwargs)


class TestResolveUvSpacingFromImageExtents:
    def test_derives_reciprocal_extents(self):
        """Omitted spacings are the reciprocals of the angular extents."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            du, dv = _resolve_uv_spacing_from_image_extents(extent_u=0.2, extent_v=0.5, du=None, dv=None, label="x")
        assert du == pytest.approx(5.0)
        assert dv == pytest.approx(2.0)

    def test_close_override_is_kept_silently(self):
        """An override within 5% of the derived value is kept without warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            du, dv = _resolve_uv_spacing_from_image_extents(extent_u=0.2, extent_v=0.5, du=5.2, dv=1.95, label="x")
        assert (du, dv) == (5.2, 1.95)

    @pytest.mark.parametrize("du, dv, axis", [(6.0, None, "du"), (None, 3.0, "dv")])
    def test_inconsistent_override_warns(self, du, dv, axis):
        """An override more than 5% from 1/extent warns and is still used."""
        with pytest.warns(RuntimeWarning, match=axis):
            result = _resolve_uv_spacing_from_image_extents(extent_u=0.2, extent_v=0.5, du=du, dv=dv, label="x")
        assert result[0 if axis == "du" else 1] == (du if axis == "du" else dv)

    @pytest.mark.parametrize("extent_u, extent_v", [(0.0, 1.0), (1.0, -1.0)])
    def test_rejects_non_positive_extents(self, extent_u, extent_v):
        """Non-positive angular extents raise ValueError."""
        with pytest.raises(ValueError, match="extent"):
            _resolve_uv_spacing_from_image_extents(extent_u=extent_u, extent_v=extent_v, du=None, dv=None, label="x")


class TestResolveModelGridSize:
    def test_omitted_uses_requirement(self):
        """An omitted size takes the required value."""
        assert _resolve_model_grid_size(value=None, required=7, label="n") == 7

    def test_requirement_floored_at_one(self):
        """A requirement below one is raised to one."""
        assert _resolve_model_grid_size(value=None, required=0, label="n") == 1

    def test_larger_override_kept(self):
        """An explicit size at or above the requirement is kept silently."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_model_grid_size(value=9, required=7, label="n") == 9

    def test_undersized_override_warns(self):
        """An explicit size below the requirement warns and is kept."""
        with pytest.warns(RuntimeWarning, match="smaller than"):
            assert _resolve_model_grid_size(value=3, required=7, label="n") == 3

    def test_rejects_non_positive_override(self):
        """An explicit size below one raises ValueError."""
        with pytest.raises(ValueError, match=">= 1"):
            _resolve_model_grid_size(value=0, required=7, label="n")


class TestResolveUvGridFromExtents:
    @pytest.fixture
    def inputs(self):
        """Two baselines and three channels for an Airy beam of diameter 14 m."""
        return dict(
            extent_u_eor=0.2,
            extent_v_eor=0.25,
            freqs_hz=np.array([150e6, 151e6, 152e6]),
            uvws_m=np.array([[[14.6, 0.0, 0.0], [7.3, 12.6, 0.0]]]),
            beam_type="airy",
            fwhm_deg=None,
            diam=14.0,
            n_u_eor=None,
            n_v_eor=None,
            du_eor=None,
            dv_eor=None,
        )

    def test_required_sizes_follow_formula(self, inputs):
        """N = 2⌈(|u|_max + aperture/2)/Δu⌉ + 1 using the shortest wavelength."""
        result = _resolve_uv_grid_from_extents(**inputs)
        wavelength = C_MS / 152e6
        aperture = Beam_Model(beam_type="airy", diam=14.0).aperture_width_uv(freqs_hz=inputs["freqs_hz"])
        u_extent = 14.6 / wavelength + 0.5 * aperture
        v_extent = 12.6 / wavelength + 0.5 * aperture
        assert result["du_eor"] == pytest.approx(5.0)
        assert result["dv_eor"] == pytest.approx(4.0)
        assert result["u_max_inst_model"] == pytest.approx(14.6 / wavelength)
        assert result["v_max_inst_model"] == pytest.approx(12.6 / wavelength)
        assert result["beam_aperture_width_uv"] == pytest.approx(aperture)
        assert result["u_extent_with_buffer"] == pytest.approx(u_extent)
        assert result["n_u_eor_required"] == int(np.ceil(u_extent / 5.0)) * 2 + 1
        assert result["n_v_eor_required"] == int(np.ceil(v_extent / 4.0)) * 2 + 1
        assert result["n_u_eor"] == result["n_u_eor_required"]
        assert result["n_v_eor"] == result["n_v_eor_required"]

    def test_required_sizes_are_odd(self, inputs):
        """Required sizes are odd, so the grid is symmetric about zero."""
        result = _resolve_uv_grid_from_extents(**inputs)
        assert result["n_u_eor_required"] % 2 == 1
        assert result["n_v_eor_required"] % 2 == 1

    def test_uniform_beam_adds_no_buffer(self, inputs):
        """A uniform beam contributes no aperture buffer."""
        inputs.update(beam_type="uniform", diam=None)
        result = _resolve_uv_grid_from_extents(**inputs)
        assert result["beam_aperture_width_uv"] == 0.0
        assert result["u_extent_with_buffer"] == pytest.approx(result["u_max_inst_model"])

    def test_explicit_sizes_override(self, inputs):
        """Explicit sizes are returned, warning when undersized."""
        inputs.update(n_u_eor=1)
        with pytest.warns(RuntimeWarning, match="n_u_eor"):
            result = _resolve_uv_grid_from_extents(**inputs)
        assert result["n_u_eor"] == 1

    def test_rejects_empty_baselines(self, inputs):
        """An empty baseline array raises ValueError."""
        inputs.update(uvws_m=np.zeros((0, 3)))
        with pytest.raises(ValueError, match="baseline"):
            _resolve_uv_grid_from_extents(**inputs)
