"""Tests for ``bayeseor_jax.data.vis``: loading and vectorising visibilities."""

from __future__ import annotations

import shutil
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c, k_B
from pyuvdata import UVData

from bayeseor_jax.data.noise import generate_gaussian_noise
from bayeseor_jax.data.vis import (
    form_pI_vis,
    get_vis_data,
    jy_to_ksr,
    preprocess_uvdata,
    uvd_to_vector,
)

FREQS_HZ = 150.0e6 + 100.0e3 * np.arange(4)
CROSS_ANTPAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
NPY_AXES = dict(n_freq=4, f0_mhz=150.0, df_mhz=0.1, n_t=4, d_t=10.0)
#: Three integrations, exercising the odd-Ntimes branch of ``calc_noise``.
ODD_TIMES = dict(jd_idx_min=0, n_times_select=3)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _rj_mk_per_jy(freqs_hz):
    """Independent Rayleigh–Jeans conversion from Jy to mK sr for a 1 sr beam."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    k_per_jy = 1e-26 * c.value**2 / (2.0 * k_B.value * freqs_hz**2)
    return 1e3 * k_per_jy


def _read(path):
    """Read a UVData file in full."""
    uvd = UVData()
    uvd.read(str(path))
    return uvd


def _reference_vector(uvd, antpairs, combine, freqs_hz):
    """Build the expected vector from ``uvd.get_data`` in time/frequency/baseline order."""
    blocks = np.stack([combine(uvd, ap) for ap in antpairs], axis=-1)  # (t, f, b)
    blocks = blocks * _rj_mk_per_jy(freqs_hz)[None, :, None]
    full = np.concatenate([blocks, blocks.conj()], axis=-1)
    return full.reshape(-1)


def _pseudo_i(uvd, antpair):
    """Return XX + YY for one antenna pair."""
    return uvd.get_data(antpair + ("xx",)) + uvd.get_data(antpair + ("yy",))


@pytest.fixture
def raw_uvd(tiny_uvh5):
    """Full raw fixture observation, read fresh for each test."""
    return _read(tiny_uvh5)


@pytest.fixture
def uvh5_copy(tmp_path, tiny_uvh5):
    """Writable copy of the fixture file in ``tmp_path``."""
    path = tmp_path / "copy.uvh5"
    shutil.copy(tiny_uvh5, path)
    return path


# ---------------------------------------------------------------------------
# jy_to_ksr
# ---------------------------------------------------------------------------

class TestJyToKsr:
    """Jansky to (milli)kelvin-steradian conversion."""

    def test_matches_rayleigh_jeans_in_millikelvin(self, rng):
        """Each channel is scaled by the Rayleigh–Jeans factor in mK sr."""
        data = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
        out = jy_to_ksr(data, FREQS_HZ, mK=True)
        np.testing.assert_allclose(out, data * _rj_mk_per_jy(FREQS_HZ)[None, :], rtol=1e-10)

    def test_kelvin_is_thousandth_of_millikelvin(self, rng):
        """The K sr result is 1000 times smaller than the mK sr result."""
        data = rng.normal(size=(3, 4))
        np.testing.assert_allclose(
            jy_to_ksr(data, FREQS_HZ) * 1e3, jy_to_ksr(data, FREQS_HZ, mK=True), rtol=1e-12
        )

    def test_accepts_astropy_quantity(self, rng):
        """Frequencies given as an astropy Quantity in MHz give the same result."""
        data = rng.normal(size=(2, 4))
        np.testing.assert_allclose(
            jy_to_ksr(data, (FREQS_HZ / 1e6) * u.MHz, mK=True),
            jy_to_ksr(data, FREQS_HZ, mK=True),
            rtol=1e-12,
        )

    def test_one_dimensional_input_gains_leading_axis(self):
        """A 1-D frequency spectrum is returned with a leading singleton axis."""
        out = jy_to_ksr(np.ones(4), FREQS_HZ, mK=True)
        assert out.shape == (1, 4)
        np.testing.assert_allclose(out[0], _rj_mk_per_jy(FREQS_HZ), rtol=1e-10)


# ---------------------------------------------------------------------------
# form_pI_vis
# ---------------------------------------------------------------------------

class TestFormPIVis:
    """Pseudo-Stokes I formation."""

    @pytest.mark.parametrize("norm", [1.0, 0.5])
    def test_sums_xx_and_yy_with_normalisation(self, raw_uvd, norm):
        """pI equals norm * (XX + YY) and replaces the polarisation axis."""
        expected = norm * _pseudo_i(raw_uvd, (0, 1))
        out = form_pI_vis(raw_uvd, norm=norm)
        assert out.Npols == 1
        assert out.get_pols() == ["pI"]
        np.testing.assert_allclose(out.get_data((0, 1, "pI")), expected, rtol=1e-12)

    def test_existing_pseudo_i_is_left_unchanged(self, raw_uvd):
        """Data already in pI are returned unchanged and the norm is not reapplied."""
        once = form_pI_vis(raw_uvd, norm=1.0)
        before = once.data_array.copy()
        twice = form_pI_vis(once, norm=3.0)
        np.testing.assert_array_equal(twice.data_array, before)

    def test_rejects_non_uvdata(self):
        """A non-UVData input fails the type assertion."""
        with pytest.raises(AssertionError):
            form_pI_vis(np.zeros(3))


# ---------------------------------------------------------------------------
# uvd_to_vector
# ---------------------------------------------------------------------------

class TestUvdToVector:
    """Vectorisation of a single-polarisation UVData object."""

    @pytest.fixture
    def xx_cross(self, raw_uvd):
        """Cross-correlation XX data in time-major order."""
        raw_uvd.select(ant_str="cross")
        raw_uvd.select(polarizations=["xx"])
        raw_uvd.reorder_blts(order="time")
        return raw_uvd

    def test_time_frequency_baseline_order_with_conjugates(self, xx_cross):
        """Row (t*Nf + f)*2Nb + b holds baseline b, and row +Nb its conjugate."""
        vec, noise = uvd_to_vector(xx_cross, CROSS_ANTPAIRS, pol="xx")
        assert noise is None
        n_b = len(CROSS_ANTPAIRS)
        for b, ap in enumerate(CROSS_ANTPAIRS):
            data = xx_cross.get_data(ap + ("xx",))
            for t in range(4):
                for f in range(4):
                    row = (t * 4 + f) * 2 * n_b + b
                    assert vec[row] == data[t, f]
                    assert vec[row + n_b] == np.conj(data[t, f])

    def test_even_odd_noise_estimate_for_even_times(self, xx_cross):
        """With even Ntimes both members of each pair receive (even - odd)."""
        _, noise = uvd_to_vector(xx_cross, CROSS_ANTPAIRS, pol="xx", calc_noise=True)
        data = xx_cross.get_data((0, 2, "xx"))
        diff = data[::2] - data[1::2]
        expected = np.repeat(diff, 2, axis=0)
        got = noise.reshape(4, 4, 12)[:, :, 1]
        np.testing.assert_allclose(got, expected)
        np.testing.assert_allclose(noise.reshape(4, 4, 12)[:, :, 7], expected.conj())

    def test_even_odd_noise_estimate_for_odd_times(self, xx_cross):
        """With odd Ntimes the last even sample is differenced with the previous one."""
        times = np.unique(xx_cross.time_array)[:3]
        xx_cross.select(times=times)
        _, noise = uvd_to_vector(xx_cross, CROSS_ANTPAIRS, pol="xx", calc_noise=True)
        data = xx_cross.get_data((0, 1, "xx"))
        diff = np.stack([data[0] - data[1], data[2] - data[1]])
        expected = np.stack([diff[0], diff[0], diff[1]])
        np.testing.assert_allclose(noise.reshape(3, 4, 12)[:, :, 0], expected)

    @pytest.mark.parametrize("calc_noise", [False, True])
    def test_single_integration_requires_noise_estimation_disabled(self, xx_cross, calc_noise):
        """A single integration is valid only when time-difference noise estimation is disabled."""
        xx_cross.select(times=np.unique(xx_cross.time_array)[:1])
        if calc_noise:
            with pytest.raises(ValueError, match="at least two integrations"):
                uvd_to_vector(xx_cross, CROSS_ANTPAIRS, pol="xx", calc_noise=True)
        else:
            vec, noise = uvd_to_vector(xx_cross, CROSS_ANTPAIRS, pol="xx")
            assert vec.shape == (4 * 12,)
            assert noise is None

    def test_rejects_multiple_polarisations(self, raw_uvd):
        """Input with more than one polarisation raises ValueError."""
        raw_uvd.select(ant_str="cross")
        with pytest.raises(ValueError, match="more than one"):
            uvd_to_vector(raw_uvd, CROSS_ANTPAIRS)


# ---------------------------------------------------------------------------
# preprocess_uvdata
# ---------------------------------------------------------------------------

class TestPreprocessUvdataDefaults:
    """Default preprocessing of the fixture file."""

    def test_output_shapes(self, tiny_uvh5):
        """Six cross baselines and their conjugates give 4×4×12 visibilities."""
        vis, antpairs, uvws, red, noise = preprocess_uvdata(tiny_uvh5)
        assert vis.shape == (4 * 4 * 12,)
        assert uvws.shape == (4, 12, 3)
        assert red.shape == (4, 12, 1)
        assert noise is None

    def test_antpairs_sorted_then_reversed(self, tiny_uvh5):
        """Antenna pairs are sorted cross pairs followed by the same pairs reversed."""
        _, antpairs, _, _, _ = preprocess_uvdata(tiny_uvh5)
        assert antpairs == CROSS_ANTPAIRS + [ap[::-1] for ap in CROSS_ANTPAIRS]

    def test_conjugate_baselines_are_negated(self, tiny_uvh5):
        """The second half of the baseline axis is minus the first half."""
        _, _, uvws, _, _ = preprocess_uvdata(tiny_uvh5)
        np.testing.assert_allclose(uvws[:, 6:], -uvws[:, :6])

    def test_baselines_match_antenna_positions(self, tiny_uvh5):
        """Baseline (0, 1) is the 14.6 m east-west separation of its antennas."""
        _, _, uvws, _, _ = preprocess_uvdata(tiny_uvh5)
        np.testing.assert_allclose(uvws[0, 0], [14.6, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(uvws[0, 2]), np.hypot(21.9, 12.644), rtol=1e-6)

    def test_redundancy_defaults_to_one(self, tiny_uvh5):
        """Without averaging every baseline has unit redundancy."""
        _, _, _, red, _ = preprocess_uvdata(tiny_uvh5)
        np.testing.assert_array_equal(red, 1.0)

    def test_values_match_independent_reference(self, tiny_uvh5, raw_uvd):
        """The vector equals RJ-converted XX + YY built directly from pyuvdata."""
        vis, _, _, _, _ = preprocess_uvdata(tiny_uvh5)
        expected = _reference_vector(raw_uvd, CROSS_ANTPAIRS, _pseudo_i, FREQS_HZ)
        np.testing.assert_allclose(vis, expected, rtol=1e-10)

    def test_return_uvd_is_pseudo_i(self, tiny_uvh5):
        """``return_uvd`` appends the processed single-polarisation UVData."""
        *_, uvd = preprocess_uvdata(tiny_uvh5, return_uvd=True)
        assert uvd.get_pols() == ["pI"]
        assert uvd.Nbls == 6

    def test_missing_file_raises(self, tmp_path):
        """A path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocess_uvdata(tmp_path / "missing.uvh5")


class TestPreprocessUvdataUnits:
    """Unit handling of raw files."""

    def test_kelvin_steradian_input_is_scaled_to_millikelvin(self, uvh5_copy, raw_uvd):
        """Data labelled K sr are multiplied by 1000 rather than RJ-converted."""
        uvd = _read(uvh5_copy)
        uvd.vis_units = "K str"
        uvd.write_uvh5(str(uvh5_copy), clobber=True)
        vis, _, _, _, _ = preprocess_uvdata(uvh5_copy)
        expected = 1e3 * _pseudo_i(raw_uvd, (0, 1))
        np.testing.assert_allclose(vis.reshape(4, 4, 12)[:, :, 0], expected, rtol=1e-12)

    def test_uncalibrated_input_is_rejected(self, uvh5_copy):
        """Data with units other than Jy or K sr raise ValueError."""
        uvd = _read(uvh5_copy)
        uvd.vis_units = "uncalib"
        uvd.write_uvh5(str(uvh5_copy), clobber=True)
        with pytest.raises(ValueError, match="calibrated"):
            preprocess_uvdata(uvh5_copy)


class TestPreprocessUvdataPolarisation:
    """Polarisation reduction."""

    @pytest.mark.parametrize("pol", ["xx", "yy"])
    def test_single_polarisation_selection(self, tiny_uvh5, raw_uvd, pol):
        """``form_pi=False`` keeps only the requested linear polarisation."""
        vis, _, _, _, _ = preprocess_uvdata(tiny_uvh5, form_pi=False, pol=pol)
        expected = _reference_vector(
            raw_uvd, CROSS_ANTPAIRS, lambda d, ap: d.get_data(ap + (pol,)), FREQS_HZ
        )
        np.testing.assert_allclose(vis, expected, rtol=1e-10)

    def test_missing_polarisation_raises(self, tiny_uvh5):
        """Requesting a polarisation absent from the file raises ValueError."""
        with pytest.raises(ValueError, match="not present"):
            preprocess_uvdata(tiny_uvh5, form_pi=False, pol="xy")

    def test_pi_norm_scales_the_vector(self, tiny_uvh5):
        """``pi_norm`` multiplies the pseudo-Stokes I vector linearly."""
        unit, *_ = preprocess_uvdata(tiny_uvh5, pi_norm=1.0)
        half, *_ = preprocess_uvdata(tiny_uvh5, pi_norm=0.5)
        np.testing.assert_allclose(half, 0.5 * unit, rtol=1e-12)


class TestPreprocessUvdataBaselines:
    """Baseline selection, flagging and redundancy."""

    def test_all_antennas_includes_autocorrelations(self, tiny_uvh5):
        """``antenna_selection='all'`` keeps the four autos alongside six crosses."""
        _, antpairs, uvws, _, _ = preprocess_uvdata(tiny_uvh5, antenna_selection="all")
        assert len(antpairs) == 20
        assert (0, 0) in antpairs

    def test_explicit_antenna_pairs(self, tiny_uvh5):
        """A pyuvdata antenna string keeps exactly the requested pairs."""
        _, antpairs, _, _, _ = preprocess_uvdata(tiny_uvh5, antenna_selection="0_1,2_3")
        assert antpairs == [(0, 1), (2, 3), (1, 0), (3, 2)]

    @pytest.mark.parametrize("cutoff", [15.0, 15.0 * u.m])
    def test_baseline_cutoff_uses_projected_length(self, tiny_uvh5, cutoff):
        """Only the 25.3 m baseline (0, 3) exceeds a 15 m cutoff."""
        _, antpairs, uvws, _, _ = preprocess_uvdata(tiny_uvh5, bl_cutoff=cutoff)
        assert (0, 3) not in antpairs
        assert len(antpairs) == 10
        assert np.all(np.hypot(uvws[..., 0], uvws[..., 1]) <= 15.0)

    def test_fully_flagged_baseline_is_removed(self, uvh5_copy):
        """A baseline flagged at every time and frequency is dropped."""
        uvd = _read(uvh5_copy)
        uvd.flag_array[uvd.antpair2ind(0, 3)] = True
        uvd.write_uvh5(str(uvh5_copy), clobber=True)
        _, antpairs, _, _, _ = preprocess_uvdata(uvh5_copy)
        assert (0, 3) not in antpairs
        assert len(antpairs) == 10

    def test_partially_flagged_baseline_is_kept(self, uvh5_copy):
        """A partially flagged baseline is kept and its data are not masked."""
        uvd = _read(uvh5_copy)
        rows = np.arange(uvd.Nblts)[uvd.antpair2ind(0, 3)]
        uvd.flag_array[rows[0]] = True
        uvd.write_uvh5(str(uvh5_copy), clobber=True)
        _, antpairs, _, _, _ = preprocess_uvdata(uvh5_copy)
        assert (0, 3) in antpairs

    def test_redundant_average_groups_and_weights(self, tiny_uvh5):
        """Two redundant pairs collapse to one baseline each with redundancy two."""
        _, antpairs, _, red, _ = preprocess_uvdata(tiny_uvh5, redundant_avg=True)
        assert antpairs[:4] == [(0, 1), (0, 2), (0, 3), (1, 2)]
        np.testing.assert_array_equal(red[0, :, 0], [2, 2, 1, 1, 2, 2, 1, 1])

    def test_redundant_average_is_group_mean(self, tiny_uvh5, raw_uvd):
        """The averaged (0, 1) visibility is the mean of (0, 1) and (2, 3)."""
        vis, *_ = preprocess_uvdata(tiny_uvh5, redundant_avg=True)
        mean = 0.5 * (_pseudo_i(raw_uvd, (0, 1)) + _pseudo_i(raw_uvd, (2, 3)))
        expected = mean * _rj_mk_per_jy(FREQS_HZ)[None, :]
        np.testing.assert_allclose(vis.reshape(4, 4, 8)[:, :, 0], expected, rtol=1e-10)

    def test_uniform_redundancy_after_averaging(self, tiny_uvh5):
        """``uniform_redundancy`` keeps unit weights even after averaging."""
        _, antpairs, _, red, _ = preprocess_uvdata(
            tiny_uvh5, redundant_avg=True, uniform_redundancy=True
        )
        assert len(antpairs) == 8
        np.testing.assert_array_equal(red, 1.0)


class TestPreprocessUvdataFrequencySelection:
    """Frequency-window selection."""

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (dict(freq_idx_min=1), [1, 2, 3]),
            (dict(freq_idx_min=1, n_freqs_select=2), [1, 2]),
            (dict(freq_min=150.15e6), [2, 3]),
            (dict(freq_min=150.15 * u.MHz, n_freqs_select=1), [2]),
            (dict(freq_center=150.2e6, n_freqs_select=2), [1, 2]),
            (dict(freq_center=150.2e6, n_freqs_select=3), [1, 2, 3]),
            (dict(n_freqs_select=2), [0, 1, 2, 3]),
        ],
    )
    def test_selected_channels(self, tiny_uvh5, kwargs, expected):
        """Each starting option and width keeps the expected channel indices."""
        *_, uvd = preprocess_uvdata(tiny_uvh5, return_uvd=True, **kwargs)
        np.testing.assert_allclose(np.ravel(uvd.freq_array), FREQS_HZ[expected])

    def test_centre_takes_precedence_over_lower_bound(self, tiny_uvh5):
        """``freq_center`` overrides ``freq_min`` and ``freq_idx_min``."""
        *_, uvd = preprocess_uvdata(
            tiny_uvh5, return_uvd=True, freq_center=150.3e6, n_freqs_select=2,
            freq_min=150.0e6, freq_idx_min=0,
        )
        np.testing.assert_allclose(np.ravel(uvd.freq_array), FREQS_HZ[[2, 3]])

    def test_lower_bound_takes_precedence_over_index(self, tiny_uvh5):
        """``freq_min`` overrides ``freq_idx_min``."""
        *_, uvd = preprocess_uvdata(tiny_uvh5, return_uvd=True, freq_min=150.3e6, freq_idx_min=0)
        np.testing.assert_allclose(np.ravel(uvd.freq_array), FREQS_HZ[[3]])

    def test_lower_bound_below_band_warns_and_keeps_all(self, tiny_uvh5):
        """A ``freq_min`` below the band warns and keeps every channel."""
        with pytest.warns(UserWarning, match="All frequencies"):
            *_, uvd = preprocess_uvdata(tiny_uvh5, return_uvd=True, freq_min=149e6)
        assert uvd.Nfreqs == 4

    def test_window_beyond_band_warns(self, tiny_uvh5):
        """A width extending past the band edge warns and is truncated."""
        with pytest.warns(UserWarning, match="fewer than n_freqs_select"):
            *_, uvd = preprocess_uvdata(
                tiny_uvh5, return_uvd=True, freq_idx_min=2, n_freqs_select=5
            )
        assert uvd.Nfreqs == 2

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(freq_center=150.1e6), "n_freqs_select"),
            (dict(freq_center=149.0e6, n_freqs_select=2), "minimum frequency"),
            (dict(freq_center=151.0e6, n_freqs_select=2), "maximum frequency"),
            (dict(freq_center=150.0e6, n_freqs_select=4), "Invalid combination"),
            (dict(freq_min=151.0e6), "maximum frequency"),
            (dict(freq_idx_min=-1), "positive"),
            (dict(freq_idx_min=4), "exceed"),
            (dict(freq_idx_min=0, n_freqs_select=0), "positive"),
        ],
    )
    def test_invalid_selections_raise(self, tiny_uvh5, kwargs, match):
        """Inconsistent or out-of-range frequency selections raise ValueError."""
        with pytest.raises(ValueError, match=match):
            preprocess_uvdata(tiny_uvh5, **kwargs)


class TestPreprocessUvdataTimeSelection:
    """Time-window selection."""

    @pytest.fixture(scope="class")
    @staticmethod
    def jds(tiny_uvh5):
        """Unique Julian dates of the fixture integrations."""
        return np.unique(_read(tiny_uvh5).time_array)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (dict(jd_idx_min=2), [2, 3]),
            (dict(jd_idx_min=1, n_times_select=2), [1, 2]),
            (dict(jd_min_index=1), [1, 2, 3]),
            (dict(jd_center_index=2, n_times_select=2), [1, 2]),
            (dict(n_times_select=2), [0, 1, 2, 3]),
        ],
    )
    def test_selected_integrations(self, tiny_uvh5, jds, kwargs, expected):
        """Each starting option and width keeps the expected integrations."""
        kwargs = dict(kwargs)
        if "jd_min_index" in kwargs:
            kwargs["jd_min"] = jds[kwargs.pop("jd_min_index")] - 1e-7
        if "jd_center_index" in kwargs:
            kwargs["jd_center"] = jds[kwargs.pop("jd_center_index")]
        *_, uvd = preprocess_uvdata(tiny_uvh5, return_uvd=True, **kwargs)
        np.testing.assert_allclose(np.unique(uvd.time_array), jds[expected])

    def test_centre_takes_precedence(self, tiny_uvh5, jds):
        """``jd_center`` overrides ``jd_min`` and ``jd_idx_min``."""
        *_, uvd = preprocess_uvdata(
            tiny_uvh5, return_uvd=True, jd_center=jds[3], n_times_select=2,
            jd_min=jds[0], jd_idx_min=0,
        )
        np.testing.assert_allclose(np.unique(uvd.time_array), jds[[2, 3]])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(jd_center_offset=0.0), "n_times_select"),
            (dict(jd_center_offset=-1.0, n_times_select=2), "minimum time"),
            (dict(jd_center_offset=1.0, n_times_select=2), "maximum time"),
            (dict(jd_center_offset=0.0, n_times_select=4), "Invalid combination"),
            (dict(jd_min_offset=1.0), "maximum time"),
            (dict(jd_idx_min=-1), "positive"),
            (dict(jd_idx_min=4), "exceed"),
            (dict(jd_idx_min=0, n_times_select=0), "positive"),
        ],
    )
    def test_invalid_selections_raise(self, tiny_uvh5, jds, kwargs, match):
        """Inconsistent or out-of-range time selections raise ValueError."""
        kwargs = dict(kwargs)
        if "jd_center_offset" in kwargs:
            kwargs["jd_center"] = jds[0] + kwargs.pop("jd_center_offset")
        if "jd_min_offset" in kwargs:
            kwargs["jd_min"] = jds[-1] + kwargs.pop("jd_min_offset")
        with pytest.raises(ValueError, match=match):
            preprocess_uvdata(tiny_uvh5, **kwargs)

    def test_lower_bound_before_data_warns(self, tiny_uvh5, jds):
        """A ``jd_min`` before the observation warns and keeps every time."""
        with pytest.warns(UserWarning, match="All times"):
            *_, uvd = preprocess_uvdata(tiny_uvh5, return_uvd=True, jd_min=jds[0] - 1.0)
        assert uvd.Ntimes == 4


class TestPreprocessUvdataNoiseAndSaving:
    """Noise estimation and saved products."""

    def test_calc_noise_returns_matching_vector(self, tiny_uvh5):
        """``calc_noise`` returns a noise vector of the same length as the data."""
        vis, _, _, _, noise = preprocess_uvdata(tiny_uvh5, calc_noise=True, **ODD_TIMES)
        assert noise.shape == vis.shape
        assert np.any(noise != 0)

    def test_saved_products_round_trip(self, tiny_uvh5, tmp_path):
        """Saved vectors and instrument arrays reload to the returned values."""
        from bayeseor_jax.io import load_numpy_dict

        vis, antpairs, uvws, red, noise = preprocess_uvdata(
            tiny_uvh5, calc_noise=True, save_vis=True, save_model=True, save_dir=tmp_path,
            **ODD_TIMES,
        )
        np.testing.assert_array_equal(load_numpy_dict(tmp_path / "vis_vector.npy"), vis)
        np.testing.assert_array_equal(load_numpy_dict(tmp_path / "noise_vector.npy"), noise)
        np.testing.assert_array_equal(load_numpy_dict(tmp_path / "uvw_model.npy"), uvws)
        np.testing.assert_array_equal(load_numpy_dict(tmp_path / "redundancy_model.npy"), red)
        assert [tuple(ap) for ap in load_numpy_dict(tmp_path / "antpairs.npy")] == antpairs

    def test_existing_products_require_clobber(self, tiny_uvh5, tmp_path):
        """Saving over existing products without ``clobber`` raises ValueError."""
        preprocess_uvdata(tiny_uvh5, save_vis=True, save_dir=tmp_path)
        with pytest.raises(ValueError, match="clobber"):
            preprocess_uvdata(tiny_uvh5, save_vis=True, save_dir=tmp_path)
        preprocess_uvdata(tiny_uvh5, save_vis=True, save_dir=tmp_path, clobber=True)

    def test_save_requires_directory(self, tiny_uvh5):
        """Requesting saved products with ``save_dir=None`` raises ValueError."""
        with pytest.raises(ValueError, match="save_dir"):
            preprocess_uvdata(tiny_uvh5, save_model=True, save_dir=None)


# ---------------------------------------------------------------------------
# get_vis_data
# ---------------------------------------------------------------------------

class TestGetVisDataRaw:
    """Loading a visibility file."""

    def test_matches_preprocess_and_metadata(self, tiny_uvh5, raw_uvd):
        """The raw route returns the preprocessed vector with its frequency/time axes."""
        out = get_vis_data(data_path=tiny_uvh5, sigma=1.0)
        vis, *_ = preprocess_uvdata(tiny_uvh5)
        np.testing.assert_allclose(out["vis"], vis)
        np.testing.assert_allclose(out["freqs_hz"], FREQS_HZ)
        np.testing.assert_allclose(out["jds"], np.unique(raw_uvd.time_array))
        assert out["df_hz"] == pytest.approx(100e3)
        assert out["dt"] == pytest.approx(10.0, abs=1e-4)
        assert out["uvd"] is not None
        assert out["redundancy"].shape == (4, 12, 1)

    def test_single_channel_and_time_use_configured_spacing(self, tiny_uvh5):
        """With one channel and one time, ``df_mhz`` and ``d_t`` supply the spacings."""
        out = get_vis_data(
            data_path=tiny_uvh5, sigma=1.0, freq_idx_min=0, n_freqs_select=1,
            jd_idx_min=0, n_times_select=1, df_mhz=0.25, d_t=7.0,
        )
        assert out["df_hz"] == pytest.approx(0.25e6)
        assert out["dt"] == pytest.approx(7.0)

    def test_single_time_without_cadence_is_zero(self, tiny_uvh5):
        """With one time and no ``d_t``, the returned cadence is zero."""
        out = get_vis_data(data_path=tiny_uvh5, sigma=1.0, jd_idx_min=0, n_times_select=1)
        assert out["dt"] == 0.0


class TestGetVisDataNpy:
    """Loading a preprocessed vector."""

    def test_loads_vector_and_instrument_model(self, tiny_npy_dir):
        """The data and baselines come from the ``.npy`` files unchanged."""
        from bayeseor_jax.io import load_numpy_dict

        out = get_vis_data(
            data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            sigma=1.0, jd_min=2459999.1, **NPY_AXES,
        )
        np.testing.assert_allclose(out["vis"], load_numpy_dict(tiny_npy_dir / "vis_vector.npy"))
        np.testing.assert_allclose(out["uvws"], load_numpy_dict(tiny_npy_dir / "uvw_model.npy"))
        assert out["redundancy"].shape == (4, 12)
        assert out["uvd"] is None

    def test_frequency_axis_from_configuration(self, tiny_npy_dir):
        """Frequencies are f0 + n*df in MHz, converted to Hz."""
        out = get_vis_data(
            data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            sigma=1.0, jd_min=2459999.1, **NPY_AXES,
        )
        np.testing.assert_allclose(out["freqs_hz"], FREQS_HZ)
        assert out["df_hz"] == pytest.approx(1e5)

    def test_time_axis_from_lower_bound(self, tiny_npy_dir):
        """``jd_min`` places the first of n_t integrations spaced by d_t."""
        out = get_vis_data(
            data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            sigma=1.0, jd_min=2459999.1, **NPY_AXES,
        )
        np.testing.assert_allclose(out["jds"], 2459999.1 + 10.0 / 86400 * np.arange(4), rtol=0, atol=1e-9)
        assert out["dt"] == 10.0

    def test_time_axis_centred(self, tiny_npy_dir):
        """``jd_center`` sits at index n_t // 2 of the time axis."""
        out = get_vis_data(
            data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            sigma=1.0, jd_center=2459999.2, **NPY_AXES,
        )
        np.testing.assert_allclose(out["jds"], 2459999.2 + 10.0 / 86400 * np.arange(-2, 2), rtol=0, atol=1e-9)

    def test_lower_bound_takes_precedence_over_centre(self, tiny_npy_dir):
        """With both anchors set, ``jd_min`` defines the time axis."""
        out = get_vis_data(
            data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            sigma=1.0, jd_min=2459999.1, jd_center=2459999.9, **NPY_AXES,
        )
        assert out["jds"][0] == pytest.approx(2459999.1)

    def test_requires_instrument_model(self, tiny_npy_dir):
        """A ``.npy`` input without ``inst_model`` raises ValueError."""
        with pytest.raises(ValueError, match="inst_model"):
            get_vis_data(data_path=tiny_npy_dir / "vis_vector.npy", sigma=1.0, jd_min=1.0, **NPY_AXES)

    @pytest.mark.parametrize("missing", ["n_freq", "f0_mhz", "df_mhz", "n_t", "d_t"])
    def test_requires_axes(self, tiny_npy_dir, missing):
        """Each frequency and time setting is required for ``.npy`` inputs."""
        axes = {**NPY_AXES, missing: None}
        with pytest.raises(ValueError, match="required"):
            get_vis_data(
                data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
                sigma=1.0, jd_min=1.0, **axes,
            )

    def test_requires_time_anchor(self, tiny_npy_dir):
        """A ``.npy`` input without ``jd_min`` or ``jd_center`` raises ValueError."""
        with pytest.raises(ValueError, match="jd_min or jd_center"):
            get_vis_data(
                data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
                sigma=1.0, **NPY_AXES,
            )


class TestGetVisDataNoise:
    """Choice between supplied, estimated and synthetic noise."""

    def _npy(self, tiny_npy_dir, **kwargs):
        """Load the ``.npy`` fixture with the given noise options."""
        return get_vis_data(
            data_path=tiny_npy_dir / "vis_vector.npy", inst_model=tiny_npy_dir,
            jd_min=2459999.1, **NPY_AXES, **kwargs,
        )

    def test_supplied_noise_is_returned_not_added(self, tiny_npy_dir, tmp_path, rng):
        """``noise_data_path`` leaves the data unchanged and returns that vector."""
        noise = rng.normal(size=192) + 1j * rng.normal(size=192)
        np.save(tmp_path / "noise.npy", noise)
        out = self._npy(tiny_npy_dir, noise_data_path=tmp_path / "noise.npy", sigma=5.0)
        np.testing.assert_array_equal(out["vis_noisy"], out["vis"])
        np.testing.assert_allclose(out["noise"], noise)
        assert out["bl_conj_pairs_map"] == {}

    def test_estimated_noise_is_returned_not_added(self, tiny_uvh5):
        """``calc_noise`` on a raw file leaves the data unchanged."""
        out = get_vis_data(data_path=tiny_uvh5, calc_noise=True, **ODD_TIMES)
        _, _, _, _, noise = preprocess_uvdata(tiny_uvh5, calc_noise=True, **ODD_TIMES)
        np.testing.assert_array_equal(out["vis_noisy"], out["vis"])
        np.testing.assert_allclose(out["noise"], noise)

    def test_supplied_noise_takes_precedence_over_estimate(self, tiny_uvh5, tmp_path):
        """A supplied noise vector overrides ``calc_noise``."""
        np.save(tmp_path / "noise.npy", np.full(144, 2.0 + 0j))
        out = get_vis_data(
            data_path=tiny_uvh5, calc_noise=True, noise_data_path=tmp_path / "noise.npy", **ODD_TIMES
        )
        np.testing.assert_allclose(out["noise"], 2.0)

    def test_synthetic_noise_is_added(self, tiny_npy_dir):
        """Without supplied or estimated noise, a seeded realisation is added."""
        out = self._npy(tiny_npy_dir, sigma=0.5, noise_seed=11)
        expected, noise, _ = generate_gaussian_noise(
            0.5, out["vis"], 4, 4, out["uvws"], out["redundancy"], random_seed=11
        )
        np.testing.assert_allclose(out["noise"], noise)
        np.testing.assert_allclose(out["vis_noisy"], out["vis"] + out["noise"])
        assert out["bl_conj_pairs_map"] == {b: b + 6 for b in range(6)}

    def test_synthetic_noise_requires_sigma(self, tiny_npy_dir):
        """Adding synthetic noise without ``sigma`` raises ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            self._npy(tiny_npy_dir)


class TestGetVisDataPaths:
    """Input path validation."""

    def test_missing_path_raises(self, tmp_path):
        """A path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_vis_data(data_path=tmp_path / "missing.uvh5")

    def test_unsupported_suffix_raises(self, tmp_path):
        """An unrecognised suffix raises ValueError."""
        path = tmp_path / "data.txt"
        path.write_text("0")
        with pytest.raises(ValueError, match="Unsupported"):
            get_vis_data(data_path=path)

    def test_string_path_is_accepted(self, tiny_uvh5):
        """A string path behaves like a ``Path``."""
        out = get_vis_data(data_path=str(tiny_uvh5), sigma=1.0)
        assert out["vis"].shape == (192,)
