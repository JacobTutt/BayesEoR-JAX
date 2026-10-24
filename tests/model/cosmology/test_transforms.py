"""Tests for ``bayeseor_jax.model.cosmology.transforms``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from astropy import units
from astropy.constants import c
from astropy.cosmology import Planck18

from bayeseor_jax.model.cosmology.transforms import Cosmology

F_21_HZ = 1.42040575177e9


@pytest.fixture(scope="module")
def cosmo() -> Cosmology:
    """Default Planck18 cosmology helper."""
    return Cosmology()


class TestDefaults:
    def test_parameters_match_planck18(self, cosmo):
        """Default parameters are those of astropy's Planck18 cosmology."""
        assert cosmo.H0_km_s_Mpc == pytest.approx(Planck18.H0.value)
        assert cosmo.Om0 == pytest.approx(Planck18.Om0)
        assert cosmo.Ode0 == pytest.approx(Planck18.Ode0)
        assert cosmo.Ok0 == pytest.approx(Planck18.Ok0)
        assert cosmo.Or0 == pytest.approx(Planck18.Ogamma0 + Planck18.Onu0)

    def test_constants(self, cosmo):
        """The 21-cm rest frequency and speed of light take their standard values."""
        assert cosmo.f_21_hz == pytest.approx(F_21_HZ)
        assert cosmo.c_ms == pytest.approx(c.to_value("m/s"))

    def test_is_frozen(self, cosmo):
        """Cosmology instances cannot be modified after construction."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            cosmo.H0_km_s_Mpc = 70.0


class TestFrequencyRedshift:
    def test_f2z_formula(self, cosmo):
        """1 + z = f_21 / f."""
        f = np.array([50e6, 150e6, 200e6])
        np.testing.assert_allclose(cosmo.f2z(f), F_21_HZ / f - 1.0, rtol=1e-14)

    def test_rest_frequency_is_redshift_zero(self, cosmo):
        """The 21-cm rest frequency maps to z = 0."""
        assert float(cosmo.f2z(F_21_HZ)) == pytest.approx(0.0, abs=1e-14)

    @pytest.mark.parametrize("z", [0.0, 6.0, 8.47, 20.0])
    def test_round_trip(self, cosmo, z):
        """z2f inverts f2z."""
        assert float(cosmo.f2z(cosmo.z2f(z))) == pytest.approx(z, rel=1e-13, abs=1e-13)

    def test_z2f_formula(self, cosmo):
        """f = f_21 / (1 + z)."""
        z = np.array([0.0, 7.0, 9.0])
        np.testing.assert_allclose(cosmo.z2f(z), F_21_HZ / (1.0 + z), rtol=1e-14)


class TestExpansionAndDistances:
    @pytest.mark.parametrize("z", [0.0, 1.0, 8.5])
    def test_e_z_matches_planck18(self, cosmo, z):
        """E(z) equals astropy Planck18.efunc."""
        assert float(cosmo.e_z(z)) == pytest.approx(Planck18.efunc(z), rel=1e-12)

    def test_e_z_unity_today(self, cosmo):
        """E(0) = 1."""
        assert float(cosmo.e_z(0.0)) == pytest.approx(1.0, rel=1e-12)

    def test_e_z_preserves_shape(self, cosmo):
        """E(z) has the shape of its input."""
        assert cosmo.e_z(np.zeros((2, 3))).shape == (2, 3)

    @pytest.mark.parametrize("z", [6.0, 8.5, 12.0])
    def test_dl_df_matches_astropy_derivative(self, cosmo, z):
        """|dχ/df| equals the numerical frequency derivative of astropy's comoving distance."""
        f = F_21_HZ / (1.0 + z)
        h = f * 1e-6
        chi = lambda freq: Planck18.comoving_distance(F_21_HZ / freq - 1.0).to_value(units.Mpc)
        numerical = abs(chi(f + h) - chi(f - h)) / (2 * h)
        assert float(cosmo.dL_df(z)) == pytest.approx(numerical, rel=1e-6)

    def test_dl_df_explicit_formula(self, cosmo):
        """|dχ/df| = (c/H0)(1+z)²/(f_21 E(z))."""
        z = 8.0
        expected = (c.to_value("km/s") / Planck18.H0.value) * (1 + z) ** 2 / (F_21_HZ * Planck18.efunc(z))
        assert float(cosmo.dL_df(z)) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("z", [0.5, 8.5])
    def test_dl_dth_matches_astropy(self, cosmo, z):
        """D_M equals astropy's transverse comoving distance in Mpc."""
        assert float(cosmo.dL_dth(z)) == pytest.approx(Planck18.comoving_transverse_distance(z).to_value(units.Mpc), rel=1e-12)

    def test_dl_dth_scalar_and_array(self, cosmo):
        """D_M returns a scalar for scalar z and an array for array z."""
        assert np.ndim(cosmo.dL_dth(8.0)) == 0
        assert cosmo.dL_dth(np.array([7.0, 8.0])).shape == (2,)

    def test_dl_dth_ignores_sample_count(self, cosmo):
        """The legacy n_z_samples argument does not change the result."""
        assert float(cosmo.dL_dth(8.0, n_z_samples=16)) == float(cosmo.dL_dth(8.0))


class TestInstToCosmoVol:
    @pytest.mark.parametrize("z", [6.0, 8.5])
    def test_is_dm_squared_times_dl_df(self, cosmo, z):
        """dV/(dΩ df) = D_M² |dχ/df|."""
        expected = float(cosmo.dL_dth(z)) ** 2 * float(cosmo.dL_df(z))
        assert float(cosmo.inst_to_cosmo_vol(z)) == pytest.approx(expected, rel=1e-13)

    def test_matches_astropy_differential_volume(self, cosmo):
        """D_M² |dχ/df| equals astropy's dV/dz/dΩ times |dz/df|."""
        z = 8.0
        f = F_21_HZ / (1 + z)
        dz_df = F_21_HZ / f**2
        expected = Planck18.differential_comoving_volume(z).to_value(units.Mpc**3 / units.sr) * dz_df
        assert float(cosmo.inst_to_cosmo_vol(z)) == pytest.approx(expected, rel=1e-6)

    def test_positive(self, cosmo):
        """The volume conversion is positive at EoR redshifts."""
        assert np.all(np.asarray(cosmo.inst_to_cosmo_vol(np.array([6.0, 10.0]))) > 0)
