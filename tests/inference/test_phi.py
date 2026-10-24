"""Tests for ``bayeseor_jax.inference.phi``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from bayeseor_jax.inference.phi import (
    PhiInverse,
    build_kbin_index_per_voxel,
    compute_power_norm_per_bin,
)

BOX = dict(ps_box_size_ra_mpc=120.0, ps_box_size_dec_mpc=90.0, ps_box_size_para_mpc=60.0)
INST_TO_COSMO = 3.5
K_VALS = np.array([0.2, 0.5])
BIN_ID = np.array([0, 1, 1, 0, 1])
VALID = np.array([True, True, False, True, True])
FG_STD = np.array([2.0, 0.5, 4.0])


def _norm(k):
    """Independent power normalisation ``(k^3/2pi^2) / V * c^2``."""
    volume = BOX["ps_box_size_ra_mpc"] * BOX["ps_box_size_dec_mpc"] * BOX["ps_box_size_para_mpc"]
    return k**3 / (2 * np.pi**2) / volume * INST_TO_COSMO**2


@pytest.fixture
def phi() -> PhiInverse:
    """Five EoR voxels in two bins, one invalid, plus three foreground columns."""
    return PhiInverse(
        bin_id_per_k_mode=BIN_ID, valid_mask=VALID, k_vals=K_VALS,
        inst_to_cosmo_vol=INST_TO_COSMO, n_par=len(BIN_ID) + len(FG_STD),
        foreground_prior_std=FG_STD, **BOX,
    )


class TestComputePowerNormPerBin:
    def test_matches_formula(self):
        """The normalisation is ``(k^3 / 2 pi^2) / V_box * c_vol^2``."""
        result = compute_power_norm_per_bin(k_vals=jnp.asarray(K_VALS), inst_to_cosmo_vol=jnp.asarray(INST_TO_COSMO), **BOX)
        np.testing.assert_allclose(result, _norm(K_VALS))


class TestBuildKbinIndexPerVoxel:
    def test_invalid_voxels_are_minus_one(self):
        """Valid voxels keep their bin and invalid ones are flagged with ``-1``."""
        np.testing.assert_array_equal(build_kbin_index_per_voxel(BIN_ID, VALID), [0, 1, -1, 0, 1])

    def test_returns_int32(self):
        """The index map is an ``int32`` array."""
        assert build_kbin_index_per_voxel(BIN_ID, VALID).dtype == jnp.int32


class TestPhiInverse:
    def test_eor_block_is_norm_over_power(self, phi):
        """Each valid EoR voxel has precision ``norm_s / phi_s`` for its bin."""
        power = np.array([3.0, 40.0])
        result = np.asarray(phi.evaluate(jnp.asarray(power)))
        expected_eor = np.where(VALID, (_norm(K_VALS) / power)[BIN_ID], 0.0)
        np.testing.assert_allclose(result[:5], expected_eor)

    def test_invalid_voxels_have_zero_precision(self, phi):
        """Voxels outside the binned range receive zero prior precision."""
        assert float(phi.evaluate(jnp.array([1.0, 1.0]))[2]) == 0.0

    def test_foreground_block_is_inverse_variance(self, phi):
        """Foreground columns have precision ``1 / std^2``, independent of power."""
        for power in ([1.0, 1.0], [1e3, 1e-3]):
            np.testing.assert_allclose(phi.evaluate(jnp.asarray(power))[5:], 1.0 / FG_STD**2)

    def test_output_length_is_n_par(self, phi):
        """The precision diagonal covers every coefficient column."""
        assert phi.evaluate(jnp.array([1.0, 2.0])).shape == (8,)

    def test_zero_power_stays_finite(self, phi):
        """Zero power is clamped to the smallest positive float, avoiding division by zero."""
        assert np.all(np.isfinite(np.asarray(phi.evaluate(jnp.array([0.0, 1.0])))))

    def test_foreground_only(self):
        """With no EoR voxels the precision is just the foreground block."""
        phi = PhiInverse(
            bin_id_per_k_mode=np.empty(0, int), valid_mask=np.empty(0, bool), k_vals=np.empty(0),
            inst_to_cosmo_vol=1.0, n_par=3, foreground_prior_std=FG_STD, **BOX,
        )
        np.testing.assert_allclose(phi.evaluate(jnp.empty(0)), 1.0 / FG_STD**2)

    def test_empty_foreground_block(self):
        """An EoR-only model accepts an empty foreground width vector."""
        phi = PhiInverse(
            bin_id_per_k_mode=BIN_ID, valid_mask=VALID, k_vals=K_VALS, inst_to_cosmo_vol=INST_TO_COSMO,
            n_par=len(BIN_ID), foreground_prior_std=np.empty(0), **BOX,
        )
        assert phi.evaluate(jnp.array([1.0, 1.0])).shape == (5,)

    def test_rejects_wrong_foreground_length(self):
        """The number of foreground widths must equal ``n_par`` minus the EoR voxels."""
        with pytest.raises(ValueError, match="length mismatch"):
            PhiInverse(
                bin_id_per_k_mode=BIN_ID, valid_mask=VALID, k_vals=K_VALS, inst_to_cosmo_vol=1.0,
                n_par=9, foreground_prior_std=FG_STD, **BOX,
            )

    @pytest.mark.parametrize(
        "widths",
        [np.array([1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0]), np.array([1.0, np.inf, 1.0]),
         np.array([1.0, 1.0, 1.0]) + 0j],
    )
    def test_rejects_invalid_foreground_widths(self, widths):
        """Foreground widths must be finite, real and positive."""
        with pytest.raises(ValueError, match="finite, real and positive"):
            PhiInverse(
                bin_id_per_k_mode=BIN_ID, valid_mask=VALID, k_vals=K_VALS, inst_to_cosmo_vol=1.0,
                n_par=8, foreground_prior_std=widths, **BOX,
            )
