"""Check physical helpers and execution utilities through their new module paths."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import Planck18

from bayeseor_jax.model.construction.parallel_layout import padded_visibility_partition
from bayeseor_jax.model.cosmology.transforms import Cosmology
from bayeseor_jax.model.instrument.baselines import InstrumentModel
from bayeseor_jax.data.noise import generate_gaussian_noise


@pytest.mark.parametrize("combined", [False, True])
def test_instrument_loader_preserves_axes_and_metadata(tmp_path, combined):
    uvw = np.array([[[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0]]])
    redundancy = np.array([[[4], [4]]])
    pairs = np.array([[0, 1], [1, 0]])
    if combined:
        np.save(tmp_path / "instrument_model.npy", {
            "uvw_model": uvw, "redundancy_model": redundancy, "antpairs": pairs,
        })
    else:
        np.save(tmp_path / "uvw_model.npy", uvw)
        np.save(tmp_path / "redundancy_model.npy", redundancy)
        np.save(tmp_path / "antpairs.npy", pairs)

    instrument = InstrumentModel(tmp_path)
    np.testing.assert_array_equal(instrument.uvw_array_m, uvw)
    np.testing.assert_array_equal(instrument.bl_red_array, redundancy[..., 0])
    np.testing.assert_array_equal(instrument.antpairs, pairs)
    assert (instrument.n_t, instrument.n_bls) == (1, 2)


def test_noise_preserves_conjugate_baselines_and_seed():
    uvw = jnp.array([[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0]])
    signal = jnp.zeros(4, dtype=jnp.complex128)
    args = dict(sigma=1.0, s=signal, n_freq=2, n_t=1, uvw_array_m=uvw,
                bl_red_array=jnp.ones(2), random_seed=17)
    data, noise, pairs = generate_gaussian_noise(**args)
    np.testing.assert_array_equal(data, noise)
    np.testing.assert_array_equal(noise, generate_gaussian_noise(**args)[1])
    np.testing.assert_array_equal(noise[1::2], noise[::2].conj())
    assert pairs == {0: 1}


def test_cosmology_conversions_match_fixed_planck_model():
    cosmology = Cosmology()
    frequencies = jnp.array([150e6, 160e6])
    redshifts = cosmology.f2z(frequencies)
    np.testing.assert_allclose(cosmology.z2f(redshifts), frequencies)
    np.testing.assert_allclose(cosmology.e_z(redshifts), Planck18.efunc(redshifts))
    np.testing.assert_allclose(
        cosmology.dL_dth(redshifts),
        Planck18.comoving_transverse_distance(redshifts).value,
    )


@pytest.mark.parametrize("n_vis,n_hosts,n_devices", [(11, 3, 2), (2, 4, 2), (12, 2, 3)])
def test_visibility_partitions_cover_data_once(n_vis, n_hosts, n_devices):
    rows = []
    for host in range(n_hosts):
        start, stop, padded = padded_visibility_partition(n_vis, n_hosts, n_devices, host)
        rows.extend(range(start, stop))
        assert padded % n_devices == 0
        assert max(0, stop - start) <= padded
    assert rows == list(range(n_vis))
