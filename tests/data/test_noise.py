"""Tests for ``bayeseor_jax.data.noise``: synthetic thermal-noise realisations."""

from __future__ import annotations

import numpy as np
import pytest

from bayeseor_jax.data.noise import (
    _normalise_redundancy,
    _normalise_uvw_for_pairing,
    generate_gaussian_noise,
)

N_T, N_F = 3, 2


def _uvws(n_t=N_T):
    """Three baselines followed by their negated copies, constant in time."""
    half = np.array([[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [5.0, 5.0, 0.0]])
    full = np.concatenate([half, -half])
    return np.broadcast_to(full, (n_t, 6, 3)).copy()


def _call(sigma=1.0, seed=7, redundancy=None, signal=None, uvws=None):
    """Run ``generate_gaussian_noise`` on the six-baseline layout."""
    uvws = _uvws() if uvws is None else uvws
    n_bls = uvws.shape[-2]
    redundancy = np.ones((N_T, n_bls)) if redundancy is None else redundancy
    signal = np.zeros(N_T * N_F * n_bls, dtype=complex) if signal is None else signal
    return generate_gaussian_noise(sigma, signal, N_F, N_T, uvws, redundancy, random_seed=seed)


class TestGenerateGaussianNoise:
    """Hermitian, redundancy-weighted Gaussian noise."""

    def test_output_is_signal_plus_noise(self, rng):
        """The noisy vector is the input signal plus the returned noise."""
        signal = rng.normal(size=36) + 1j * rng.normal(size=36)
        d, noise, _ = _call(signal=signal)
        np.testing.assert_allclose(d, signal + noise)

    def test_conjugate_pairs_map(self):
        """Each baseline is paired with its negated copy."""
        _, _, pairs = _call()
        assert pairs == {0: 3, 1: 4, 2: 5}

    def test_conjugate_baselines_receive_conjugate_noise(self):
        """Within every time/frequency block, the second half is the conjugate of the first."""
        _, noise, _ = _call()
        blocks = np.asarray(noise).reshape(N_T * N_F, 6)
        np.testing.assert_allclose(blocks[:, 3:], blocks[:, :3].conj())

    def test_matches_legacy_seeding(self):
        """Real and imaginary parts use seeds s and 123*s, then rescale to RMS sigma."""
        sigma, seed, n = 0.3, 5, 36
        np.random.seed(seed)
        real = np.random.normal(0.0, sigma / np.sqrt(2), n)
        np.random.seed(seed * 123)
        imag = np.random.normal(0.0, sigma / np.sqrt(2), n)
        raw = real + 1j * imag
        raw = raw * (sigma / raw.std())
        blocks = raw.reshape(N_T * N_F, 6)
        blocks[:, 3:] = blocks[:, :3].conj()
        _, noise, _ = _call(sigma=sigma, seed=seed)
        np.testing.assert_allclose(noise, blocks.reshape(-1))

    def test_redundancy_divides_by_square_root(self):
        """Noise on a baseline of redundancy R is scaled by 1/sqrt(R)."""
        red = np.ones((N_T, 6))
        red[:, [1, 4]] = 4.0
        _, unit, _ = _call()
        _, scaled, _ = _call(redundancy=red)
        ratio = np.asarray(scaled).reshape(N_T * N_F, 6) / np.asarray(unit).reshape(N_T * N_F, 6)
        np.testing.assert_allclose(ratio[:, [1, 4]], 0.5)
        np.testing.assert_allclose(ratio[:, [0, 2, 3, 5]], 1.0)

    def test_same_seed_is_reproducible(self):
        """Two calls with the same non-zero seed give identical noise."""
        np.testing.assert_array_equal(_call(seed=3)[1], _call(seed=3)[1])

    def test_different_seeds_differ(self):
        """Different seeds give different realisations."""
        assert not np.allclose(_call(seed=3)[1], _call(seed=4)[1])

    def test_seed_zero_is_reproducible(self):
        """A seed of zero is honoured like any other seed."""
        np.random.seed(1)
        first = np.asarray(_call(seed=0)[1])
        np.random.seed(2)
        second = np.asarray(_call(seed=0)[1])
        np.testing.assert_array_equal(first, second)

    def test_seed_zero_real_and_imaginary_parts_differ(self):
        """With seed 0 the real and imaginary draws still come from different streams."""
        noise = np.asarray(_call(seed=0)[1])
        assert not np.allclose(noise.real, noise.imag)

    def test_zero_sigma_gives_zero_noise(self):
        """``sigma=0`` returns an all-zero noise vector."""
        _, noise, _ = _call(sigma=0.0)
        np.testing.assert_array_equal(noise, 0.0)

    def test_accepts_two_dimensional_baselines(self):
        """A single (n_bls, 3) baseline list gives the same result as a time stack."""
        np.testing.assert_allclose(_call(uvws=_uvws()[0])[1], _call()[1])

    def test_wrong_signal_length_raises(self):
        """A signal not of length n_t*n_freq*n_bls raises ValueError."""
        with pytest.raises(ValueError, match="expected"):
            _call(signal=np.zeros(10, dtype=complex))


class TestNormaliseUvwForPairing:
    """Baseline-list normalisation for conjugate pairing."""

    def test_two_dimensional_passthrough(self):
        """An (n_bls, 3) array is returned unchanged."""
        uvw = _uvws()[0]
        np.testing.assert_array_equal(_normalise_uvw_for_pairing(uvw), uvw)

    def test_time_stack_uses_first_integration(self):
        """An (n_t, n_bls, 3) array returns its first integration."""
        uvw = _uvws()
        uvw[1:] += 1.0
        np.testing.assert_array_equal(_normalise_uvw_for_pairing(uvw), uvw[0])

    @pytest.mark.parametrize("shape", [(6, 2), (2, 6, 2), (6,), (1, 2, 6, 3)])
    def test_invalid_shapes_raise(self, shape):
        """Arrays without a trailing axis of length three raise ValueError."""
        with pytest.raises(ValueError):
            _normalise_uvw_for_pairing(np.zeros(shape))


class TestNormaliseRedundancy:
    """Redundancy-array normalisation to (n_t, n_bls)."""

    @pytest.mark.parametrize(
        "array",
        [
            np.arange(1.0, 5.0),
            np.arange(1.0, 5.0)[:, None],
            np.broadcast_to(np.arange(1.0, 5.0), (3, 4)),
            np.broadcast_to(np.arange(1.0, 5.0), (3, 4))[..., None],
        ],
        ids=["vector", "column", "time-baseline", "trailing-singleton"],
    )
    def test_accepted_shapes_broadcast_to_time_baseline(self, array):
        """Every accepted layout becomes the same (n_t, n_bls) array."""
        out = _normalise_redundancy(array, n_t=3, n_bls=4)
        np.testing.assert_array_equal(out, np.broadcast_to(np.arange(1.0, 5.0), (3, 4)))

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (4, 3), (3, 4, 2), (1, 3, 4, 1)])
    def test_incompatible_shapes_raise(self, shape):
        """Arrays inconsistent with n_t=3, n_bls=4 raise ValueError."""
        with pytest.raises(ValueError):
            _normalise_redundancy(np.ones(shape), n_t=3, n_bls=4)
