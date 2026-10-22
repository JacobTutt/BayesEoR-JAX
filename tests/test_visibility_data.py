"""Check the data-package API and documented visibility-loading conventions."""

from types import SimpleNamespace

import numpy as np
import pytest

from bayeseor_jax.data import vis
from bayeseor_jax.io import save_numpy_dict


def test_data_exports_preserve_top_level_visibility_helpers():
    import bayeseor_jax as api
    from bayeseor_jax import data
    from bayeseor_jax.data.noise import generate_gaussian_noise

    for name in ("get_vis_data", "preprocess_uvdata", "uvd_to_vector", "form_pI_vis", "jy_to_ksr"):
        assert getattr(api, name) is getattr(data, name) is getattr(vis, name)
    assert data.generate_gaussian_noise is generate_gaussian_noise


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("external_noise", [False, True])
def test_preprocessed_numpy_formats(tmp_path, wrapped, external_noise):
    signal = np.array([1+2j, 1-2j, 3+4j, 3-4j])
    data_path = tmp_path / "vis.npy"
    if wrapped:
        save_numpy_dict(data_path, signal, args={})
    else:
        np.save(data_path, signal)
    np.save(tmp_path / "uvw_model.npy", [[[1., 2., 0.], [-1., -2., 0.]]])
    np.save(tmp_path / "redundancy_model.npy", [[1., 1.]])
    noise_path = tmp_path / "noise.npy" if external_noise else None
    if external_noise:
        np.save(noise_path, np.ones(4))
    result = vis.get_vis_data(
        data_path=data_path, inst_model=tmp_path, n_freq=2, f0_mhz=150., df_mhz=1.,
        n_t=1, d_t=10., jd_min=2459000., sigma=0., noise_data_path=noise_path,
    )
    np.testing.assert_array_equal(result["vis"], signal)
    np.testing.assert_array_equal(result["vis_noisy"], signal)
    np.testing.assert_array_equal(result["noise"], np.ones(4) if external_noise else np.zeros(4))
    np.testing.assert_array_equal(result["freqs_hz"], [150e6, 151e6])
    assert result["df_hz"] == 1e6
    assert result["dt"] == 10.
    assert result["uvd"] is None
    assert result["antpairs"] is None
    assert result["uvws"].shape == (1, 2, 3)


@pytest.mark.parametrize("suffix", [".uvh5", ".uvfits", ".ms"])
def test_raw_suffixes_dispatch_to_preprocessing(tmp_path, monkeypatch, suffix):
    path = tmp_path / ("observation" + suffix)
    path.mkdir() if suffix == ".ms" else path.touch()
    signal = np.array([1+2j, 1-2j, 3+4j, 3-4j])
    uvd = SimpleNamespace(freq_array=np.array([150e6, 151e6]), time_array=np.array([2459000.]))
    calls = []

    def preprocess(**kwargs):
        calls.append(kwargs)
        return signal, [(1, 2), (2, 1)], np.zeros((1, 2, 3)), np.ones((1, 2, 1)), np.ones(4), uvd

    monkeypatch.setattr(vis, "preprocess_uvdata", preprocess)
    result = vis.get_vis_data(data_path=path, calc_noise=True)
    assert calls[0]["filepath"] == path
    assert calls[0]["return_uvd"] is True
    assert result["uvd"] is uvd
    np.testing.assert_array_equal(result["vis_noisy"], signal)


def test_vector_order_and_conjugate_copies():
    pairs = [(1, 2), (1, 3)]
    blocks = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]],
                       [[5+5j, 6+6j], [7+7j, 8+8j]]])
    uvd = SimpleNamespace(Npols=1, Nblts=4, Nbls=2, Nfreqs=2,
                          freq_array=np.array([150e6, 151e6]),
                          get_data=lambda key: blocks[pairs.index(key[:2])])
    vector, noise = vis.uvd_to_vector(uvd, pairs)
    physical = blocks.transpose(1, 2, 0)
    expected = np.concatenate([physical, physical.conj()], axis=2).reshape(-1)
    np.testing.assert_array_equal(vector, expected)
    assert noise is None
