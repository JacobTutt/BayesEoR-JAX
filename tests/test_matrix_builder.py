from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from bayeseor_jax.build_matricies.overall import BuildMatrices


def _minimal_builder(tmp_path):
    builder = BuildMatrices.__new__(BuildMatrices)
    builder.n_t = 2
    builder.n_freq = 2
    builder.nbls = 1
    builder.beta_fg = [2.0, 3.0]
    builder.n_fg_components = 3
    builder.f0_mhz = 100.0
    builder.df_mhz = 10.0
    builder.freq_hz = jnp.asarray([100.0e6, 110.0e6])
    builder.n_fg_spatial_modes = 2
    builder.array_save_directory = tmp_path
    builder.inst_uvw_irad = jnp.asarray(
        [
            [[0.0], [0.0], [0.0]],
            [[0.25], [0.0], [0.0]],
        ]
    )
    builder.beam_response = jnp.asarray(
        [
            [[1.0, 0.5], [0.75, 0.25]],
            [[0.8, 0.4], [0.6, 0.2]],
        ]
    )
    builder.fourier_geometry = SimpleNamespace(
        uv=jnp.asarray([[0.5], [0.0]]),
        eta=jnp.asarray([-0.5]),
        transverse_basis=jnp.asarray([[1.0], [-1.0]]),
        spectral_basis=jnp.asarray([[-1.0], [1.0]]),
    )
    builder.image_geometry = SimpleNamespace(
        lmn_fg=jnp.asarray(
            [
                [[0.0, 0.5], [0.0, 0.0], [1.0, np.sqrt(0.75)]],
                [[0.1, 0.4], [0.0, 0.0], [np.sqrt(0.99), np.sqrt(0.84)]],
            ]
        ),
        time_ind_center=0,
        eor_fg_shared_axis=jnp.asarray([0, 1]),
        pixel_area_sr=0.5,
    )
    return builder


def test_component_builders_match_concatenated_operator(tmp_path):
    builder = _minimal_builder(tmp_path)

    eor_path = builder.build_T_eor()
    foreground_path = builder.build_T_fg()
    joint_path = builder.build_T()

    eor = np.load(eor_path)
    foreground = np.load(foreground_path)
    joint = np.load(joint_path)

    assert eor.shape == (4, 1)
    assert foreground.shape == (4, 6)
    assert joint.shape == (4, 7)
    np.testing.assert_allclose(joint, np.concatenate([eor, foreground], axis=1))


def test_blockwise_gram_matrix_is_published_only_when_complete(tmp_path):
    builder = BuildMatrices.__new__(BuildMatrices)
    builder.array_save_directory = tmp_path
    builder.t_ninv_t_block_cols = 2
    builder.t_ninv_t_right_block_batch = 1

    T = np.asarray(
        [
            [1.0 + 1.0j, 2.0, -1.0j],
            [0.5, -2.0j, 3.0],
            [1.5 - 0.5j, 1.0, 2.0j],
        ],
        dtype=np.complex128,
    )
    Ninv = np.asarray([2.0, 0.5, 3.0], dtype=np.complex128)
    np.save(tmp_path / "T.npy", T)
    np.save(tmp_path / "Ninv.npy", Ninv)

    builder.save_T_Ninv_T_blockwise_memmap()

    expected = T.conj().T @ (Ninv[:, None] * T)
    np.testing.assert_allclose(np.load(tmp_path / "T_Ninv_T.npy"), expected)
    assert not (tmp_path / "T_Ninv_T.partial.npy").exists()
    assert not (tmp_path / "T_Ninv_T.partial.npy.ready").exists()


def test_builder_reuses_precomputed_geometries(tmp_path):
    image_geometry = SimpleNamespace(
        az_za_fg=jnp.zeros((1, 2, 1)),
        n_pix_fg=1,
    )
    fourier_geometry = SimpleNamespace(
        uv=jnp.zeros((2, 0)),
        eta=jnp.zeros((0,)),
        transverse_basis=jnp.zeros((1, 0)),
        spectral_basis=jnp.zeros((1, 0)),
    )

    builder = BuildMatrices(
        n_u_eor=1,
        du_eor=1.0,
        n_v_eor=1,
        dv_eor=1.0,
        n_freq=1,
        f0_mhz=100.0,
        df_mhz=1.0,
        n_eta=1,
        n_t=1,
        d_t=10.0,
        image_geometry=image_geometry,
        fourier_geometry=fourier_geometry,
        uvw_array_m=jnp.zeros((1, 1, 3)),
        array_save_directory=tmp_path,
    )

    assert builder.image_geometry is image_geometry
    assert builder.fourier_geometry is fourier_geometry
