"""Check the public helper imports and relocated I/O and sampler utilities."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import pytest

import bayeseor_jax as api
from bayeseor_jax.configuration import BayesEoRParser, RunConfig
from bayeseor_jax.inference.multinest_paths import (
    MultiNestPathManager, _prepare_multinest_path, _cleanup_multinest_path,
)
from bayeseor_jax.inference.prior import parse_uprior_inds
from bayeseor_jax.inference.run_ns import estimate_average_eval_time
from bayeseor_jax.io import (
    load_numpy_dict, save_numpy_dict, load_sharded_npy, write_analysis_metadata,
    write_log_files,
)


def test_helper_public_exports_survive_reorganisation():
    assert api.BayesEoRParser is BayesEoRParser
    assert api.parse_uprior_inds is parse_uprior_inds
    assert api.load_numpy_dict is load_numpy_dict
    assert api.save_numpy_dict is save_numpy_dict
    np.testing.assert_array_equal(parse_uprior_inds("1:3", 4), [False, True, True, False])


def test_numpy_payload_roundtrip_and_overwrite_guard(tmp_path):
    path = tmp_path / "data.npy"
    data = np.array([1 + 2j, 3 - 4j])
    save_numpy_dict(path, data, {"model": "fg"}, version="test")
    np.testing.assert_array_equal(load_numpy_dict(path, as_jax=True), data)
    with pytest.raises(ValueError, match="clobber"):
        save_numpy_dict(path, data, {})
    save_numpy_dict(path, data * 2, {}, clobber=True)
    np.testing.assert_array_equal(load_numpy_dict(path), data * 2)


def test_sharded_loading_preserves_plain_array(tmp_path):
    path = tmp_path / "gram.npy"
    data = np.arange(9).reshape(3, 3).astype(complex)
    np.save(path, data)
    sharding = NamedSharding(Mesh(np.asarray(jax.devices()[:1]), ("x",)), PartitionSpec())
    np.testing.assert_array_equal(load_sharded_npy(path, sharding), data)
    np.testing.assert_array_equal(load_numpy_dict(path), data)


def test_run_metadata_preserves_existing_settings_and_version(tmp_path):
    write_log_files({"model": "fg"}, out_dir=tmp_path, version="first")
    write_log_files(RunConfig(model="eor"), out_dir=tmp_path, version="second")
    assert json.loads((tmp_path / "args.json").read_text()) == {"model": "fg"}
    assert (tmp_path / "version.txt").read_text() == "first\n"


def test_numpy_payload_accepts_run_configuration(tmp_path):
    path = tmp_path / "data.npy"
    save_numpy_dict(path, np.arange(3), RunConfig(model="fg"))
    saved = np.load(path, allow_pickle=True).item()
    assert saved["args"]["model"] == "fg"
    np.testing.assert_array_equal(saved["data"], np.arange(3))


@pytest.mark.parametrize("args", [None, SimpleNamespace(model="fg")])
def test_metadata_rejects_unsupported_argument_types(tmp_path, args):
    with pytest.raises(TypeError, match="RunConfig or dictionary"):
        write_log_files(args, out_dir=tmp_path)
    with pytest.raises(TypeError, match="RunConfig or dictionary"):
        save_numpy_dict(tmp_path / "data.npy", np.arange(3), args)
    assert not (tmp_path / "args.json").exists()
    assert not (tmp_path / "data.npy").exists()


def test_analysis_metadata_retains_output_format(tmp_path):
    write_analysis_metadata(
        output_dir=tmp_path, args={"model": "eor"}, k_vals=[.15, .4],
        mod_k_vo=[.1, .2, .4], bin_id_per_k_mode=[0, 0, 1],
        valid_mask=[True, True, True], version="test",
    )
    assert json.loads((tmp_path / "args.json").read_text()) == {"model": "eor"}
    np.testing.assert_allclose(np.loadtxt(tmp_path / "k-vals.txt"), [.15, .4])
    np.testing.assert_allclose(np.loadtxt(tmp_path / "k-vals-bins.txt"), [.1, .4, .4])
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "k-vals-nsamples.txt"), [2, 1])


def test_multinest_short_path_preserves_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / ("long-output-" * 12)
    manager = MultiNestPathManager(output)
    basename = _prepare_multinest_path(manager)
    assert len(basename + "post_equal_weights.dat") <= 100
    product = Path(basename + "post_equal_weights.dat")
    product.write_text("saved samples")
    _cleanup_multinest_path(manager)
    assert not manager.short_link.is_symlink()
    assert (output / "data-post_equal_weights.dat").read_text() == "saved samples"


def test_multinest_short_path_refuses_file_collision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = MultiNestPathManager(tmp_path / ("long-output-" * 12))
    manager.short_link.parent.mkdir(parents=True)
    manager.short_link.write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        _prepare_multinest_path(manager)
    assert manager.short_link.read_text() == "do not overwrite"


def test_likelihood_timing_checks_values():
    assert estimate_average_eval_time(lambda theta: -1., np.zeros(1), 2) >= 0
    with pytest.raises(ValueError, match="Non-finite"):
        estimate_average_eval_time(lambda theta: np.nan, np.zeros(1), 1)
