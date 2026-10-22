"""Small native JAXMG smoke check; launch one process per GPU with shared storage.

Example under an existing Slurm allocation (adjust binding to the cluster):
    srun python tests/gpu/run_foreground_svd.py /shared/scratch/kl-smoke

The destination must be a new directory. This does not use observation data or
run sampling, and checks the actual distributed solver rather than a substitute.
"""

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    # Follow the one-process-per-GPU Slurm convention used by JAXMG.
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        local_device = 0 if len(visible) == 1 else int(os.environ["SLURM_LOCALID"])
        jax.distributed.initialize(local_device_ids=[local_device])

    from bayeseor_jax.model.construction.kl_reduction import T_fg_svd_build_save

    if jax.process_index() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if jax.distributed.is_initialized():
        multihost_utils.sync_global_devices("smoke-directory")
    results = []
    for index, (shape, f) in enumerate([((71, 199), .1), ((199, 71), .1),
                                        ((71, 199), 0.), ((71, 199), 100.)]):
        rng = np.random.default_rng(index)
        left, _ = np.linalg.qr(rng.normal(size=(shape[0], 5)) + 1j*rng.normal(size=(shape[0], 5)))
        right, _ = np.linalg.qr(rng.normal(size=(shape[1], 5)) + 1j*rng.normal(size=(shape[1], 5)))
        values = np.array([20., 5., 2., .3, .02])
        weighted = (left * values) @ right.conj().T
        inverse_noise = np.linspace(.5, 2.1, shape[0])
        a_path = args.output_dir / f"A_{index}.npy"
        n_path = args.output_dir / f"Ninv_{index}.npy"
        out_path = args.output_dir / f"reduced_{index}.npy"
        if jax.process_index() == 0:
            np.save(a_path, weighted)
            np.save(n_path, inverse_noise)
        if jax.distributed.is_initialized():
            multihost_utils.sync_global_devices(f"smoke-inputs-{index}")
        T_fg_svd_build_save(a_path, n_path, output_path=out_path, f=f, tile_size=32)
        reduced = np.load(out_path)
        metadata = json.loads(out_path.with_suffix(".svd.json").read_text())
        expected_rank = 3 if f == .1 else (5 if f == 0 else 0)
        expected = left[:, :expected_rank] * values[:expected_rank] / np.sqrt(inverse_noise[:, None])
        if f != 0:
            assert reduced.shape == (shape[0], expected_rank)
        np.testing.assert_allclose(reduced @ reduced.conj().T, expected @ expected.conj().T,
                                   rtol=1e-8, atol=1e-9)
        assert metadata["discarded_rms_over_noise_rms"] <= f
        results.append(dict(shape=shape, f=f, retained_rank=reduced.shape[1],
                            discarded_rms=metadata["discarded_rms_over_noise_rms"]))
    if jax.process_index() == 0:
        print(json.dumps({"passed": results, "gpu_count": jax.device_count()}, indent=2), flush=True)
    if jax.distributed.is_initialized():
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
