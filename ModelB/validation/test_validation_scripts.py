"""Small synthetic regression tests for the conditional validators."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from validation.analyze_cond_samples import main as analyze_main
from validation.analyze_cond_samples import radial_sk, read_indexed_cfgs
from validation.compare_conditional_gt import main as compare_main


def _scalar(handle, name, value):
    if isinstance(value, str):
        value = np.bytes_(value)
    handle.create_dataset(name, data=value)


def _make_fixture(tmp_path):
    lattice_size = 8
    n_total = 6
    indices = np.array([1, 4], dtype=np.int64)
    n_per_cond = 4
    rng = np.random.default_rng(123)

    conditions = rng.normal(size=(n_total, lattice_size, lattice_size))
    cond_path = tmp_path / "conditions.jld2"
    with h5py.File(cond_path, "w") as handle:
        handle.create_dataset("cfgs", data=conditions)
        _scalar(handle, "N", lattice_size)
        _scalar(handle, "κ", 0.28)
        _scalar(handle, "λ", 0.022)

    phi0 = conditions[indices]
    true_delta = rng.normal(scale=0.2, size=phi0.shape)
    target_fields = conditions.copy()
    target_fields[indices] = phi0 + true_delta
    target_path = tmp_path / "target.jld2"
    with h5py.File(target_path, "w") as handle:
        handle.create_dataset("cfgs", data=target_fields.astype(np.float32))
        _scalar(handle, "L", lattice_size)
        _scalar(handle, "kappa", np.float32(0.28))
        _scalar(handle, "lambda", np.float32(0.022))
        _scalar(handle, "time", np.float32(10.0))
        _scalar(handle, "source_path", str(cond_path))

    dm_delta = rng.normal(
        scale=0.2,
        size=(indices.size, n_per_cond, lattice_size, lattice_size),
    )
    phi_dm = phi0[:, None] + dm_delta
    run_dir = tmp_path / "run"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True)
    dm_path = samples_dir / "synthetic.npz"
    np.savez_compressed(
        dm_path,
        phi_gen=phi_dm.reshape(-1, lattice_size, lattice_size).astype(np.float32),
        dphi=dm_delta.reshape(-1, lattice_size, lattice_size).astype(np.float32),
        cond_indices=indices,
        n_per_cond=n_per_cond,
        zero_mode_raw=dm_delta.mean(axis=(2, 3)).reshape(-1).astype(np.float32),
    )
    dm_meta = {
        "L": lattice_size,
        "kappa": 0.28,
        "lam": 0.022,
        "horizon": 10,
        "cond_path": str(cond_path),
        "target_path": str(target_path),
        "n_cond": int(indices.size),
        "n_per_cond": n_per_cond,
    }
    dm_path.with_suffix(".json").write_text(json.dumps(dm_meta), encoding="utf-8")

    gt_replicates = (
        phi0[:, None]
        + true_delta[:, None]
        + rng.normal(
            scale=0.15,
            size=(indices.size, n_per_cond, lattice_size, lattice_size),
        )
    )
    gt_path = tmp_path / "conditional_gt.jld2"
    with h5py.File(gt_path, "w") as handle:
        handle.create_dataset(
            "cfgs",
            data=gt_replicates.reshape(-1, lattice_size, lattice_size).astype(
                np.float32
            ),
        )
        handle.create_dataset("cond_indices_0based", data=indices)
        _scalar(handle, "n_per_cond", n_per_cond)
        _scalar(handle, "L", lattice_size)
        _scalar(handle, "kappa", np.float32(0.28))
        _scalar(handle, "lambda", np.float32(0.022))
        _scalar(handle, "time", np.float32(10.0))
        _scalar(handle, "source_path", str(cond_path))
    return dm_path, target_path, gt_path


class ValidationScriptTests(unittest.TestCase):
    def test_radial_sk_excludes_corner_overflow(self):
        lattice_size = 128
        coord = np.arange(lattice_size)
        corner_mode = np.cos(
            2 * np.pi * (48 * coord[:, None] + 48 * coord[None, :])
            / lattice_size
        )[None]
        kk, spectrum = radial_sk(corner_mode)
        self.assertEqual(kk.size, lattice_size // 2)
        self.assertLessEqual(kk[-1], lattice_size / 2)
        self.assertLess(spectrum[-1], 1e-20)

    def test_indexed_hdf_read_ignores_unselected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            fields = np.arange(6 * 8 * 8, dtype=np.float64).reshape(6, 8, 8)
            fields[2] = np.nan
            path = tmp_path / "indexed.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("cfgs", data=fields)
            selected, _ = read_indexed_cfgs(path, np.array([4, 1]))
            self.assertTrue(np.array_equal(selected, fields[[4, 1]]))

    def test_analyze_and_compare_end_to_end_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            dm_path, _, gt_path = _make_fixture(tmp_path)
            analyze_main(["--samples", str(dm_path)])
            analysis_path = (
                tmp_path / "run" / "validation" / "synthetic_analysis.json"
            )
            self.assertTrue(analysis_path.is_file())
            self.assertTrue(
                json.loads(analysis_path.read_text())["all_dm_replicates_used"]
            )
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                analyze_main(["--samples", str(dm_path)])

            compare_main(["--dm", str(dm_path), "--gt", str(gt_path)])
            comparison_path = (
                tmp_path / "run" / "validation" / "synthetic_condcompare.json"
            )
            comparison = json.loads(comparison_path.read_text())
            self.assertEqual(
                comparison["gt_split_half_noise_floor_correction"],
                "difference / sqrt(2)",
            )
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                compare_main(["--dm", str(dm_path), "--gt", str(gt_path)])

    def test_compare_rejects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            dm_path, _, gt_path = _make_fixture(tmp_path)
            with h5py.File(gt_path, "r+") as handle:
                handle["time"][...] = np.float32(11.0)
            with self.assertRaisesRegex(ValueError, "time mismatch"):
                compare_main(["--dm", str(dm_path), "--gt", str(gt_path)])


if __name__ == "__main__":
    unittest.main()
