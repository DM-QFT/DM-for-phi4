"""Lightweight regression tests for the conditional Model B pipeline."""

import argparse
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from training.cond_lightning import (
    ConditionalDiffusionModel,
    ModelBPairDataModule,
    NCSNppCond2D,
)
from diffusion_lightning import marginal_prob_std
from training.train_modelB_cond import Checkpoints
from training.sample_modelB_cond import (
    _save_exclusive,
    checkpoint_epoch_from_name,
    select_checkpoint,
    validate_schedule,
)


class ModelBPipelineTests(unittest.TestCase):
    def test_all_pairs_are_normalized_and_setup_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            count, lattice = 12, 8
            rng = np.random.default_rng(19)
            # Match the real files: float64 conditions and float32 targets.
            cond = rng.normal(size=(count, lattice, lattice)).astype(np.float64)
            cond -= cond.mean(axis=(1, 2), keepdims=True)
            scales = np.linspace(0.05, 0.2, count, dtype=np.float32)
            scales[-1] = 100.0
            residual = cond * scales[:, None, None]
            target = (cond + residual).astype(np.float32)

            cond_path = directory / "cond.h5"
            target_path = directory / "target.h5"
            with h5py.File(cond_path, "w") as handle:
                handle.create_dataset("cfgs", data=cond)
            with h5py.File(target_path, "w") as handle:
                handle.create_dataset("cfgs", data=target)

            module = ModelBPairDataModule(
                str(cond_path), str(target_path), batch_size=3, device="cpu",
            )
            module.setup()
            first_train_data = module.data_on_gpu
            module.setup()
            self.assertIs(module.data_on_gpu, first_train_data)

            # HDF5 source dtypes govern normalization, as in the old run.
            residual_as_loaded = target - cond
            expected_A_res = float(np.max(np.abs(residual_as_loaded)))
            expected_A_cond = float(np.max(np.abs(cond)))
            self.assertEqual(module.A_res, expected_A_res)
            self.assertEqual(module.A_cond, expected_A_cond)
            self.assertEqual(module.num_pairs, count)
            self.assertEqual(len(module.data_on_gpu), count)
            self.assertFalse(hasattr(module, "val_data"))

            expected = np.stack(
                [residual_as_loaded / expected_A_res, cond / expected_A_cond],
                axis=1,
            ).astype(np.float32)
            np.testing.assert_array_equal(
                module.data_on_gpu.numpy(), expected
            )
            batch = next(iter(module.train_dataloader()))
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.shape, (3, 2, lattice, lattice))

    def test_mixed_l_batches_have_one_size_and_shared_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rng = np.random.default_rng(23)
            cond_paths, target_paths = [], []
            for lattice in (8, 16):
                cond = rng.normal(size=(6, lattice, lattice)).astype(np.float32)
                residual = np.float32(lattice / 20) * cond
                target = cond + residual
                cond_path = directory / f"cond_L{lattice}.h5"
                target_path = directory / f"target_L{lattice}.h5"
                with h5py.File(cond_path, "w") as handle:
                    handle.create_dataset("cfgs", data=cond)
                with h5py.File(target_path, "w") as handle:
                    handle.create_dataset("cfgs", data=target)
                cond_paths.append(cond_path)
                target_paths.append(target_path)

            module = ModelBPairDataModule(
                cond_paths, target_paths, batch_size=3,
                device="cpu", expected_L=(8, 16),
            )
            module.setup()
            self.assertEqual(module.num_pairs_by_L, {8: 6, 16: 6})
            self.assertEqual(module.num_pairs, 12)
            self.assertEqual(set(module.data_by_L), {8, 16})
            self.assertAlmostEqual(
                max(t[:, 0].abs().max().item() for t in module.data_by_L.values()),
                1.0,
            )
            self.assertAlmostEqual(
                max(t[:, 1].abs().max().item() for t in module.data_by_L.values()),
                1.0,
            )
            loader = module.train_dataloader()
            self.assertEqual(len(loader), 4)
            for batch in loader:
                self.assertEqual(batch.shape[:2], (3, 2))
                self.assertIn(batch.shape[-1], (8, 16))
                self.assertEqual(batch.shape[-2], batch.shape[-1])

    def test_log_only_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = Checkpoints(directory, max_epochs=100, num_checkpoints=5)

            class Trainer:
                def __init__(self):
                    self.current_epoch = 0
                    self.paths = []

                def save_checkpoint(self, path):
                    self.paths.append(Path(path).name)

            trainer = Trainer()
            unsaved = next(
                epoch for epoch in range(1, 101)
                if epoch not in callback.save_epochs
            )
            trainer.current_epoch = unsaved - 1
            callback.on_train_epoch_end(trainer, None)
            self.assertEqual(trainer.paths, [])

            trainer.current_epoch = 99
            callback.on_train_epoch_end(trainer, None)
            self.assertEqual(trainer.paths, ["epoch=0100.ckpt"])

    def test_foreach_ema_matches_previous_update(self):
        score = NCSNppCond2D(
            lambda time: marginal_prob_std(time, sigma=25.0),
            channels=[8, 16, 32, 64],
            embed_dim=32,
        )
        model = ConditionalDiffusionModel(
            score, sigma=25.0, L=16, ema_decay=0.9,
        )
        reference = ExponentialMovingAverage(
            model._ema_parameters, decay=0.9,
        )
        with torch.no_grad():
            for parameter in model._ema_parameters:
                parameter.add_(torch.randn_like(parameter) * 0.01)

        model._update_ema_foreach()
        reference.update()

        self.assertEqual(model.ema.num_updates, reference.num_updates)
        for actual, expected in zip(
                model.ema.shadow_params, reference.shadow_params):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_numeric_checkpoint_selection_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            models = run_dir / "models"
            models.mkdir()
            for epoch in (20, 120, 9111, 10000):
                (models / f"epoch={epoch:04d}.ckpt").touch()
            (models / "notes-epoch=99999.ckpt").touch()
            (models / "last.ckpt").touch()
            self.assertIsNone(
                checkpoint_epoch_from_name("notes-epoch=99999.ckpt")
            )
            selected, epoch = select_checkpoint(run_dir, None)
            self.assertEqual((selected.name, epoch), ("epoch=10000.ckpt", 10000))
            selected, epoch = select_checkpoint(run_dir, 20)
            self.assertEqual((selected.name, epoch), ("epoch=0020.ckpt", 20))

    def test_outputs_publish_exclusively_and_schedule_typos_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            npz_path = directory / "sample.npz"
            json_path = directory / "sample.json"
            arrays = {"value": np.arange(4, dtype=np.float32)}
            _save_exclusive(npz_path, json_path, arrays, {"seed": 4})
            with np.load(npz_path) as archive:
                self.assertTrue(np.array_equal(archive["value"], arrays["value"]))
            self.assertEqual(npz_path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(json_path.stat().st_mode & 0o777, 0o640)
            with self.assertRaises(FileExistsError):
                _save_exclusive(npz_path, json_path, arrays, {"seed": 4})
            with self.assertRaises(argparse.ArgumentTypeError):
                validate_schedule("logs")


if __name__ == "__main__":
    unittest.main()
