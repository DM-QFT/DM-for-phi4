"""Train the conditional Model B diffusion model on every available pair."""

import argparse
import functools
import json
import os
import re
import sys

MODEL_B_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DM_DIR = os.path.dirname(MODEL_B_DIR)
sys.path.insert(0, DM_DIR)

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from diffusion_lightning import marginal_prob_std
if __package__:
    from .cond_lightning import (
        ConditionalDiffusionModel, ModelBPairDataModule, NCSNppCond2D,
    )
else:
    sys.path.insert(0, MODEL_B_DIR)
    from training.cond_lightning import (
        ConditionalDiffusionModel, ModelBPairDataModule, NCSNppCond2D,
    )

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def canonical_path(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def default_run_dir(args):
    return canonical_path(os.path.join(
        MODEL_B_DIR, "runs",
        f"modelBcond_L{args.L}_k{args.kappa}_l{args.lam}"
        f"_t{args.horizon}_ncsnpp{args.output_suffix}",
    ))


def validate_args(args, parser):
    if args.L <= 0 or args.L % 8:
        parser.error("--L must be a positive multiple of 8")
    for name in ("kappa", "sigma", "lr"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.sigma <= 1:
        parser.error("--sigma must be greater than 1")
    if not np.isfinite(args.lam) or args.lam < 0:
        parser.error("--lam must be finite and non-negative")
    if (not np.isfinite(args.zero_mode_loss_weight)
            or args.zero_mode_loss_weight < 0):
        parser.error("--zero_mode_loss_weight must be finite and non-negative")
    for name in ("horizon", "batch_size", "epochs", "num_ckpts"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0 <= args.ema_start < args.epochs:
        parser.error("--ema_start must satisfy 0 <= ema_start < epochs")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        parser.error("--device must be cuda or cuda:N")
    if (len(args.output_suffix) > 64
            or not re.fullmatch(r"[A-Za-z0-9_.-]*", args.output_suffix)):
        parser.error("invalid --output_suffix")


def checkpoint_run_dir(checkpoint_path):
    checkpoint_path = canonical_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    models_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(models_dir) != "models":
        raise ValueError("resume checkpoint must be inside a models directory")
    return checkpoint_path, os.path.dirname(models_dir)


def validate_resume(checkpoint_path, args, data_module):
    """Check only the normalization/model facts needed for a safe resume."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError(f"cannot load checkpoint: {exc}") from exc
    hparams = checkpoint.get("hyper_parameters", {})
    if hparams.get("L") != args.L:
        raise ValueError(
            f"checkpoint L={hparams.get('L')} does not match L={args.L}"
        )
    expected_scales = {
        "norm_min": -data_module.A_res,
        "norm_max": data_module.A_res,
        "cond_norm_min": -data_module.A_cond,
        "cond_norm_max": data_module.A_cond,
    }
    for key, expected in expected_scales.items():
        try:
            matches = np.isclose(
                float(hparams[key]), expected, rtol=1e-12, atol=1e-12
            )
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"checkpoint {key}={hparams.get(key)!r} does not match "
                f"the full-data scale {expected:.17g}"
            )
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError("checkpoint has no integer epoch")
    completed_epochs = epoch + 1
    if args.epochs <= completed_epochs:
        raise ValueError(
            f"--epochs must exceed the {completed_epochs} completed epochs"
        )
    return completed_epochs


def write_run_metadata(output_dir, args, data_module):
    metadata = {
        "training_data_scope": "all_data",
        "normalization_scope": "all_data",
        "validation": "none",
        "A_res": data_module.A_res,
        "A_cond": data_module.A_cond,
        "num_pairs": data_module.num_pairs,
        "L": args.L,
        "kappa": args.kappa,
        "lam": args.lam,
        "horizon": args.horizon,
        "sigma": args.sigma,
        "cond_path": args.cond_path,
        "target_path": args.target_path,
    }
    with open(os.path.join(output_dir, "meta.json"), "x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


class Checkpoints(Callback):
    """Save only the requested log-spaced checkpoint history."""

    def __init__(self, dirpath, max_epochs, num_checkpoints):
        self.dirpath = dirpath
        epochs = np.geomspace(1, max_epochs, num=num_checkpoints).astype(int)
        self.save_epochs = set(np.unique(epochs).tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch in self.save_epochs:
            path = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            if os.path.exists(path):
                raise FileExistsError(f"refusing to overwrite {path}")
            trainer.save_checkpoint(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--kappa", type=float, default=0.28)
    parser.add_argument("--lam", type=float, default=0.022)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=250.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--ema_start", type=int, default=0)
    parser.add_argument("--zero_mode_loss_weight", type=float, default=0.0)
    parser.add_argument("--no_z2_aug", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cond_path")
    parser.add_argument("--target_path")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--num_ckpts", type=int, default=100)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument(
        "--compile_mode", default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    args = parser.parse_args()
    validate_args(args, parser)
    return args


def main():
    args = parse_args()
    if args.cond_path is None:
        args.cond_path = os.path.join(
            DM_DIR, "2Dphi4", "trainingdata",
            f"cfgs_wolff_fahmc_k={args.kappa}_l={args.lam}_{args.L}^2.jld2",
        )
    if args.target_path is None:
        args.target_path = os.path.join(
            MODEL_B_DIR, "runs",
            f"L{args.L}_Ne10240_dt0p002_hmc_t1_t10_t100",
            f"modelB_L{args.L}_Ne10240_hmc_kappa_{args.kappa}"
            f"_t{args.horizon}_dt0.002.jld2",
        )
    args.cond_path = canonical_path(args.cond_path)
    args.target_path = canonical_path(args.target_path)

    if args.ckpt_path:
        args.ckpt_path, output_dir = checkpoint_run_dir(args.ckpt_path)
    else:
        output_dir = default_run_dir(args)
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"output already exists: {output_dir}; use a new --output_suffix"
            )

    data_module = ModelBPairDataModule(
        args.cond_path, args.target_path,
        batch_size=args.batch_size,
        device=args.device,
        expected_L=args.L,
    )
    data_module.setup()

    if args.ckpt_path:
        resume_epoch = validate_resume(args.ckpt_path, args, data_module)
        print(f"Resuming after epoch {resume_epoch}; all {data_module.num_pairs} "
              "pairs are used from the next epoch")
    else:
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=False)
        write_run_metadata(output_dir, args, data_module)

    marginal_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    score_model = NCSNppCond2D(marginal_fn, in_channels=2)
    score_model = score_model.to(memory_format=torch.channels_last)
    score_model = torch.compile(score_model, mode=args.compile_mode)
    model = ConditionalDiffusionModel(
        score_model, sigma=args.sigma, lr=args.lr, L=args.L,
        ema_start_epoch=args.ema_start,
        norm_min=-data_module.A_res, norm_max=data_module.A_res,
        cond_norm_min=-data_module.A_cond, cond_norm_max=data_module.A_cond,
        zero_mode_loss_weight=args.zero_mode_loss_weight,
        kappa=args.kappa, lam=args.lam, horizon=args.horizon,
        cond_data_path=args.cond_path, target_data_path=args.target_path,
        run_dir=output_dir, in_channels=2, z2_aug=not args.no_z2_aug,
    )

    models_dir = os.path.join(output_dir, "models")
    checkpoints = Checkpoints(
        models_dir, args.epochs, args.num_ckpts
    )
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=[device_id],
        callbacks=[checkpoints],
        default_root_dir=output_dir, precision="bf16-mixed",
        log_every_n_steps=10, enable_progress_bar=False,
        enable_checkpointing=False,
    )
    print(f"Output directory: {output_dir}/")
    print(f"Log-scale checkpoints: {len(checkpoints.save_epochs)}")
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("Training complete!")


if __name__ == "__main__":
    main()
