"""
Fine-tune the diffusion model from a baseline checkpoint for a small number of
additional epochs, saving snapshots at fixed-interval epochs.

Purpose: diagnose late-training snapshot variance (drift vs noise) without paying
the cost of N independent seed retrainings. NOT a substitute for true model
uncertainty — interpret σ_between as a lower bound on init variance.

Usage:
    python finetune_diagnostic.py \
        --run_dir runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640 \
        --device cuda:0
"""

import sys
sys.path.append("..")

import os
import argparse
import functools

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import FieldDataModule


class FixedEpochsCheckpoint(Callback):
    """Save checkpoints at a fixed set of 1-based epoch numbers."""

    def __init__(self, dirpath, save_at_epochs, name_fmt="finetune_ep{epoch}.ckpt"):
        super().__init__()
        self.dirpath = dirpath
        self.save_at_epochs = set(int(e) for e in save_at_epochs)
        self.name_fmt = name_fmt

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-based, matches LogScaleCheckpoint
        if epoch in self.save_at_epochs:
            filepath = os.path.join(self.dirpath, self.name_fmt.format(epoch=epoch))
            trainer.save_checkpoint(filepath)
            print(f"\n[FixedEpochsCheckpoint] Saved {filepath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Run directory containing models/ and lightning_logs/")
    p.add_argument("--base_ckpt", default=None,
                   help="Override baseline checkpoint path. Default: <run_dir>/lightning_logs/version_0/checkpoints/last")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma", type=float, default=640.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--data_path", default=None,
                   help="Default: trainingdata/cfgs_wolff_fahmc_k={k}_l={l}_{L}^2.jld2")
    p.add_argument("--start_epoch", type=int, default=10000,
                   help="1-based epoch count of baseline ckpt")
    p.add_argument("--extra_epochs", type=int, default=100)
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"

    if args.base_ckpt is None:
        import glob
        cands = sorted(glob.glob(f"{args.run_dir}/lightning_logs/version_*/checkpoints/*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"No baseline ckpt under {args.run_dir}/lightning_logs/")
        args.base_ckpt = cands[-1]
    print(f"Baseline checkpoint: {args.base_ckpt}")

    out_models = os.path.join(args.run_dir, "models")
    os.makedirs(out_models, exist_ok=True)

    max_epochs = args.start_epoch + args.extra_epochs
    save_at = list(range(args.start_epoch + args.save_every,
                         max_epochs + 1, args.save_every))
    print(f"Fine-tune from epoch {args.start_epoch} to {max_epochs}")
    print(f"Saving snapshots at 1-based epochs: {save_at}")

    data_module = FieldDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device,
    )
    data_module.setup()
    print(f"Normalization range: [{data_module.cfgs_min:.4f}, {data_module.cfgs_max:.4f}]")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)
    score_model = torch.compile(score_model, mode="reduce-overhead")

    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=args.L,
        ema_start_epoch=0,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )

    callback = FixedEpochsCheckpoint(out_models, save_at)

    device_id = int(args.device.split(":")[-1])
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[callback],
        default_root_dir=args.run_dir,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module, ckpt_path=args.base_ckpt)
    print("\nFine-tune complete!")


if __name__ == "__main__":
    main()
