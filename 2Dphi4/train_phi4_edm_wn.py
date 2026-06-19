"""
EDM training (regular NCSNpp2D, with bias, plain SiLU, plain GroupNorm) +
forced weight normalisation applied after each optimizer step.

This isolates the "weight normalization" component of EDM2 to test whether
it alone is responsible for the cross-L ability.

Usage:
    python train_phi4_edm_wn.py --L 32 --k 0.28 --sigma_data 0.372 \\
        --epochs 2000 --batch_size 2048 --device cuda:1
"""

import sys
sys.path.append("..")

import math
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from networks import NCSNpp2D
from edm_lightning import EDMDiffusionModel, trivial_std
from data import FieldDataModule


class LogScaleCheckpoint(Callback):
    def __init__(self, dirpath, max_epochs, num_checkpoints=100):
        super().__init__()
        self.dirpath = dirpath
        log_epochs = np.unique(np.geomspace(1, max_epochs, num=num_checkpoints).astype(int))
        self.save_epochs = set(log_epochs.tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch in self.save_epochs:
            filepath = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            trainer.save_checkpoint(filepath)


class ForceWeightNormCallback(Callback):
    """After each optimizer step, project Conv2d/Linear weights to a fixed
    per-output-channel norm. This isolates the 'forced weight normalization'
    component of EDM2's MP recipe.

    target_norm = sqrt(2) preserves the He-init equivalent magnitude
    (since He-init weights have per-output-channel L2 norm ~ sqrt(2) on
    average); we just FREEZE this norm against drift during training.
    """

    def __init__(self, target_norm: float = math.sqrt(2.0)):
        super().__init__()
        self.target_norm = float(target_norm)

    @torch.no_grad()
    def _renorm(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
                w = m.weight.data
                w_flat = w.reshape(w.shape[0], -1)
                norm = w_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                w_flat.mul_(self.target_norm / norm)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # Called after optimizer.step() and before zero_grad — perfect for renorm.
        self._renorm(pl_module)

    def on_fit_start(self, trainer, pl_module):
        # Apply once at start to initialize at the target norm.
        self._renorm(pl_module)


def main():
    p = argparse.ArgumentParser(description="EDM + forced weight norm on 2D phi4")
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, required=True)
    p.add_argument("--P_mean", type=float, default=-1.2)
    p.add_argument("--P_std", type=float, default=1.2)
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="EDM default (lower than EDM2's 5e-3)")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--data_prefix", type=str, default="trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--num_ckpts", type=int, default=30)
    p.add_argument("--gpu_data", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--output_suffix", type=str, default="_wn")
    p.add_argument("--target_norm", type=float, default=math.sqrt(2.0),
                   help="Target per-output-channel norm for weight projection")
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = f"{args.data_prefix}_k={args.k}_l={args.l}_{args.L}^2.jld2"
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(args.data_path)

    data_module = FieldDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device if args.gpu_data else None,
    )
    data_module.setup()
    print(f"Normalization range: [{data_module.cfgs_min:.4f}, {data_module.cfgs_max:.4f}]")

    # EDM-style backbone (with bias, GroupNorm, plain SiLU). Bypass
    # marginal_prob_std with trivial_std so the network returns raw F_θ.
    score_network = NCSNpp2D(trivial_std, channels=[16, 32, 64, 128])
    print(f"NCSNpp2D + EDM precond + forced weight norm  "
          f"(target_norm={args.target_norm:.3f})")

    if not args.no_compile:
        score_network = torch.compile(score_network, mode="reduce-overhead")

    model = EDMDiffusionModel(
        score_network=score_network,
        sigma_data=args.sigma_data,
        lr=args.lr,
        L=args.L,
        P_mean=args.P_mean, P_std=args.P_std,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )

    output_dir = (f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm"
                  f"_sd{args.sigma_data}{args.output_suffix}")
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    callbacks = [
        ForceWeightNormCallback(target_norm=args.target_norm),
        LogScaleCheckpoint(dirpath=f"{output_dir}/models",
                           max_epochs=args.epochs,
                           num_checkpoints=args.num_ckpts),
    ]

    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=callbacks,
        default_root_dir=output_dir,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
