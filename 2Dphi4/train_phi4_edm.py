"""
Train an EDM (Karras et al. 2022) diffusion model on 2D phi4 lattice
configurations. Parallel to train_phi4.py but uses EDM preconditioning
instead of VE-SDE.

Usage:
    python train_phi4_edm.py --L 32 --k 0.28 --sigma_data 0.30 \
        --epochs 5000 --device cuda:3
"""

import sys
sys.path.append("..")

import os
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from networks import NCSNpp2D
from edm_lightning import EDMDiffusionModel, trivial_std
from data import FieldDataModule


class LogScaleCheckpoint(Callback):
    """Save checkpoints on a log-scale schedule: denser early, sparser later."""

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


def main():
    p = argparse.ArgumentParser(description="Train EDM diffusion model on 2D phi4")
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=float, default=0.28, help="Hopping parameter (0.28 = broken phase)")
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, required=True,
                   help="Per-site std of training data in [-1,1] norm space "
                        "(run analysis/calibrate_sigma_data.py first)")
    p.add_argument("--P_mean", type=float, default=-1.2,
                   help="log-normal sigma sampling mean (Karras default)")
    p.add_argument("--P_std", type=float, default=1.2,
                   help="log-normal sigma sampling std (Karras default)")
    p.add_argument("--sigma_min", type=float, default=0.002,
                   help="Min sigma for Heun sampler schedule")
    p.add_argument("--sigma_max", type=float, default=80.0,
                   help="Max sigma for Heun sampler schedule")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--data_prefix", type=str, default="trainingdata/cfgs_wolff_fahmc",
                   help="Data file prefix (default uses Wolff+FAHMC dataset)")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Resume training from this checkpoint")
    p.add_argument("--num_ckpts", type=int, default=100)
    p.add_argument("--gpu_data", action="store_true",
                   help="Load dataset onto GPU once (avoids per-epoch H2D transfer)")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile (useful for debugging)")
    p.add_argument("--output_suffix", type=str, default="")
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = f"{args.data_prefix}_k={args.k}_l={args.l}_{args.L}^2.jld2"

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(args.data_path)

    # Data
    data_module = FieldDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device if args.gpu_data else None,
    )
    data_module.setup()
    print(f"Normalization range: [{data_module.cfgs_min:.4f}, {data_module.cfgs_max:.4f}]")

    # Score network: built with trivial_std so the internal /std division is a no-op.
    # EDM preconditioning is applied externally by EDMPrecond.
    score_network = NCSNpp2D(trivial_std, channels=[16, 32, 64, 128])
    print("Using NCSNpp2D (periodic BC) + EDM preconditioning")

    if not args.no_compile:
        score_network = torch.compile(score_network, mode="reduce-overhead")

    model = EDMDiffusionModel(
        score_network=score_network,
        sigma_data=args.sigma_data,
        lr=args.lr,
        L=args.L,
        P_mean=args.P_mean,
        P_std=args.P_std,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )
    print(f"EDM config: sigma_data={args.sigma_data}, "
          f"P_mean={args.P_mean}, P_std={args.P_std}, "
          f"sigma_min={args.sigma_min}, sigma_max={args.sigma_max}")
    print(f"EMA starts at epoch: {args.ema_start}")

    output_dir = (f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm"
                  f"_sd{args.sigma_data}{args.output_suffix}")
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    checkpoint_log = LogScaleCheckpoint(
        dirpath=f"{output_dir}/models",
        max_epochs=args.epochs,
        num_checkpoints=args.num_ckpts,
    )
    print(f"Log-scale checkpoints ({len(checkpoint_log.save_epochs)} total): "
          f"{sorted(checkpoint_log.save_epochs)[:5]} ... "
          f"{sorted(checkpoint_log.save_epochs)[-5:]}")

    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_log],
        default_root_dir=output_dir,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
