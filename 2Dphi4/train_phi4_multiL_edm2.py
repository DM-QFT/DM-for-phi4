"""
Train a multi-L EDM2-style (MPNCSNpp2D) diffusion model on 2D phi4 lattices.
Trains jointly on multiple lattice sizes with shared convolutional weights.

Usage:
    python train_phi4_multiL_edm2.py --L_list 8,16,32,64 --k 0.28 \
        --epochs 10000 --batch_size 1024 --device cuda:3 --gpu_data
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

from networks_edm2 import MPNCSNpp2D
from edm_lightning import EDMDiffusionModel
from mp_layers import apply_forced_weight_norm
from data import MultiLFieldDataModule


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


def parse_L_list(s):
    return [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description="Train multi-L EDM2 on 2D phi4")
    p.add_argument("--L_list", type=str, default="8,16,32,64",
                   help="Comma-sep lattice sizes for training")
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, default=None,
                   help="If None, auto-compute from data (per-site std)")
    p.add_argument("--P_mean", type=float, default=-1.2)
    p.add_argument("--P_std", type=float, default=1.2)
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--lr", type=float, default=5e-3,
                   help="EDM2 default; MP layers allow higher LR")
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Per-step batch (single L per step)")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--data_prefix", type=str,
                   default="trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--num_ckpts", type=int, default=80)
    p.add_argument("--gpu_data", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--channels", type=int, nargs="+",
                   default=[16, 32, 64, 128])
    p.add_argument("--output_suffix", type=str, default="")
    args = p.parse_args()

    L_list = parse_L_list(args.L_list)
    print(f"Training on L_list = {L_list}")

    data_paths = [
        f"{args.data_prefix}_k={args.k}_l={args.l}_{L}^2.jld2" for L in L_list
    ]
    for p_ in data_paths:
        if not os.path.isfile(p_):
            raise FileNotFoundError(p_)

    data_module = MultiLFieldDataModule(
        data_paths=data_paths,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device if args.gpu_data else None,
    )
    data_module.setup()
    print(f"Pooled normalization: [{data_module.cfgs_min:.4f}, {data_module.cfgs_max:.4f}]")

    # Auto-compute sigma_data if not provided (per-site std across multi-L data)
    if args.sigma_data is None:
        per_L_stds = []
        for L, t in data_module.data_by_L.items():
            arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            std_val = float(arr.std())
            per_L_stds.append(std_val)
            print(f"  L={L:>3}: per-site std (norm) = {std_val:.4f}  N={arr.shape[0]}")
        args.sigma_data = float(np.mean(per_L_stds))
        print(f"Auto sigma_data = {args.sigma_data:.4f}  (mean across L)")

    # Build MP NCSNpp2D
    score_network = MPNCSNpp2D(
        channels=tuple(args.channels),
    )
    apply_forced_weight_norm(score_network)
    n_params = sum(p.numel() for p in score_network.parameters())
    print(f"MPNCSNpp2D: channels={args.channels}  params={n_params/1e6:.2f}M")

    if not args.no_compile:
        # dynamic=False so torch.compile builds a separate static graph per L
        score_network = torch.compile(score_network, mode="reduce-overhead",
                                      dynamic=False)
        print("torch.compile enabled (static per-L specialization)")

    # L_for_state = max(L_list) is used for default _sample_shape; sampling can
    # override.
    L_for_state = max(L_list)
    model = EDMDiffusionModel(
        score_network=score_network,
        sigma_data=args.sigma_data,
        lr=args.lr,
        L=L_for_state,
        P_mean=args.P_mean,
        P_std=args.P_std,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )
    print(f"EDM config: sigma_data={args.sigma_data}, P_mean={args.P_mean}, "
          f"P_std={args.P_std}, sigma_min={args.sigma_min}, sigma_max={args.sigma_max}")
    print(f"LR: {args.lr}")

    L_tag = "-".join(str(L) for L in L_list)
    output_dir = (f"runs/phi4_Lmulti{L_tag}_k{args.k}_l{args.l}_edm2"
                  f"_sd{args.sigma_data:.3f}{args.output_suffix}")
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    checkpoint_log = LogScaleCheckpoint(
        dirpath=f"{output_dir}/models",
        max_epochs=args.epochs,
        num_checkpoints=args.num_ckpts,
    )
    print(f"Log-scale checkpoints ({len(checkpoint_log.save_epochs)} total)")

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
