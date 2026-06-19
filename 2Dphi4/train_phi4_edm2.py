"""
Train an EDM2-style diffusion model on 2D phi4 lattice configurations.
Uses MPNCSNpp2D (magnitude-preserving NCSN++) from networks_edm2.py with
the same EDMDiffusionModel and EDMPrecond from edm_lightning.py.

Differences vs train_phi4_edm.py:
- Network: MPNCSNpp2D (bias-free, forced weight norm) instead of NCSNpp2D
- LR default raised to 5e-3 (EDM2 paper allows higher LR with MP layers)
- torch.compile retained, mp_silu/MPConv friendly

Usage:
    python train_phi4_edm2.py --L 32 --k 0.28 --sigma_data 0.372 \
        --epochs 10000 --batch_size 2048 --device cuda:2
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


def main():
    p = argparse.ArgumentParser(description="Train EDM2 diffusion on 2D phi4")
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, required=True)
    p.add_argument("--P_mean", type=float, default=-1.2)
    p.add_argument("--P_std", type=float, default=1.2)
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--lr", type=float, default=5e-3,
                   help="EDM2 default; MP layers allow higher LR")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--data_prefix", type=str,
                   default="trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--num_ckpts", type=int, default=80)
    p.add_argument("--gpu_data", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--channels", type=int, nargs="+",
                   default=[16, 32, 64, 128])
    p.add_argument("--use_attention", action="store_true")
    p.add_argument("--l_cond", action="store_true")
    # Ablation flags
    p.add_argument("--use_bias", action="store_true",
                   help="Re-introduce bias to MP conv/linear (ablation)")
    p.add_argument("--no_mp_silu", action="store_true",
                   help="Use plain SiLU instead of mp_silu (ablation)")
    p.add_argument("--no_mp_sum", action="store_true",
                   help="Use plain + for residuals instead of mp_sum (ablation)")
    p.add_argument("--output_suffix", type=str, default="")
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

    # MP NCSNpp2D — no marginal_prob_std plumbing needed (it returns raw F_θ).
    score_network = MPNCSNpp2D(
        channels=tuple(args.channels),
        use_attention=args.use_attention,
        l_cond=args.l_cond,
        use_bias=args.use_bias,
        use_mp_silu=not args.no_mp_silu,
        use_mp_sum=not args.no_mp_sum,
    )
    # Initial forced weight normalisation (saves an inconsistent step on epoch 1).
    apply_forced_weight_norm(score_network)
    n_params = sum(p.numel() for p in score_network.parameters())
    print(f"MPNCSNpp2D: channels={args.channels}  params={n_params/1e6:.2f}M  "
          f"l_cond={args.l_cond}  attention={args.use_attention}")

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
    print(f"EDM config: sigma_data={args.sigma_data}, P_mean={args.P_mean}, "
          f"P_std={args.P_std}, sigma_min={args.sigma_min}, sigma_max={args.sigma_max}")
    print(f"LR: {args.lr}    (EDM2 default 5e-3 with MP layers)")

    output_dir = (f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm2"
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
