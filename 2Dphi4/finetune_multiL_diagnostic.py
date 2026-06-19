"""
Fine-tune a multi-L 2D phi^4 diffusion model from an existing checkpoint for
100 more epochs, saving snapshots every 20 epochs. Same pattern as
finetune_diagnostic.py but uses MultiLFieldDataModule and saves with the
``epoch={N:04d}.ckpt`` naming convention so that sample_phi4_crossL.py can
pick the snapshots up via its --ep argument.

Usage:
    python finetune_multiL_diagnostic.py \
        --run_dir runs/phi4_Lmulti8-16-32-64_k0.26_l0.022_ncsnpp \
        --base_ckpt runs/phi4_Lmulti8-16-32-64_k0.26_l0.022_ncsnpp/models/epoch=9111.ckpt \
        --start_epoch 9111 --device cuda:0
"""

import sys
sys.path.append("..")

import os
import re
import argparse
import functools

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import MultiLFieldDataModule


class FixedEpochsCheckpoint(Callback):
    """Save checkpoints at a fixed set of 1-based epoch numbers, using the
    ``epoch={N:04d}.ckpt`` naming convention."""

    def __init__(self, dirpath, save_at_epochs):
        super().__init__()
        self.dirpath = dirpath
        self.save_at_epochs = set(int(e) for e in save_at_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch in self.save_at_epochs:
            filepath = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            trainer.save_checkpoint(filepath)
            print(f"\n[FixedEpochsCheckpoint] Saved {filepath}")


def parse_L_list_from_dir(rdir):
    m = re.search(r"Lmulti([\d\-]+)_k", rdir)
    if not m:
        raise ValueError(f"Cannot parse L_list from run_dir name: {rdir}")
    return [int(x) for x in m.group(1).split("-")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--L_list", default=None,
                   help="Override L_list (default: parsed from --run_dir)")
    p.add_argument("--data_prefix", default="trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--start_epoch", type=int, required=True,
                   help="1-based epoch number of baseline ckpt")
    p.add_argument("--extra_epochs", type=int, default=100)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    if args.L_list is None:
        L_list = parse_L_list_from_dir(args.run_dir)
    else:
        L_list = [int(x) for x in args.L_list.split(",") if x.strip()]
    print(f"L_list = {L_list}")

    max_epochs = args.start_epoch + args.extra_epochs
    save_at = list(range(args.start_epoch + args.save_every,
                         max_epochs + 1, args.save_every))
    print(f"Fine-tune from epoch {args.start_epoch} to {max_epochs}")
    print(f"Saving snapshots at 1-based epochs: {save_at}")

    # Detect l_cond from the checkpoint state_dict
    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    sigma = h.get("sigma", 240.0)
    norm_min = h.get("norm_min", -4.0)
    norm_max = h.get("norm_max", 4.0)
    L_max = h.get("L", max(L_list))
    sd = ckpt.get("state_dict", {})
    l_cond_detected = any("L_embed" in k for k in sd.keys())
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]  "
          f"L_max={L_max}  l_cond={l_cond_detected}")

    data_paths = [
        f"{args.data_prefix}_k={args.k}_l={args.l}_{L}^2.jld2"
        for L in L_list
    ]
    for path in data_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    data_module = MultiLFieldDataModule(
        data_paths=data_paths,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device,
    )
    data_module.setup()

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn, l_cond=l_cond_detected)
    # Multi-L runs typically need static per-L compilation. Match training
    # config by using ``reduce-overhead`` mode with dynamic=False.
    score_model = torch.compile(score_model, mode="reduce-overhead", dynamic=False)

    model = DiffusionModel(
        score_model=score_model,
        sigma=sigma,
        lr=h.get("lr", 1e-3),
        L=L_max,
        ema_start_epoch=h.get("ema_start_epoch", 0),
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )

    out_models = os.path.join(args.run_dir, "models")
    os.makedirs(out_models, exist_ok=True)
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
