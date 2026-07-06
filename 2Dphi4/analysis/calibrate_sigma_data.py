"""
Compute sigma_data (per-site std of training data in [-1,1] norm space) for EDM.

EDM is robust to the exact value, so the mean-over-sites std is the recommended
target. Also reports max-over-sites for diagnostics.

Usage:
    python calibrate_sigma_data.py --k 0.28 --L 32
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from data import FieldDataModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--data_prefix", type=str,
                   default="../trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--data_path", type=str, default=None,
                   help="Override full data path")
    args = p.parse_args()

    if args.data_path is None:
        args.data_path = f"{args.data_prefix}_k={args.k}_l={args.l}_{args.L}^2.jld2"

    print(f"Reading: {args.data_path}")
    dm = FieldDataModule(data_path=args.data_path, batch_size=64,
                         normalize=True, device=None)
    dm.setup()

    # train_data is TensorDataset(configs, configs); configs is (N, 1, L, L) in [-1,1]
    x = dm.train_data.tensors[0].numpy()
    N = x.shape[0]
    print(f"Loaded {N} configurations of shape {x.shape[1:]}")
    print(f"Pooled normalisation: [{dm.cfgs_min:.4f}, {dm.cfgs_max:.4f}]")

    # Per-site std across samples
    per_site_std = x.std(axis=0)              # (1, L, L)
    sigma_mean = float(per_site_std.mean())
    sigma_max  = float(per_site_std.max())
    sigma_min  = float(per_site_std.min())

    # Also global std (collapse all axes) and per-site mean for context
    global_std = float(x.std())
    per_site_mean = x.mean(axis=0)
    abs_mean_max = float(np.abs(per_site_mean).max())

    print()
    print("=== sigma_data calibration (EDM) ===")
    print(f"  per-site std:   mean = {sigma_mean:.4f}")
    print(f"                  min  = {sigma_min:.4f}")
    print(f"                  max  = {sigma_max:.4f}")
    print(f"  global std (all axes) = {global_std:.4f}")
    print(f"  max |per-site mean|   = {abs_mean_max:.4f}    "
          f"(non-zero ⇒ broken-phase data not yet Z2-symmetrised)")
    print()
    print(f"  RECOMMENDED for EDM:  --sigma_data {sigma_mean:.3f}")
    print()
    print("  (EDM is robust to sigma_data within a factor of ~2; mean per-site std "
          "is the Karras-recommended choice.)")


if __name__ == "__main__":
    main()
