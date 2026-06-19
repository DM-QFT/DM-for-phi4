"""
Check the empirical sigma rule for 2D phi^4 L=128 at three kappa values.

For each kappa:
  1. Load training cfgs (HMC) and the ckpt's norm_min/norm_max
  2. Map cfgs to normalised [-1, 1] space
  3. Estimate D_max_norm = max pairwise L2 distance
  4. Compare with std(t=1) = sqrt((sigma^2 - 1)/(2 ln sigma))
     Rule: std(t=1) ≈ 2 * D_max_norm

Usage: python analysis/check_sigma_choice.py
"""
from __future__ import annotations
from pathlib import Path
import math
import sys

import numpy as np
import h5py
import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[1]


def std_t1(sigma: float) -> float:
    return math.sqrt((sigma ** 2 - 1.0) / (2.0 * math.log(sigma)))


def max_pairwise_distance(X: np.ndarray) -> tuple[float, float, float]:
    """X: (N, D) float64. Returns (d_max, d_mean, d_q99)."""
    N = X.shape[0]
    sq = (X * X).sum(axis=1)
    # D^2_ij = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    G = X @ X.T
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.fill_diagonal(D2, 0.0)
    D2 = np.clip(D2, 0.0, None)
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(D2[iu])
    return float(d.max()), float(d.mean()), float(np.quantile(d, 0.99))


RUNS = {
    "0.26":   ("sigma300", 300),
    "0.2705": ("sigma450", 450),
    "0.28":   ("sigma640", 640),
}


def main():
    n_sub = 256  # subsample for pairwise cost (256^2/2 ≈ 32k pairs, each 16k-dim)
    L = 128
    print(f"L={L}, training cfgs subsample N={n_sub} per kappa\n")
    print(f"{'kappa':>7s}  {'sigma':>6s}  {'norm_min':>10s}  {'norm_max':>9s}  "
          f"{'D_max':>9s}  {'D_q99':>9s}  {'D_mean':>9s}  "
          f"{'std(t=1)':>10s}  {'std/2D':>8s}  {'comment':<10s}")

    for k_str, (sfx, sigma) in RUNS.items():
        # 1. read norm_min/max from a recent ckpt
        ckpt_path = REPO / "runs" / f"phi4_L128_k{k_str}_l0.022_ncsnpp_{sfx}" / "models" / "epoch=10000.ckpt"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        h = ckpt.get("hyper_parameters", {})
        norm_min = float(h.get("norm_min") or -6.22)
        norm_max = float(h.get("norm_max") or 6.19)
        sigma_h = float(h.get("sigma") or sigma)

        # 2. load training cfgs
        hmc_path = REPO / "trainingdata" / f"cfgs_wolff_fahmc_k={k_str}_l=0.022_{L}^2.jld2"
        with h5py.File(hmc_path, "r") as f:
            cfgs = np.array(f["cfgs"], dtype=np.float64)
        rng = np.random.default_rng(0)
        idx = rng.choice(cfgs.shape[0], size=min(n_sub, cfgs.shape[0]), replace=False)
        cfgs_sub = cfgs[idx]

        # 3. normalise to [-1, 1] using ckpt's norm_min/max
        x = (cfgs_sub - norm_min) / (norm_max - norm_min) * 2.0 - 1.0
        X = x.reshape(x.shape[0], -1)  # (N, V)

        d_max, d_mean, d_q99 = max_pairwise_distance(X)
        sd = std_t1(sigma_h)
        ratio = sd / (2.0 * d_max)
        verdict = "ok" if 0.8 <= ratio <= 1.4 else ("low!" if ratio < 0.8 else "high")

        print(f"{k_str:>7s}  {sigma_h:>6.0f}  {norm_min:>10.4f}  {norm_max:>9.4f}  "
              f"{d_max:>9.2f}  {d_q99:>9.2f}  {d_mean:>9.2f}  "
              f"{sd:>10.2f}  {ratio:>8.3f}  {verdict:<10s}")


if __name__ == "__main__":
    main()
