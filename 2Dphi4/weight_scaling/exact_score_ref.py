"""Exact-score baseline: score covariance of the empirical Gaussian mixture.

p_t = (1/N) sum_i N(phi_i, sigma_t^2); s_exact via softmax weights. Reference =
first half of configs, queries drawn from second half (approximates the true
measure's score away from the memorization limit).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def exact_score(phit, ref, std2, chunk=64):
    # ||a-b||^2 = |a|^2 + |b|^2 - 2ab, softmax over mixture components
    s = torch.empty_like(phit)
    r2 = (ref * ref).sum(1)
    for i in range(0, phit.shape[0], chunk):
        q = phit[i:i + chunk]
        d2 = (q * q).sum(1, keepdim=True) + r2 - 2.0 * q @ ref.T
        w = torch.softmax(-d2 / (2 * std2), dim=1)
        s[i:i + chunk] = (w @ ref - q) / std2
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfgs", required=True)
    p.add_argument("--norm_min", type=float, default=-4.352510452270508)
    p.add_argument("--norm_max", type=float, default=4.3837151527404785)
    p.add_argument("--sigma", type=float, default=360.0)
    p.add_argument("--n_ref", type=int, default=5120)
    p.add_argument("--n_query", type=int, default=512)
    p.add_argument("--t_min", type=float, default=1e-4)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--n_t", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    with h5py.File(args.cfgs, "r") as f:
        cfgs = f["cfgs"][:].astype(np.float32)
    L = cfgs.shape[-1]
    phin = 2 * (cfgs - args.norm_min) / (args.norm_max - args.norm_min) - 1
    phin = torch.from_numpy(phin).to(args.device).reshape(len(phin), -1)
    ref, held = phin[: args.n_ref], phin[args.n_ref:]
    idx = torch.randperm(held.shape[0])[: args.n_query]
    q0 = held[idx]

    ts = np.geomspace(args.t_min, args.t_max, args.n_t)
    rows = []
    for t in ts:
        std2 = (args.sigma ** (2 * t) - 1.0) / (2.0 * np.log(args.sigma))
        phit = q0 + np.sqrt(std2) * torch.randn_like(q0)
        s = exact_score(phit, ref, std2).reshape(-1, L, L)
        S = (torch.fft.rfft2(s).abs() ** 2).mean(0) / (L * L)
        Sx = S[0, : L // 2 + 1]
        Sy = 0.5 * (S[1: L // 2, 0] + S[L // 2 + 1:, 0].flip(0))
        R = Sx.clone()
        R[1: L // 2] = 0.5 * (Sx[1: L // 2] + Sy)
        R[L // 2] = 0.5 * (Sx[L // 2] + S[L // 2, 0])
        R = R.cpu().numpy()
        for n in range(L // 2 + 1):
            k = 2 * np.pi * n / L
            khat = 2 * np.sin(k / 2)
            rows.append((t, std2, n, k, khat, R[n], 1.0 / R[n] - std2))
        print(f"t={t:.4g} done")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = "t\tsigma_t2\tn\tk\tkhat\tR\tG_net"
    np.savetxt(out, np.array(rows), fmt="%.8g", delimiter="\t", header=hdr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
