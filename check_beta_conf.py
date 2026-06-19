"""Verify the configurational-temperature estimator on phi^4 HMC data.

For P(φ) ∝ exp(−S_phi4) the per-sample estimator
    β̂(φ) = Tr(H)/g² − 2·g^T H g / (g²)²
satisfies ⟨β̂⟩ = 1 exactly. This script runs three checks against the
analytic implementation in `phi4_action.phi4_beta_conf`:

  [A] autograd cross-check on a few small-L cfgs (formula correctness)
  [B] full-ensemble ⟨β̂⟩ on HMC data (should → 1)
  [C] same answer via the normalised-x path with the true (pmin, pmax)

Examples (run from repo root):
  python check_beta_conf.py --data 2Dphi4/trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_64^2.jld2 \
      --k 0.2705 --l 0.022 --dim 2 --device cuda:2
  python check_beta_conf.py --data 3Dphi4/trainingdata/cfgs_wolff_fahmc_k=0.1923_l=0.9_64^3.jld2 \
      --k 0.1923 --l 0.9 --dim 3 --device cuda:2 --max_n 512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.autograd.functional as autoF

from phi4_action import phi4_action, phi4_beta_conf


def load_hmc(path: Path, dim: int, max_n: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    if cfgs.ndim != 1 + dim:
        raise ValueError(f"shape {cfgs.shape} not compatible with dim={dim}")
    sample_axis = int(np.argmax(cfgs.shape))
    if sample_axis != 0:
        cfgs = np.moveaxis(cfgs, sample_axis, 0)
    return cfgs[:max_n]


def beta_hat_batched(phi: torch.Tensor, k: float, l: float, phi_min: float,
                     phi_max: float, dim: int, batch: int) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, phi.shape[0], batch):
            out.append(
                phi4_beta_conf(phi[i:i + batch], k, l, phi_min, phi_max,
                               spatial_dims=dim).cpu().numpy()
            )
    return np.concatenate(out)


def beta_hat_autograd(phi_cfg: torch.Tensor, k: float, l: float, dim: int) -> float:
    """Brute-force β̂ via autograd Hessian. phi_cfg shape: (L, ...) in p-space."""
    shape = tuple(phi_cfg.shape)
    N = phi_cfg.numel()
    x = phi_cfg.detach().reshape(-1).clone()

    def S_flat(z: torch.Tensor) -> torch.Tensor:
        return phi4_action(z.view(1, 1, *shape), k, l, -1.0, 1.0, dim).squeeze()

    H = autoF.hessian(S_flat, x).reshape(N, N)
    g = autoF.jacobian(S_flat, x).reshape(N)
    g2 = (g * g).sum()
    gHg = g @ (H @ g)
    return (torch.diagonal(H).sum() / g2 - 2.0 * gHg / g2 ** 2).item()


def bootstrap_mean(x: np.ndarray, n_boot: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = x[rng.integers(0, N, size=N)].mean()
    return float(means.mean()), float(means.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="HMC .jld2 path")
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, required=True)
    p.add_argument("--dim", type=int, choices=[2, 3], required=True)
    p.add_argument("--max_n", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n_autograd", type=int, default=3)
    p.add_argument("--autograd_L", type=int, default=6,
                   help="Crop cfgs to this L for the autograd test (speed).")
    args = p.parse_args()

    cfgs = load_hmc(Path(args.data), args.dim, args.max_n)
    L = cfgs.shape[1]
    print(f"Loaded {Path(args.data).name}")
    print(f"  N={cfgs.shape[0]}  dim={args.dim}  L={L}  κ={args.k}  λ={args.l}")
    print(f"  φ range [{cfgs.min():.3f}, {cfgs.max():.3f}]  "
          f"⟨φ⟩={cfgs.mean():.4f}  ⟨φ²⟩={(cfgs ** 2).mean():.4f}")

    phi = torch.from_numpy(cfgs).unsqueeze(1).to(args.device).double()

    # [A] autograd cross-check on small cropped cfgs
    print(f"\n[A] autograd cross-check  (first {args.n_autograd} cfgs, "
          f"cropped to L={args.autograd_L})")
    crop = (slice(None),) * 2 + (slice(0, args.autograd_L),) * args.dim
    for i in range(min(args.n_autograd, phi.shape[0])):
        sample_full = phi[i:i + 1]
        sample = sample_full[crop]
        with torch.no_grad():
            b_an = phi4_beta_conf(sample, args.k, args.l, -1.0, 1.0,
                                  spatial_dims=args.dim).item()
        sample_cfg = sample.squeeze(0).squeeze(0)
        b_ag = beta_hat_autograd(sample_cfg, args.k, args.l, args.dim)
        print(f"    cfg {i}: analytic β̂={b_an:+.8f}  "
              f"autograd β̂={b_ag:+.8f}  |Δ|={abs(b_an - b_ag):.2e}")

    # [B] full-ensemble β̂ in p-space (phi treated as p directly)
    beta_p = beta_hat_batched(phi, args.k, args.l, -1.0, 1.0, args.dim, args.batch)
    mean, err = bootstrap_mean(beta_p)
    print(f"\n[B] full ensemble (p-space)")
    print(f"    ⟨β̂⟩ = {mean:.5f} ± {err:.5f}   (N={len(beta_p)}, expect → 1.0)")
    print(f"    median={np.median(beta_p):.4f}   "
          f"5–95%=[{np.percentile(beta_p, 5):.4f}, "
          f"{np.percentile(beta_p, 95):.4f}]")

    # [C] same data, but via the normalised-x denorm path
    pmin, pmax = float(cfgs.min()), float(cfgs.max())
    phi_norm = 2.0 * (phi - pmin) / (pmax - pmin) - 1.0
    beta_x = beta_hat_batched(phi_norm, args.k, args.l, pmin, pmax,
                              args.dim, args.batch)
    diff = float(np.max(np.abs(beta_p - beta_x)))
    print(f"\n[C] same ensemble via norm path  (pmin={pmin:.4f}, pmax={pmax:.4f})")
    print(f"    max |β̂_p − β̂_x| over N samples = {diff:.2e}  "
          f"(expect ~1e-12 from fp64)")


if __name__ == "__main__":
    main()
