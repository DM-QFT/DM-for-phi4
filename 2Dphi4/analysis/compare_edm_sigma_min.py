"""
EDM sigma_min ablation on 2D phi4 L=32 broken phase. Compares moments,
propagator G(k), and correlation length xi against Wolff+FAHMC ground truth.

Reuses helpers from compare_edm_churn.py. Baseline (sigma_min=0.002) reads
the existing samples_edm_ep10000_N10240_steps36_seed0.h5 file.

Usage (from 2Dphi4/):
    python analysis/compare_edm_sigma_min.py
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py


def diagonal_propagator(cfgs, n_boot=200, seed=0):
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V
    n_diag = np.arange(1, L // 2 + 1)
    g = pk[:, n_diag, n_diag]
    k_lat = np.sqrt(8.0 * np.sin(np.pi * n_diag / L) ** 2)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, g.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g[idx].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k2, G, n_low=5):
    if len(k2) < n_low:
        return float("nan")
    invG = 1.0 / G[:n_low]
    a, b = np.polyfit(k2[:n_low], invG, 1)
    if a <= 0 or b <= 0:
        return float("nan")
    return 1.0 / np.sqrt(b / a)


def load_cfgs(path: Path) -> np.ndarray:
    if str(path).endswith(".npy"):
        cfgs = np.load(path).astype(np.float64)
        if cfgs.ndim == 3 and cfgs.shape[0] == cfgs.shape[1] and cfgs.shape[2] != cfgs.shape[0]:
            cfgs = cfgs.transpose(2, 0, 1)
        return cfgs
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    if cfgs.ndim == 4 and cfgs.shape[1] == 1:
        cfgs = cfgs[:, 0]
    return cfgs


def moments(cfgs):
    M = cfgs.mean(axis=(1, 2))
    return dict(
        N=cfgs.shape[0],
        phi=float(cfgs.mean()),
        absM=float(np.abs(M).mean()),
        phi2=float((cfgs ** 2).mean()),
        phi4=float((cfgs ** 4).mean()),
        binder=1.0 - float((M ** 4).mean()) / (3 * float((M ** 2).mean()) ** 2),
        chi=float((M ** 2).mean() - np.abs(M).mean() ** 2),
        pct_pos=float((M > 0).mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, default=0.372)
    p.add_argument("--ep", type=str, default="10000")
    p.add_argument("--num_steps", type=int, default=36)
    p.add_argument("--sigma_min", type=float, nargs="+",
                   default=[2e-3, 5e-4, 1e-4, 1e-5, 1e-6])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--gt_path", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm_sd{args.sigma_data}")
    if args.data_dir is None:
        args.data_dir = str(run_dir / "data")
    if args.gt_path is None:
        args.gt_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    if args.out is None:
        args.out = str(run_dir / "data"
                       / f"edm_sigma_min_compare_steps{args.num_steps}_ep{args.ep}.pdf")

    gt = load_cfgs(Path(args.gt_path))
    print(f"GT (Wolff+FAHMC): shape={gt.shape}")
    sources = [("Wolff+FAHMC (GT)", "k", "o", "-", gt)]

    cmap = plt.cm.plasma
    n = len(args.sigma_min)
    markers = ["s", "^", "v", "D", "p", "*"]
    DEFAULT_SMIN = 2e-3   # Baseline file has no _smin tag (uses ckpt default)
    for i, smin in enumerate(args.sigma_min):
        if abs(smin - DEFAULT_SMIN) < 1e-9:
            # File saved without _smin tag (ckpt default = 0.002)
            fname = (f"samples_edm_ep{args.ep}_N10240"
                     f"_steps{args.num_steps}_seed{args.seed}.h5")
        else:
            fname = (f"samples_edm_ep{args.ep}_N10240"
                     f"_steps{args.num_steps}_smin{smin:g}_seed{args.seed}.h5")
        fpath = Path(args.data_dir) / fname
        if not fpath.exists():
            print(f"[WARN] missing {fpath}")
            continue
        cfgs = load_cfgs(fpath)
        color = cmap(i / max(n - 1, 1))
        sources.append((f"EDM Heun sigma_min={smin:g}",
                        color, markers[i % len(markers)], "-", cfgs))
        print(f"sigma_min={smin:>8g}: shape={cfgs.shape}  from {fpath.name}")

    print()
    print("=" * 100)
    print(f"{'method':<32} {'N':>6} {'<phi>':>8} {'<|M|>':>8} {'<phi^2>':>8} "
          f"{'<phi^4>':>9} {'U_4':>8} {'chi':>9} {'%>0':>6}")
    print("=" * 100)
    rows = []
    for name, color, marker, ls, cfgs in sources:
        s = moments(cfgs)
        rows.append((name, color, marker, ls, cfgs, s))
        print(f"{name:<32} {s['N']:>6} {s['phi']:>+8.4f} {s['absM']:>8.4f} "
              f"{s['phi2']:>8.4f} {s['phi4']:>9.4f} {s['binder']:>8.4f} "
              f"{s['chi']:>9.4f} {s['pct_pos']*100:>5.1f}%")
    print("=" * 100)

    print()
    print(f"{'method':<32}  {'xi_fit':>7}  {'G(k_min)':>16}")
    g_rows = []
    for name, color, marker, ls, cfgs, s in rows:
        k_lat, G, E = diagonal_propagator(cfgs)
        xi = fit_xi(k_lat ** 2, G, n_low=5)
        g_rows.append((name, color, marker, ls, k_lat, G, E, xi))
        print(f"{name:<32}  {xi:>7.2f}  {G[0]:>10.4f}+/-{E[0]:<6.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    for name, color, marker, ls, k_lat, G, E, xi in g_rows:
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4,
                    ms=6, capsize=2,
                    label=f"{name}  (xi={xi:.2f})" if not np.isnan(xi) else name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G(k)$  (diagonal modes)")
    ax.set_title(f"L={args.L}, k={args.k} (broken), lam={args.l}, "
                 f"ckpt ep={args.ep}, num_steps={args.num_steps}\nlog-log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[1]
    gt_G = g_rows[0][5]
    for name, color, marker, ls, k_lat, G, E, xi in g_rows[1:]:
        ratio = G / gt_G
        ratio_err = E / gt_G
        ax.errorbar(k_lat, ratio, yerr=ratio_err, fmt=marker + ls, color=color, lw=1.4,
                    ms=6, capsize=2, label=name)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G_{\rm EDM}(k) / G_{\rm GT}(k)$")
    ax.set_title("Relative deviation from GT")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_ylim(0.85, 1.15)

    fig.suptitle("EDM sigma_min ablation: diagonal propagator vs Wolff+FAHMC",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
