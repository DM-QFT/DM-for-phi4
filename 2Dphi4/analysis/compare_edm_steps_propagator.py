"""
Compare EDM Heun-step ablation samples against Wolff+FAHMC ground truth for
L=32 broken-phase data. Computes diagonal-mode propagator G(k) plus a
moment/cumulant table.

Usage (from 2Dphi4/):
    python analysis/compare_edm_steps_propagator.py
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py


def diagonal_propagator(cfgs: np.ndarray, n_boot: int = 200, seed: int = 0):
    """Returns (k_lat, G_mean, G_err) on diagonal modes (n, n), n=1..L/2.

    Conventions match 2Dphi4/julia/CorrelationUtils.jl:
        k̂² = Σ_i 4 sin²(π n_i / L)
        G(k) = ⟨|φ̃(k)|²⟩ / V    with  φ̃ = FFT(φ − ⟨φ⟩)
    """
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V

    n_diag = np.arange(1, L // 2 + 1)
    g_per_cfg = pk[:, n_diag, n_diag]

    k2 = 8.0 * np.sin(np.pi * n_diag / L) ** 2
    k_lat = np.sqrt(k2)

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, g_per_cfg.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k2, G, n_low: int = 5):
    """1/G = a k² + b → ξ = 1/√(b/a)."""
    if len(k2) < n_low:
        return float("nan")
    invG = 1.0 / G[:n_low]
    a, b = np.polyfit(k2[:n_low], invG, 1)
    if a <= 0 or b <= 0:
        return float("nan")
    return 1.0 / np.sqrt(b / a)


def load_cfgs(path: Path) -> np.ndarray:
    """Load configs from .h5 (key 'cfgs', (N,L,L)) or .npy ((L,L,N) per
    sample_phi4.py convention; auto-transposes)."""
    if str(path).endswith(".npy"):
        cfgs = np.load(path).astype(np.float64)
        # sample_phi4.py saves as (L, L, N); transpose to (N, L, L)
        if cfgs.ndim == 3 and cfgs.shape[0] == cfgs.shape[1] and cfgs.shape[2] != cfgs.shape[0]:
            cfgs = cfgs.transpose(2, 0, 1)
        return cfgs
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    if cfgs.ndim == 4 and cfgs.shape[1] == 1:
        cfgs = cfgs[:, 0]
    return cfgs


def moment_table(name, cfgs):
    """Return dict of scalar observables for a config set."""
    M = cfgs.mean(axis=(1, 2))
    return dict(
        name=name,
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
    p.add_argument("--steps", type=int, nargs="+",
                   default=[36, 72, 128, 256])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str,
                   default=None,
                   help="EDM sample data dir (default: runs/<auto>/data)")
    p.add_argument("--gt_path", type=str,
                   default=None)
    p.add_argument("--vesde_path", type=str,
                   default=None,
                   help="Path to VE-SDE samples .npy for head-to-head comparison")
    p.add_argument("--vesde_label", type=str,
                   default="VE-SDE σ=180 EM2000")
    p.add_argument("--out", type=str,
                   default=None)
    args = p.parse_args()

    # Default paths
    run_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm_sd{args.sigma_data}")
    if args.data_dir is None:
        args.data_dir = str(run_dir / "data")
    if args.gt_path is None:
        args.gt_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    if args.out is None:
        args.out = str(run_dir / "data" / f"edm_steps_compare_ep{args.ep}.pdf")

    # ---- Load GT ----
    gt = load_cfgs(Path(args.gt_path))
    print(f"GT (Wolff+FAHMC):  shape={gt.shape}  from {args.gt_path}")

    sources = [("Wolff+FAHMC (GT)", "k", "o", "-", gt)]

    # ---- Load EDM samples for each step count ----
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    markers = ["s", "^", "v", "D", "p", "*", "h"]
    for i, steps in enumerate(args.steps):
        fname = f"samples_edm_ep{args.ep}_N10240_steps{steps}_seed{args.seed}.h5"
        fpath = Path(args.data_dir) / fname
        if not fpath.exists():
            print(f"[WARN] missing {fpath}")
            continue
        cfgs = load_cfgs(fpath)
        sources.append((f"EDM Heun, steps={steps}",
                        colors[i % len(colors)], markers[i % len(markers)], "-", cfgs))
        print(f"EDM steps={steps:>4}: shape={cfgs.shape}  from {fpath.name}")

    # ---- Optional: VE-SDE baseline ----
    if args.vesde_path is not None:
        vp = Path(args.vesde_path)
        if vp.exists():
            cfgs = load_cfgs(vp)
            sources.append((args.vesde_label, "purple", "x", "--", cfgs))
            print(f"VE-SDE baseline: shape={cfgs.shape}  from {vp.name}")
        else:
            print(f"[WARN] VE-SDE path missing: {vp}")

    # ---- Moment table ----
    print()
    print("=" * 100)
    print(f"{'method':<28} {'N':>6} {'<phi>':>8} {'<|M|>':>8} {'<phi^2>':>8} "
          f"{'<phi^4>':>9} {'U_4':>8} {'chi':>9} {'%>0':>6}")
    print("=" * 100)
    rows = []
    for name, color, marker, ls, cfgs in sources:
        s = moment_table(name, cfgs)
        rows.append((s, color, marker, ls, cfgs))
        print(f"{s['name']:<28} {s['N']:>6} {s['phi']:>+8.4f} {s['absM']:>8.4f} "
              f"{s['phi2']:>8.4f} {s['phi4']:>9.4f} {s['binder']:>8.4f} "
              f"{s['chi']:>9.4f} {s['pct_pos']*100:>5.1f}%")
    print("=" * 100)

    # ---- Propagator G(k) ----
    print()
    print(f"{'method':<28}  {'ξ_fit':>7}  {'G(k_min)':>16}")
    g_rows = []
    for s, color, marker, ls, cfgs in rows:
        k_lat, G, E = diagonal_propagator(cfgs)
        xi = fit_xi(k_lat ** 2, G, n_low=5)
        g_rows.append((s['name'], color, marker, ls, k_lat, G, E, xi))
        print(f"{s['name']:<28}  {xi:>7.2f}  {G[0]:>10.4f}±{E[0]:<6.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    for name, color, marker, ls, k_lat, G, E, xi in g_rows:
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4,
                    ms=6, capsize=2,
                    label=f"{name}  (ξ={xi:.2f})" if not np.isnan(xi) else name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G(k)$  (diagonal modes)")
    ax.set_title(f"L={args.L}, κ={args.k} (broken), λ={args.l}, ckpt ep={args.ep}\nlog–log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")

    # Relative deviation: G_EDM / G_GT
    ax = axes[1]
    gt_G = g_rows[0][5]
    for name, color, marker, ls, k_lat, G, E, xi in g_rows[1:]:
        ratio = G / gt_G
        ratio_err = E / gt_G   # GT error negligible compared to bootstrap
        ax.errorbar(k_lat, ratio, yerr=ratio_err, fmt=marker + ls, color=color, lw=1.4,
                    ms=6, capsize=2, label=name)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G_{\rm EDM}(k) / G_{\rm GT}(k)$")
    ax.set_title("Relative deviation from GT")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_ylim(0.7, 1.3)

    fig.suptitle("EDM Heun-step ablation: diagonal propagator vs Wolff+FAHMC",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved figure: {out}")
    print(f"Saved figure: {str(out).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
