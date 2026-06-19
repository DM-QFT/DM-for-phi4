"""
EDM sampling-time hyperparameter grid evaluation. Takes a list of
(num_steps, S_churn, sigma_min) triples, computes moments + propagator,
and picks the best by an aggregate metric.

Usage (from 2Dphi4/):
    python analysis/compare_edm_hp_grid.py
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py


DEFAULT_SMIN = 2e-3   # baseline: ckpt-stored sigma_min


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
    )


def build_filename(steps, Sc, smin, ep="10000"):
    churn_tag = f"_Sc{Sc:g}" if Sc > 0 else ""
    smin_tag = f"_smin{smin:g}" if abs(smin - DEFAULT_SMIN) > 1e-9 else ""
    return f"samples_edm_ep{ep}_N10240_steps{steps}{churn_tag}{smin_tag}_seed0.h5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, default=0.372)
    p.add_argument("--ep", type=str, default="10000")
    p.add_argument("--combos", type=str, nargs="+",
                   default=[
                       "36,0,2e-3",     # deterministic baseline
                       "36,5,2e-3",     # best single S_churn
                       "36,0,1e-4",     # best single sigma_min
                       "36,3,1e-4",     # combo C2
                       "36,5,1e-4",     # combo C1
                       "72,5,1e-4",     # combo C3
                       "72,7,1e-4",     # combo C3'
                       "72,10,1e-4",    # combo C4
                   ],
                   help="Comma-sep tuples: num_steps,S_churn,sigma_min")
    p.add_argument("--gt_path", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm_sd{args.sigma_data}")
    data_dir = run_dir / "data"
    if args.gt_path is None:
        args.gt_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    if args.out is None:
        args.out = str(data_dir / f"edm_hp_grid_ep{args.ep}.pdf")

    gt = load_cfgs(Path(args.gt_path))
    gt_M = gt.mean(axis=(1, 2))
    print(f"GT (Wolff+FAHMC): N={gt.shape[0]}")

    gt_mom = moments(gt)
    k_lat, gt_G, gt_E = diagonal_propagator(gt)
    gt_xi = fit_xi(k_lat ** 2, gt_G, n_low=5)
    print(f"GT: xi={gt_xi:.3f}  G(k_min)={gt_G[0]:.4f}+/-{gt_E[0]:.4f}  "
          f"<|M|>={gt_mom['absM']:.4f}  <phi^2>={gt_mom['phi2']:.4f}  "
          f"<phi^4>={gt_mom['phi4']:.4f}  U_4={gt_mom['binder']:.4f}")
    print()

    rows = []
    for combo in args.combos:
        ns, sc, sm = combo.split(",")
        ns, sc, sm = int(ns), float(sc), float(sm)
        fname = build_filename(ns, sc, sm, ep=args.ep)
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[MISS] {fname}")
            continue
        cfgs = load_cfgs(fpath)
        mom = moments(cfgs)
        k_lat, G, E = diagonal_propagator(cfgs)
        xi = fit_xi(k_lat ** 2, G, n_low=5)
        rows.append(dict(
            label=f"({ns:>3},{sc:>4g},{sm:.0e})",
            ns=ns, sc=sc, sm=sm,
            mom=mom, k=k_lat, G=G, E=E, xi=xi,
        ))

    # Aggregate metric: weighted sum of |ratio - 1| for low-k modes + |xi - xi_GT|/xi_GT
    print(f"{'(steps,Sc,smin)':<22}  {'xi':>6} {'|dxi|':>7} {'|dG_kmin|%':>12} "
          f"{'|d<|M|>|%':>10} {'|d<phi2>|%':>11} {'|d<phi4>|%':>11} {'|dU4|':>8} {'score':>8}")
    print("=" * 110)
    best_score = float("inf")
    best_row = None
    for r in rows:
        dxi = abs(r['xi'] - gt_xi)
        dxi_pct = dxi / gt_xi * 100
        dG_min_pct = abs(r['G'][0] / gt_G[0] - 1) * 100
        dabsM = abs(r['mom']['absM'] / gt_mom['absM'] - 1) * 100
        dphi2 = abs(r['mom']['phi2'] / gt_mom['phi2'] - 1) * 100
        dphi4 = abs(r['mom']['phi4'] / gt_mom['phi4'] - 1) * 100
        dU4 = abs(r['mom']['binder'] - gt_mom['binder'])
        # Aggregate: equal weight on xi % deviation and avg low-k G ratio deviation
        low_k_ratio_dev = np.mean(np.abs(r['G'][:5] / gt_G[:5] - 1)) * 100
        score = 0.5 * dxi_pct + 0.5 * low_k_ratio_dev
        r['score'] = score
        if score < best_score:
            best_score = score
            best_row = r
        print(f"{r['label']:<22}  {r['xi']:>6.3f} {dxi_pct:>7.2f} {dG_min_pct:>11.2f} "
              f"{dabsM:>10.2f} {dphi2:>10.2f} {dphi4:>10.2f} {dU4:>8.4f} {score:>8.3f}")
    print("=" * 110)
    if best_row is not None:
        print(f"\nBEST by aggregate score (0.5 |dxi|% + 0.5 mean low-k |dG/G_GT|%):")
        print(f"  {best_row['label']}: score={best_row['score']:.3f}")

    # Plot ratio for each combo
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.viridis
    n = len(rows)
    for i, r in enumerate(rows):
        ratio = r['G'] / gt_G
        ratio_err = r['E'] / gt_G
        color = cmap(i / max(n - 1, 1))
        is_best = (best_row is not None and r is best_row)
        ax.errorbar(r['k'], ratio, yerr=ratio_err,
                    fmt="o-" if is_best else "o--",
                    color=color, lw=2 if is_best else 1.2,
                    ms=7 if is_best else 5,
                    capsize=2,
                    label=f"{r['label']}  xi={r['xi']:.2f}" + (" ★" if is_best else ""))
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G_{\rm EDM}(k) / G_{\rm GT}(k)$")
    ax.set_title(f"EDM sampling hp grid (L={args.L}, k={args.k}, ckpt ep={args.ep}); "
                 f"GT xi={gt_xi:.3f}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_ylim(0.85, 1.15)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
