"""
Compare multi-L EDM2 samples against Wolff+FAHMC ground truth across
L ∈ {8, 16, 32, 64, 128}. Computes moments + propagator G(k) + correlation
length xi for each L, plots and tables them all.

This is the Phase 4c analysis for the multi-L training experiment.

Usage (from 2Dphi4/):
    python analysis/compare_edm2_multiL_propagator.py \
        --run_dir runs/phi4_Lmulti8-16-32-64_k0.28_l0.022_edm2_sd0.343 \
        --ep 5000
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
    nd = np.arange(1, L // 2 + 1)
    g = pk[:, nd, nd]
    k_lat = np.sqrt(8.0 * np.sin(np.pi * nd / L) ** 2)
    rng = np.random.default_rng(seed)
    b = np.empty((n_boot, g.shape[1]))
    for i in range(n_boot):
        b[i] = g[rng.integers(0, N, size=N)].mean(axis=0)
    return k_lat, b.mean(axis=0), b.std(axis=0)


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
        absM=float(np.abs(M).mean()),
        phi2=float((cfgs ** 2).mean()),
        phi4=float((cfgs ** 4).mean()),
        binder=1.0 - float((M ** 4).mean()) / (3 * float((M ** 2).mean()) ** 2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ep", type=str, default="5000")
    p.add_argument("--L_list", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128])
    p.add_argument("--training_Ls", type=int, nargs="+",
                   default=[8, 16, 32, 64],
                   help="L values used during training (for marking in-dist vs OOD)")
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--num_steps", type=int, default=36)
    p.add_argument("--S_churn", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gt_prefix", type=str,
                   default="trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = run_dir / "data"
    if args.out is None:
        args.out = str(data_dir / f"multiL_compare_ep{args.ep}.pdf")

    # Find generated sample filename pattern.
    # Format: samples_edm_ep{EP}_N{N}_steps{S}{churn}{smin}{cross_L}_seed{seed}.h5
    rows = []
    print(f"{'L':>4}  {'in/oo':<5}  {'N_sample':>9}  {'<|M|>':>8}  {'<phi^2>':>8}  "
          f"{'<phi^4>':>9}  {'U_4':>8}  {'xi_gt':>6} {'xi_dm':>6}  "
          f"{'|dxi|%':>7}  {'G_gt':>8} {'G_dm':>8}  {'|dG_kmin|%':>11}")
    print("=" * 130)
    for L in args.L_list:
        gt_path = Path(f"{args.gt_prefix}_k={args.k}_l={args.l}_{L}^2.jld2")
        if not gt_path.exists():
            print(f"[WARN] no GT for L={L}: {gt_path}")
            continue
        gt = load_cfgs(gt_path)
        # Locate generated sample
        sc_tag = f"_Sc{args.S_churn:g}" if args.S_churn > 0 else ""
        # If L matches training, no cross_L tag; if not, has _L{L}.
        # But for multi-L training, "trained at" is ambiguous. Use heuristic:
        # check the existence of either filename.
        cand1 = data_dir / (f"samples_edm_ep{args.ep}_N*_steps{args.num_steps}"
                            f"{sc_tag}_seed{args.seed}.h5")  # L matches ckpt L
        cand2 = data_dir / (f"samples_edm_ep{args.ep}_N*_steps{args.num_steps}"
                            f"{sc_tag}_L{L}_seed{args.seed}.h5")
        matches = sorted(data_dir.glob(cand2.name))
        if not matches:
            matches = sorted(data_dir.glob(cand1.name))
            # filter to current L by reading attrs
            matches = [m for m in matches if _file_L(m) == L]
        if not matches:
            print(f"[MISS] no sample for L={L} in {data_dir}")
            continue
        sample_path = matches[-1]
        cfgs = load_cfgs(sample_path)
        m_gt = moments(gt); m_dm = moments(cfgs)
        k, G_gt, _ = diagonal_propagator(gt)
        _, G_dm, E_dm = diagonal_propagator(cfgs)
        xi_gt = fit_xi(k ** 2, G_gt, 5)
        xi_dm = fit_xi(k ** 2, G_dm, 5)
        in_dist = L in args.training_Ls
        dxi = abs(xi_dm - xi_gt) / xi_gt * 100
        dGkmin = abs(G_dm[0] - G_gt[0]) / G_gt[0] * 100
        rows.append(dict(L=L, in_dist=in_dist, gt=gt, cfgs=cfgs,
                         m_gt=m_gt, m_dm=m_dm,
                         k=k, G_gt=G_gt, G_dm=G_dm, E_dm=E_dm,
                         xi_gt=xi_gt, xi_dm=xi_dm))
        print(f"{L:>4}  {'in' if in_dist else 'OOD':<5}  {cfgs.shape[0]:>9}  "
              f"{m_dm['absM']:>8.4f}  {m_dm['phi2']:>8.4f}  {m_dm['phi4']:>9.4f}  "
              f"{m_dm['binder']:>8.4f}  {xi_gt:>6.3f} {xi_dm:>6.3f}  "
              f"{dxi:>6.2f}%  {G_gt[0]:>8.4f} {G_dm[0]:>8.4f}  {dGkmin:>10.2f}%")
    print("=" * 130)

    # Plot propagator across L's
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap_in = plt.cm.viridis
    cmap_ood = plt.cm.autumn
    n_L = len(rows)

    ax = axes[0]
    for i, r in enumerate(rows):
        color = cmap_in(i / max(n_L - 1, 1)) if r['in_dist'] else "red"
        lbl_pref = "L=" + str(r['L'])
        lbl_tag = "" if r['in_dist'] else " (OOD)"
        ax.errorbar(r['k'], r['G_gt'], fmt="o-", color=color, alpha=0.5, ms=4,
                    label=f"{lbl_pref} GT" + lbl_tag)
        ax.errorbar(r['k'], r['G_dm'], yerr=r['E_dm'], fmt="s--", color=color,
                    ms=5, capsize=2,
                    label=f"{lbl_pref} EDM2  xi={r['xi_dm']:.2f}" + lbl_tag)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\hat k|$ (lattice units)")
    ax.set_ylabel(r"$G(k)$")
    ax.set_title("Multi-L EDM2 vs Wolff+FAHMC: propagators across L")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    ax = axes[1]
    for i, r in enumerate(rows):
        color = cmap_in(i / max(n_L - 1, 1)) if r['in_dist'] else "red"
        ratio = r['G_dm'] / r['G_gt']
        ax.errorbar(r['k'], ratio, yerr=r['E_dm'] / r['G_gt'],
                    fmt="o-" if r['in_dist'] else "s--",
                    color=color, lw=1.5, ms=6, capsize=2,
                    label=f"L={r['L']}" + (" (OOD)" if not r['in_dist'] else ""))
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat k|$"); ax.set_ylabel(r"$G_{\rm EDM2}/G_{\rm GT}$")
    ax.set_title("Relative deviation per L")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_ylim(0.7, 1.3)

    fig.suptitle(f"Multi-L EDM2 (trained on L={args.training_Ls}), "
                 f"ckpt ep={args.ep}", y=1.02, fontsize=12)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


def _file_L(path: Path) -> int:
    """Read L attribute from H5 file, else infer from shape."""
    try:
        with h5py.File(path, "r") as f:
            if "L" in f.attrs:
                return int(f.attrs["L"])
            return int(f["cfgs"].shape[-1])
    except Exception:
        return -1


if __name__ == "__main__":
    main()
