"""
Compute DM.tex-style observables and propagator-mismatch metrics for the k=0.2
multi-L L=[4,8,16,32] training-variant sweep at L=64.

Per variant:
  Observables (DM.tex Sec. 'Observables'):
    ⟨|φ̄|⟩, χ = V (⟨|φ̄|²⟩ - ⟨|φ̄|⟩²), U_L = 1 - ⟨|φ̄|⁴⟩ / (3 ⟨|φ̄|²⟩²)
    with jackknife errors.
  Propagator G(|k_diag|) with bootstrap errors.
  Propagator-mismatch (DM vs HMC, DM.tex Sec. 'Quantitative propagator-mismatch'):
    D_k = ½(r_k - 1 - log r_k),  r_k = G_train/G_DM
    z_k = (G_DM - G_train) / √(σ_DM² + σ_train²)
    w_k = k^(D-1) (G_DM - G_train),  D=3
    D_tot = Σ D_k,  L1^pw = Σ k^(D-1) |G_DM - G_train|,  R_var = ΣG_DM / ΣG_train

Output: summary CSV + 3-panel figure (D_k, z_k, w_k) per schedule.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)

D_DIM = 3


def jackknife_observables(cfgs):
    """Return (⟨|M|⟩, χ, U_L) and their jackknife stderrs.

    cfgs: (N, L, L, L)
    """
    L = cfgs.shape[1]
    V = L ** D_DIM
    M = cfgs.mean(axis=(1, 2, 3))           # (N,) per-cfg magnetization
    absM = np.abs(M)
    M2 = M ** 2
    M4 = M ** 4

    def stats(absM_, M2_, M4_):
        a1 = absM_.mean()
        a2 = M2_.mean()
        a4 = M4_.mean()
        chi = V * (a2 - a1 ** 2)
        U = 1.0 - a4 / (3.0 * a2 ** 2)
        return a1, chi, U

    a1, chi, U = stats(absM, M2, M4)
    N = len(M)
    # Block jackknife with 32 blocks for speed (N=512 → 16/block)
    n_blocks = min(32, N)
    block_size = N // n_blocks
    jk = np.empty((n_blocks, 3))
    for b in range(n_blocks):
        m = np.ones(N, dtype=bool)
        m[b * block_size:(b + 1) * block_size] = False
        jk[b] = stats(absM[m], M2[m], M4[m])
    err = np.sqrt((n_blocks - 1) / n_blocks * ((jk - jk.mean(0)) ** 2).sum(0))
    return (a1, chi, U), tuple(err)


def propagator_mismatch(k_lat, G_dm, E_dm, G_hmc, E_hmc):
    """Per-mode and aggregate metrics."""
    r = G_hmc / G_dm
    D_k = 0.5 * (r - 1.0 - np.log(r))
    z_k = (G_dm - G_hmc) / np.sqrt(E_dm ** 2 + E_hmc ** 2)
    w_k = (k_lat ** (D_DIM - 1)) * (G_dm - G_hmc)
    D_tot = D_k.sum()
    L1_pw = (k_lat ** (D_DIM - 1) * np.abs(G_dm - G_hmc)).sum()
    R_var = G_dm.sum() / G_hmc.sum()
    return D_k, z_k, w_k, D_tot, L1_pw, R_var


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="linear",
                   choices=["log", "linear"])
    args = p.parse_args()
    K = 0.2
    L = 64
    sx = f"em_{args.schedule}_steps{args.num_steps}"

    f_hmc = Path(f"trainingdata/cfgs_wolff_fahmc_k={K}_l={args.lam}_{L}^3.jld2")
    if not f_hmc.exists():
        raise FileNotFoundError(f_hmc)
    hmc = load_hmc_3d(f_hmc, max_n=4096)

    paths = [
        ("L=[4,8] σ=100",          "C1", "s",
         f"runs/phi4_3d_Lmulti4-8_k{K}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train8_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16] σ=280",       "C2", "^",
         f"runs/phi4_3d_Lmulti4-8-16_k{K}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train16_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32] σ=800",    "C3", "D",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32] σ=400",    "C4", "P",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_sigma400/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32] σ=200",    "C5", "*",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_sigma200/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32] σ=100",    "C6", "v",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_sigma100/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32] σ=800 bs256 ep20k", "C7", "X",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_bs256/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep20000.npy"),
    ]

    cases = [("HMC L=64", "C0", "o", hmc)]
    for name, color, marker, path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[skip] {name}: {path}")
            continue
        cases.append((name, color, marker, load_dm_3d(path)))

    # HMC propagator (the train reference)
    k_lat, G_hmc, E_hmc = diagonal_propagator_3d(hmc)

    print(f"\n=== κ=0.2, λ={args.lam}, L={L}, schedule={args.schedule} ===\n")
    hdr = f"{'variant':>34} {'⟨|M|⟩':>13} {'χ':>11} {'U_L':>11} {'G(k_min)':>14} {'D_tot':>9} {'L1^pw':>9} {'R_var':>7}"
    print(hdr); print("-" * len(hdr))

    rows = []
    spec = {}  # for plotting
    for name, color, marker, d in cases:
        (a1, chi, U), (e_a1, e_chi, e_U) = jackknife_observables(d)
        kk, G, E = diagonal_propagator_3d(d)
        if name == "HMC L=64":
            row = (name, a1, e_a1, chi, e_chi, U, e_U, G[0], E[0],
                   0.0, 0.0, 1.0, None, None, None, color, marker)
            spec[name] = (color, marker, kk, G, E, None, None, None)
        else:
            D_k, z_k, w_k, D_tot, L1_pw, R_var = propagator_mismatch(
                kk, G, E, G_hmc, E_hmc)
            row = (name, a1, e_a1, chi, e_chi, U, e_U, G[0], E[0],
                   D_tot, L1_pw, R_var, D_k, z_k, w_k, color, marker)
            spec[name] = (color, marker, kk, G, E, D_k, z_k, w_k)
        rows.append(row)
        print(f"{name:>34} {a1:8.4f}±{e_a1:5.4f} {chi:5.2f}±{e_chi:5.2f} "
              f"{U:6.4f}±{e_U:5.4f} {G[0]:7.3f}±{E[0]:5.3f} "
              f"{row[9]:9.4f} {row[10]:9.4f} {row[11]:7.4f}")

    # CSV
    out_csv = Path(f"results/L{L}_crossL_3d_k{K}_metrics_em_{args.schedule}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("variant,abs_M,abs_M_err,chi,chi_err,U_L,U_L_err,"
                "G_kmin,G_kmin_err,D_tot,L1_pw,R_var\n")
        for r in rows:
            f.write(",".join([r[0]] +
                             [f"{x:.6g}" for x in r[1:12]]) + "\n")
    print(f"\nSaved CSV: {out_csv}")

    # 3-panel figure: D_k, z_k, w_k vs |k_lat|
    dm_names = [n for n, *_ in spec.items() if n != "HMC L=64"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax_D, ax_z, ax_w = axes

    for name in dm_names:
        color, marker, kk, G, E, D_k, z_k, w_k = spec[name]
        ax_D.plot(kk, D_k, marker + "-", color=color, lw=1.4, ms=5, label=name)
        ax_z.plot(kk, z_k, marker + "-", color=color, lw=1.4, ms=5, label=name)
        ax_w.plot(kk, w_k, marker + "-", color=color, lw=1.4, ms=5, label=name)

    ax_D.set_yscale("log")
    ax_D.set_xscale("log")
    ax_D.set_xlabel(r"$|\hat k|$"); ax_D.set_ylabel(r"$D_k$  (per-mode KL)")
    ax_D.set_title(r"Per-mode Gaussian KL  $D_k = \frac{1}{2}(r-1-\log r)$,  $r=G_{\rm HMC}/G_{\rm DM}$")
    ax_D.grid(alpha=0.3, which="both"); ax_D.legend(fontsize=7.5, loc="best")

    ax_z.axhspan(-2, 2, color="0.85", alpha=0.5, zorder=0)
    ax_z.axhline(0, color="k", lw=0.7)
    ax_z.set_xscale("log")
    ax_z.set_xlabel(r"$|\hat k|$"); ax_z.set_ylabel(r"$z_k$  (statistical residual)")
    ax_z.set_title(r"$z_k = (G_{\rm DM}-G_{\rm HMC})/\sqrt{\sigma_{\rm DM}^2+\sigma_{\rm HMC}^2}$")
    ax_z.grid(alpha=0.3, which="both"); ax_z.legend(fontsize=7.5, loc="best")

    ax_w.axhline(0, color="k", lw=0.7)
    ax_w.set_xscale("log")
    ax_w.set_xlabel(r"$|\hat k|$"); ax_w.set_ylabel(r"$w_k = k^{D-1}\,(G_{\rm DM}-G_{\rm HMC})$,  $D=3$")
    ax_w.set_title("Phase-space-weighted propagator difference")
    ax_w.grid(alpha=0.3, which="both"); ax_w.legend(fontsize=7.5, loc="best")

    fig.suptitle(
        f"3D φ⁴ L={L} cross-L OOD,  κ={K}, λ={args.lam},  EM/{args.schedule}/steps={args.num_steps}",
        y=1.0, fontsize=13)
    fig.tight_layout()

    out = Path(f"results/L{L}_crossL_3d_k{K}_propmetrics_em_{args.schedule}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
