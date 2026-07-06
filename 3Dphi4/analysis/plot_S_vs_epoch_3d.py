"""
Merge worker NPZs from dm_mc_S_vs_epoch_3d.py and plot tau_int(S) and
tau_int(|M|) vs training epoch.

Usage:
  python analysis/plot_S_vs_epoch_3d.py \
      --inputs results/dm_mc_S_vs_epoch_3d_L16_part_*.npz \
      --out_prefix results/dm_mc_S_vs_epoch_3d_L16
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    rows = []                    # list of (ep, accept, tau_S, se_S, W_S, cut_S, tau_M, ...)
    meta = None
    for path in args.inputs:
        d = np.load(path)
        if meta is None:
            meta = dict(L=int(d["L"]), k=float(d["k"]), l=float(d["l"]),
                        c=float(d["c"]), t_mh=float(d["t_mh"]),
                        num_chains=int(d["num_chains"]),
                        mh_steps=int(d["mh_steps"]))
        for ie, ep in enumerate(d["epochs"]):
            rows.append((int(ep),
                         float(d["accept"][ie]),
                         float(d["tau_S"][ie]), float(d["se_S"][ie]),
                         int(d["W_S"][ie]), float(d["cut_S"][ie]),
                         float(d["tau_M"][ie]), float(d["se_M"][ie]),
                         int(d["W_M"][ie]), float(d["cut_M"][ie])))
    rows.sort()
    eps = np.array([r[0] for r in rows])
    acc = np.array([r[1] for r in rows])
    tau_S = np.array([r[2] for r in rows])
    se_S = np.array([r[3] for r in rows])
    W_S = np.array([r[4] for r in rows])
    cut_S = np.array([r[5] for r in rows])
    tau_M = np.array([r[6] for r in rows])
    se_M = np.array([r[7] for r in rows])
    W_M = np.array([r[8] for r in rows])
    cut_M = np.array([r[9] for r in rows])

    print(f"{'epoch':>6s}  {'accept':>7s}  "
          f"{'tau_S':>8s} {'+/-':>6s} {'W_S':>4s}  "
          f"{'tau_|M|':>8s} {'+/-':>6s} {'W_M':>4s}")
    print("-" * 65)
    for r in rows:
        print(f"{r[0]:>6d}  {r[1]:>7.3f}  "
              f"{r[2]:>8.2f} {r[3]:>6.2f} {r[4]:>4d}  "
              f"{r[6]:>8.2f} {r[7]:>6.2f} {r[8]:>4d}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    title_str = (rf"3D $\phi^4$ multi-L (train $L\in\{{4,8,16,32\}}$), "
                 rf"$L={meta['L']},\;\kappa={meta['k']},\;\lambda={meta['l']:g}$")

    # Combined tau plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(eps, tau_S, yerr=se_S, marker="o", ms=5, capsize=2,
                color="C0", label=r"$\tau_{\rm int}(S)$")
    ax.errorbar(eps, tau_M, yerr=se_M, marker="s", ms=5, capsize=2,
                color="C3", label=r"$\tau_{\rm int}(|M|)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$  [MH steps]")
    ax.set_title(title_str)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out_tau = Path(args.out_prefix + "_tau.pdf")
    fig.savefig(out_tau)
    plt.close(fig)
    print(f"\nSaved tau figure to {out_tau}")

    # Acceptance plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eps, acc, "o-", color="C2", ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MALA acceptance rate")
    ax.set_title(title_str)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_acc = Path(args.out_prefix + "_accept.pdf")
    fig.savefig(out_acc)
    plt.close(fig)
    print(f"Saved acceptance figure to {out_acc}")

    # Sokal W/N diagnostic plot
    fig, ax = plt.subplots(figsize=(7, 5))
    N = meta["mh_steps"]
    ax.plot(eps, W_S / N, "o-", color="C0", label=r"$W^*/N$ for $S$")
    ax.plot(eps, W_M / N, "s-", color="C3", label=r"$W^*/N$ for $|M|$")
    ax.axhline(0.2, color="k", ls=":", lw=1, label="convergence ceiling 0.2")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$W^* / N$")
    ax.set_title(title_str + r" (Sokal window saturation)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out_W = Path(args.out_prefix + "_W_ratio.pdf")
    fig.savefig(out_W)
    plt.close(fig)
    print(f"Saved Sokal-window figure to {out_W}")

    np.savez(Path(args.out_prefix + ".npz"),
             epochs=eps, accept=acc,
             tau_S=tau_S, se_S=se_S, W_S=W_S, cut_S=cut_S,
             tau_M=tau_M, se_M=se_M, W_M=W_M, cut_M=cut_M,
             **meta)
    print(f"Saved merged data to {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
