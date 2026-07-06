"""
Merge worker NPZs from dm_mc_tau_convergence_3d.py and plot tau_int(n)
convergence curves, one per epoch, across all input files.

Usage:
  python analysis/plot_tau_convergence_3d.py \
      --inputs results/dm_mc_tau_convergence_3d_L16_N2000_part_*.npz \
      --out results/dm_mc_tau_convergence_3d_L16_N2000.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_ep = {}                      # {epoch: dict(accept, taus, n_grid)}
    meta = None
    for path in args.inputs:
        d = np.load(path)
        if meta is None:
            meta = dict(L=int(d["L"]), k=float(d["k"]), l=float(d["l"]),
                        c=float(d["c"]), t_mh=float(d["t_mh"]),
                        num_chains=int(d["num_chains"]),
                        mh_steps=int(d["mh_steps"]))
        n_grid = d["n_grid"]
        for ep in d["epochs"]:
            ep = int(ep)
            per_ep[ep] = dict(
                accept=float(d[f"ep{ep}_accept"]),
                taus=d[f"ep{ep}_taus"],
                n_grid=n_grid,
            )

    epochs = sorted(per_ep.keys())
    print(f"merged {len(epochs)} epochs: {epochs}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    cmap = plt.get_cmap("viridis")
    fig, (ax_tau, ax_ratio) = plt.subplots(1, 2, figsize=(13, 5))

    title_str = (rf"3D $\phi^4$ multi-L (train $L\in\{{4,8,16,32\}}$), "
                 rf"$\kappa={meta['k']},\;\lambda={meta['l']:g}$, $L={meta['L']}$")

    for i, ep in enumerate(epochs):
        col = cmap(i / max(1, len(epochs) - 1))
        r = per_ep[ep]
        lbl = rf"ep={ep}  (accept={r['accept']:.3f})"
        ax_tau.plot(r["n_grid"], r["taus"], marker="o", ms=4, color=col, label=lbl)
        ratio = r["n_grid"] / r["taus"]
        ax_ratio.plot(r["n_grid"], ratio, marker="o", ms=4, color=col, label=lbl)

    ax_tau.set_xlabel("chain length $n$ (MH steps used)")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(|M|)$")
    ax_tau.set_yscale("log")
    ax_tau.set_title(title_str)
    ax_tau.grid(alpha=0.3, which="both")
    ax_tau.legend(fontsize=9, loc="best")

    ax_ratio.axhline(5.0, color="k", lw=1, ls=":",
                     label=r"Sokal threshold $n/\tau\geq 5$")
    ax_ratio.set_xlabel("chain length $n$")
    ax_ratio.set_ylabel(r"$n / \tau_{\mathrm{int}}(n)$")
    ax_ratio.set_title(title_str)
    ax_ratio.grid(alpha=0.3, which="both")
    ax_ratio.legend(fontsize=9, loc="best")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
