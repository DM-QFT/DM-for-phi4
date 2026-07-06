"""
Merge worker NPZs from dm_mc_S_vs_epoch_3d.py across multiple L (each L
using its own single-L in-distribution score model) and plot tau_int
vs epoch with one curve per L.

Usage:
  python analysis/plot_tau_vs_epoch_3d_indist.py \
      --inputs results/dm_mc_S_vs_epoch_3d_indist_L*.npz \
      --out_prefix results/dm_mc_S_vs_epoch_3d_indist
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    per_L = {}
    meta_common = None
    for path in args.inputs:
        d = np.load(path)
        L = int(d["L"])
        rows = per_L.setdefault(L, [])
        for ie, ep in enumerate(d["epochs"]):
            rows.append((int(ep),
                         float(d["accept"][ie]),
                         float(d["tau_S"][ie]), float(d["se_S"][ie]),
                         int(d["W_S"][ie]),
                         float(d["tau_M"][ie]), float(d["se_M"][ie]),
                         int(d["W_M"][ie])))
        if meta_common is None:
            meta_common = dict(k=float(d["k"]), l=float(d["l"]),
                               c=float(d["c"]), t_mh=float(d["t_mh"]),
                               num_chains=int(d["num_chains"]),
                               mh_steps=int(d["mh_steps"]))

    Ls = sorted(per_L.keys())
    print(f"L values found: {Ls}")
    for L in Ls:
        per_L[L] = sorted(set(per_L[L]), key=lambda r: r[0])
        print(f"  L={L}: {len(per_L[L])} epochs")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    title_str = (rf"3D $\phi^4$, "
                 rf"$\kappa={meta_common['k']},\;\lambda={meta_common['l']:g}$")
    cmap = plt.get_cmap("viridis")

    def make_plot(col_idx, ylabel, title_suffix, log_y, out_name):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for i, L in enumerate(Ls):
            color = cmap(i / max(1, len(Ls) - 1))
            rows = per_L[L]
            eps = np.array([r[0] for r in rows])
            ys = np.array([r[col_idx] for r in rows])
            if col_idx == 2:                   # tau_S
                yerr = np.array([r[3] for r in rows])
                ax.errorbar(eps, ys, yerr=yerr, marker="o", ms=4, capsize=2,
                            color=color, label=rf"$L={L}$")
            elif col_idx == 5:                 # tau_M
                yerr = np.array([r[6] for r in rows])
                ax.errorbar(eps, ys, yerr=yerr, marker="s", ms=4, capsize=2,
                            color=color, label=rf"$L={L}$")
            else:
                ax.plot(eps, ys, marker="o", ms=4, color=color, label=rf"$L={L}$")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0,
                           subs=(1.5, 2.5, 4.0, 6.0, 8.0, 9.0), numticks=60))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title_str + title_suffix)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best")
        fig.tight_layout()
        out = Path(args.out_prefix + out_name)
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    make_plot(2, r"$\tau_{\mathrm{int}}(S)$  [MH steps]", "", True, "_tau_S.pdf")
    make_plot(5, r"$\tau_{\mathrm{int}}(|M|)$  [MH steps]", "", True, "_tau_M.pdf")
    make_plot(1, "MALA acceptance rate", "", False, "_accept.pdf")

    # Sokal W/N saturation diagnostic (use tau_S column)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    N = meta_common["mh_steps"]
    for i, L in enumerate(Ls):
        color = cmap(i / max(1, len(Ls) - 1))
        rows = per_L[L]
        eps = np.array([r[0] for r in rows])
        Ws = np.array([r[4] for r in rows])
        ax.plot(eps, Ws / N, "o-", color=color, label=rf"$L={L}$")
    ax.axhline(0.2, color="k", ls=":", lw=1, label="convergence ceiling 0.2")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$W^* / N$ (Sokal window for $S$)")
    ax.set_title(title_str)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = Path(args.out_prefix + "_W_S_ratio.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Also print the table.
    print(f"\n{'L':>4s}  {'epoch':>6s}  {'accept':>7s}  "
          f"{'tau_S':>8s}  {'W_S':>5s}  {'tau_|M|':>8s}  {'W_M':>5s}")
    print("-" * 64)
    for L in Ls:
        for r in per_L[L]:
            print(f"{L:>4d}  {r[0]:>6d}  {r[1]:>7.3f}  "
                  f"{r[2]:>8.2f}  {r[4]:>5d}  "
                  f"{r[5]:>8.2f}  {r[7]:>5d}")

    save = dict(**meta_common)
    for L in Ls:
        rows = per_L[L]
        save[f"L{L}_epochs"] = np.array([r[0] for r in rows])
        save[f"L{L}_accept"] = np.array([r[1] for r in rows])
        save[f"L{L}_tau_S"] = np.array([r[2] for r in rows])
        save[f"L{L}_se_S"] = np.array([r[3] for r in rows])
        save[f"L{L}_W_S"] = np.array([r[4] for r in rows])
        save[f"L{L}_tau_M"] = np.array([r[5] for r in rows])
        save[f"L{L}_se_M"] = np.array([r[6] for r in rows])
        save[f"L{L}_W_M"] = np.array([r[7] for r in rows])
    np.savez(Path(args.out_prefix + ".npz"), **save)
    print(f"Saved merged data to {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
