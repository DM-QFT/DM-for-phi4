"""
Merge worker NPZs from dm_mc_tau_vs_epoch_3d.py and plot tau_int(|M|)
vs training epoch, one curve per L.

Usage:
  python analysis/plot_tau_vs_epoch_3d.py \
      --inputs results/dm_mc_tau_vs_epoch_3d_part_a.npz \
               results/dm_mc_tau_vs_epoch_3d_part_b.npz \
      --out results/dm_mc_tau_vs_epoch_3d_multiL.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Aggregate per-L results: { L: list of (epoch, tau, se, W, cut, acc) }
    per_L = {}
    meta = None
    for path in args.inputs:
        d = np.load(path)
        L_list = list(int(x) for x in d["L_list"])
        epochs = list(int(x) for x in d["epochs"])
        if meta is None:
            meta = dict(c=float(d["c"]), t_mh=float(d["t_mh"]),
                        k=float(d["k"]), l=float(d["l"]),
                        num_chains=int(d["num_chains"]),
                        mh_steps=int(d["mh_steps"]))
        for iL, L in enumerate(L_list):
            entries = per_L.setdefault(L, [])
            for ie, ep in enumerate(epochs):
                entries.append((ep,
                                float(d["tau_sokal"][ie, iL]),
                                float(d["tau_se"][ie, iL]),
                                int(d["sokal_W"][ie, iL]),
                                float(d["tau_cutoff"][ie, iL]),
                                float(d["accept"][ie, iL])))

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.5, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    cmap = plt.get_cmap("viridis")
    Ls = sorted(per_L.keys())

    print(f"{'L':>4s}  {'epoch':>6s}  {'accept':>7s}  "
          f"{'tau_sokal':>10s} {'+/-se':>6s}  {'W':>4s}  {'tau_cut':>8s}")
    print("-" * 60)

    title_str = (rf"3D $\phi^4$ multi-L (train $L\in\{{4,8,16,32\}}$), "
                 rf"$\kappa={meta['k']},\;\lambda={meta['l']:g}$")

    fig_tau, ax_tau = plt.subplots(figsize=(7, 5))
    fig_acc, ax_acc = plt.subplots(figsize=(7, 5))

    for i, L in enumerate(Ls):
        rows = sorted(per_L[L], key=lambda r: r[0])
        eps  = np.array([r[0] for r in rows])
        taus = np.array([r[1] for r in rows])
        ses  = np.array([r[2] for r in rows])
        Ws   = np.array([r[3] for r in rows])
        cuts = np.array([r[4] for r in rows])
        accs = np.array([r[5] for r in rows])
        col = cmap(i / max(1, len(Ls) - 1))
        ax_tau.errorbar(eps, taus, yerr=ses, marker="o", ms=5, capsize=2,
                        color=col, label=rf"$L={L}$")
        ax_acc.plot(eps, accs, marker="o", ms=5, color=col, label=rf"$L={L}$")
        for ep, tau, se, W, cut, acc in rows:
            print(f"{L:>4d}  {ep:>6d}  {acc:>7.3f}  "
                  f"{tau:>10.2f} {se:>6.2f}  {W:>4d}  {cut:>8.2f}")

    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel("Epoch")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(|M|)$  [MH steps]")
    ax_tau.set_title(title_str)
    ax_tau.grid(alpha=0.3, which="both")
    ax_tau.legend(loc="best")

    ax_acc.set_xscale("log")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("MALA acceptance rate")
    ax_acc.set_title(title_str)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.grid(alpha=0.3, which="both")
    ax_acc.legend(loc="best")

    fig_tau.tight_layout()
    fig_acc.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out_tau = out.with_name(out.stem + "_tau" + out.suffix)
    out_acc = out.with_name(out.stem + "_accept" + out.suffix)
    fig_tau.savefig(out_tau)
    fig_acc.savefig(out_acc)
    plt.close(fig_tau)
    plt.close(fig_acc)
    print(f"\nSaved tau figure   to {out_tau}")
    print(f"Saved accept figure to {out_acc}")

    # Save merged data.
    save = dict(L_list=np.array(Ls), **meta)
    for L in Ls:
        rows = sorted(per_L[L], key=lambda r: r[0])
        save[f"L{L}_epochs"] = np.array([r[0] for r in rows])
        save[f"L{L}_tau_sokal"] = np.array([r[1] for r in rows])
        save[f"L{L}_tau_se"] = np.array([r[2] for r in rows])
        save[f"L{L}_sokal_W"] = np.array([r[3] for r in rows])
        save[f"L{L}_tau_cutoff"] = np.array([r[4] for r in rows])
        save[f"L{L}_accept"] = np.array([r[5] for r in rows])
    np.savez(out.with_suffix(".npz"), **save)
    print(f"Saved merged data to {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
