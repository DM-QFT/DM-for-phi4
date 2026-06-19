"""
Overlay 2D acceptance-vs-epoch curves for kappa = 0.26, 0.2705, 0.28.

Usage (from 2Dphi4/):
    python acceptance_rate/plot_acceptance_vs_epoch_combined.py \
        --inputs \
            acceptance_rate/acceptance_vs_epoch_L128_k0.26_sigma300_ema.npz \
            acceptance_rate/acceptance_vs_epoch_L128_k0.2705_sigma450_ema.npz \
            acceptance_rate/acceptance_vs_epoch_L128_k0.28_sigma640_ema.npz \
        --out ../draft/figures/2Dphi4/acceptance_vs_epoch_L128_multi_kappa_ema.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PALETTE = {
    0.26:   "C0",
    0.2705: "C1",
    0.28:   "C2",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = [dict(np.load(f, allow_pickle=True)) for f in args.inputs]
    kappas = [float(d["k"]) for d in files]

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'lines.linewidth': 1.5, 'lines.markersize': 5,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for d, k in zip(files, kappas):
        color = PALETTE.get(k, "C7")
        ep = d["epochs"]
        ax.errorbar(ep, d["acc"], yerr=d["sem"],
                    marker='o', capsize=3, color=color,
                    label=rf'$\kappa={k}$ ($n={len(ep)}$ ckpts)')

    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Acceptance rate')
    L = int(files[0]["L"])
    c = float(files[0]["c"])
    t_mh = float(files[0]["t_mh"])
    ax.set_title(rf'$L={L},\;c={c},\;t_{{\mathrm{{mh}}}}={t_mh:.0e}$')
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.png')}")
    plt.close()


if __name__ == "__main__":
    main()
