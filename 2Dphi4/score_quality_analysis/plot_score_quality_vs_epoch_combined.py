"""
Overlay 2D score-quality (cos_sim, rel_mse, mag_ratio) vs epoch for
kappa = 0.26, 0.2705, 0.28.

Usage (from 2Dphi4/):
    python score_quality_analysis/plot_score_quality_vs_epoch_combined.py \
        --inputs \
            score_quality_analysis/score_quality_vs_epoch_L128_k0.26_sigma300_ema.npz \
            score_quality_analysis/score_quality_vs_epoch_L128_k0.2705_sigma450_ema.npz \
            score_quality_analysis/score_quality_vs_epoch_L128_k0.28_sigma640_ema.npz \
        --out ../draft/figures/2Dphi4/score_quality_vs_epoch_L128_3panel_ema.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PALETTE = {0.26: "C0", 0.2705: "C1", 0.28: "C2"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = [dict(np.load(f, allow_pickle=True)) for f in args.inputs]
    kappas = [float(d["k"]) for d in files]

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'lines.linewidth': 1.2, 'lines.markersize': 4,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ('cos_sim', 'cos_sim_sem', 'Cosine similarity'),
        ('rel_mse', 'rel_mse_sem', 'Relative MSE'),
        ('mag_ratio', 'mag_ratio_sem', 'Magnitude ratio'),
    ]
    for ax, (mkey, skey, mlabel) in zip(axes, metrics):
        for d, k in zip(files, kappas):
            color = PALETTE.get(k, "C7")
            ax.errorbar(d["epochs"], d[mkey], yerr=d[skey],
                        marker='o', capsize=2, color=color,
                        label=rf'$\kappa={k}$')
        ax.set_xscale('log')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel(mlabel)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')

    axes[2].axhline(1.0, color='gray', ls='--', lw=0.8)

    L = int(files[0]["L"])
    t_eval = float(files[0]["t_eval"])
    fig.suptitle(rf'$L={L},\;t_{{\mathrm{{eval}}}}={t_eval:.0e}$', y=1.02)
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
