"""
Overlay 3D score-quality vs epoch for kappa = 0.18, 0.1923, 0.2.

Usage (from 3Dphi4/):
    python score_quality_analysis/plot_score_quality_vs_epoch_combined.py \
        --inputs \
            score_quality_analysis/score_quality_vs_epoch_3d_L64_k0.18_sigma2000_ema.npz \
            score_quality_analysis/score_quality_vs_epoch_3d_L64_k0.1923_sigma2048_ema.npz \
            score_quality_analysis/score_quality_vs_epoch_3d_L64_k0.2_sigma2760_ema.npz \
        --out ../draft/figures/3Dphi4/score_quality_3d_L64_3panel_ema.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PALETTE = {0.18: "C0", 0.1923: "C1", 0.2: "C2"}


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
    fig.suptitle(rf'3D: $L={L},\;t_{{\mathrm{{eval}}}}={t_eval:.0e}$', y=1.02)
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
