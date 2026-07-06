"""
Overlay 3D dm_mc_tau_vs_c curves for kappa = 0.18, 0.1923, 0.2.
Two-panel: (left) tau_int(|M|) and tau_int(S), (right) ESS, vs c.

Usage (from 3Dphi4/):
    python analysis/plot_dm_mc_tau_vs_c_combined.py \
        --inputs \
            analysis/results/dm_mc_tau_vs_c_3d_L64_multiL_k0.18_ep20000_ema.npz \
            analysis/results/dm_mc_tau_vs_c_3d_L64_multiL_k0.1923_ep20000_ema.npz \
            analysis/results/dm_mc_tau_vs_c_3d_L64_multiL_k0.2_ep20000_ema.npz \
        --out ../draft/figures/3Dphi4/dm_mc_tau_vs_c_3d_L64_multi_kappa_ema.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
        'legend.fontsize': 10, 'lines.linewidth': 1.5, 'lines.markersize': 5,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for d, k in zip(files, kappas):
        color = PALETTE.get(k, "C7")
        c = d["c_list"]
        ax1.plot(c, d["tau_M_fit"],
                 marker='o', mfc='white', ls='--', color=color,
                 label=rf'$\kappa={k}$  $\tau_{{\mathrm{{fit}}}}(|\bar\phi|)$')
        ax1.plot(c, d["tau_S_fit"],
                 marker='s', ls='-', color=color,
                 label=rf'$\kappa={k}$  $\tau_{{\mathrm{{fit}}}}(S)$')
        ax2.plot(c, d["ess_M"], marker='o', mfc='white', ls='--', color=color,
                 label=rf'$\kappa={k}$  $\mathrm{{ESS}}_{{|\bar\phi|}}$')
        ax2.plot(c, d["ess_S"], marker='s', ls='-', color=color,
                 label=rf'$\kappa={k}$  $\mathrm{{ESS}}_S$')

    L = int(files[0]["L"])
    ep = int(files[0]["epoch"])
    t_mh = float(files[0]["t_mh"])
    for ax in (ax1, ax2):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$c\;(h = c / L^3)$')
        ax.grid(alpha=0.3)
    ax1.set_ylabel(r'$\tau_{\mathrm{int}}$')
    ax2.set_ylabel('ESS')
    ax1.legend(loc='best', ncol=1, fontsize=8)
    ax2.legend(loc='best', ncol=1, fontsize=8)
    fig.suptitle(rf'3D: $L={L},\;\text{{epoch}}={ep},\;t_{{\mathrm{{mh}}}}={t_mh:.0e}$', y=1.02)
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
