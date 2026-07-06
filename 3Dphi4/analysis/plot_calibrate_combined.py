"""
Combine the per-kappa .npz outputs of analysis/calibrate_step_size.py /
analysis/calibrate_t_mh.py into a single 3D figure overlaying
kappa=0.18, 0.1923, 0.2.  Mirrors 2Dphi4/acceptance_rate/plot_calibrate_combined.py.

Usage (from 3Dphi4/):
    python analysis/plot_calibrate_combined.py --kind h \
        --inputs analysis/calibrate_h_3d_L64_k0.18_sigma2000.npz \
                 analysis/calibrate_h_3d_L64_k0.1923_sigma2048.npz \
                 analysis/calibrate_h_3d_L64_k0.2_sigma2760.npz \
        --out ../draft/figures/3Dphi4/calibrate_h_3d_L64_multi_kappa.pdf

    python analysis/plot_calibrate_combined.py --kind t_mh \
        --inputs analysis/calibrate_t_mh_3d_L64_k0.18_sigma2000.npz \
                 analysis/calibrate_t_mh_3d_L64_k0.1923_sigma2048.npz \
                 analysis/calibrate_t_mh_3d_L64_k0.2_sigma2760.npz \
        --out ../draft/figures/3Dphi4/calibrate_t_mh_3d_L64_multi_kappa.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PALETTE = {
    0.18:   "C0",   # blue
    0.1923: "C1",   # orange
    0.2:    "C2",   # green
}


def _sci(v):
    e = int(np.floor(np.log10(v)))
    m = v / 10**e
    return rf'10^{{{e}}}' if m == 1 else rf'{m:.0f}\!\times\!10^{{{e}}}'


def _load(path):
    return dict(np.load(path, allow_pickle=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["h", "t_mh"])
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = [_load(f) for f in args.inputs]
    kappas = [float(d["k"]) for d in files]

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ----- panel 1: acceptance vs (c or t_mh), overlay kappas -----
    for d, k in zip(files, kappas):
        color = PALETTE.get(k, "C7")
        if args.kind == "h":
            x = d["c_values"]
        else:
            x = d["t_mh_values"]
        nE = int(d["early_epochs"].size)
        nL = int(d["late_epochs"].size)
        ax1.errorbar(x, d["acc_early"], yerr=d["err_early"],
                     marker='o', mfc='white', ls='--',
                     capsize=3, color=color,
                     label=rf'$\kappa={k}$ early ($n={nE}$)')
        ax1.errorbar(x, d["acc_late"], yerr=d["err_late"],
                     marker='s', ls='-',
                     capsize=3, color=color,
                     label=rf'$\kappa={k}$ late ($n={nL}$)')

    ax1.set_xscale('log')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('Acceptance rate')
    L = int(files[0]["L"])
    if args.kind == "h":
        ax1.set_xlabel(r'$c\;(h = c / L^3)$')
        t_mh = float(files[0]["t_mh"])
        ax1.set_title(rf'3D: $L={L},\;t_{{\mathrm{{mh}}}}={_sci(t_mh)}$')
    else:
        ax1.set_xlabel(r'$t_{\mathrm{mh}}$')
        c = float(files[0]["c"])
        ax1.set_title(rf'3D: $L={L},\;c={c}$')
    ax1.legend(loc='upper right', framealpha=0.9, ncol=1)

    # ----- panel 2: grouped gap bars -----
    if args.kind == "h":
        x = files[0]["c_values"]
        tick_labels = [f'{c:.2g}' for c in x]
        xlabel = r'$c$'
    else:
        x = files[0]["t_mh_values"]
        tick_labels = []
        for t in x:
            e = int(np.floor(np.log10(t)))
            m = t / 10**e
            tick_labels.append(rf'$10^{{{e}}}$' if m == 1 else rf'${m:.0f}\!\times\!10^{{{e}}}$')
        xlabel = r'$t_{\mathrm{mh}}$'

    n_kappa = len(files)
    width = 0.8 / n_kappa
    idx = np.arange(len(x))
    for i, (d, k) in enumerate(zip(files, kappas)):
        color = PALETTE.get(k, "C7")
        offset = (i - (n_kappa - 1) / 2) * width
        ax2.bar(idx + offset, d["gap"], width=width,
                color=color, label=rf'$\kappa={k}$',
                edgecolor='black', linewidth=0.4)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    if args.kind == "t_mh":
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend(loc='best', framealpha=0.9)

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
