"""
Replot a calibration figure (h or t_mh) from a saved .npz produced by
`calibrate_step_size.py` or `calibrate_t_mh.py`. No GPU needed.

Usage:
    python analysis/replot_calibration.py results/calibration/calibrate_h_3d_L64_k0.2.npz
    python analysis/replot_calibration.py results/calibration/calibrate_t_mh_3d_L64_k0.2.npz \\
        --useful_hi 0.90
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def replot_h(npz_path, out_pdf, useful_lo, useful_hi):
    d = np.load(npz_path, allow_pickle=True)
    c_values = d["c_values"]
    acc_early = d["acc_early"]
    acc_late = d["acc_late"]
    err_early = d["err_early"] if "err_early" in d.files else np.zeros_like(acc_early)
    err_late = d["err_late"] if "err_late" in d.files else np.zeros_like(acc_late)
    gap = d["gap"]
    sigma_ve = float(d["sigma_ve"])
    t_mh = float(d["t_mh"])
    noise_scale_at_tmh = float(d["noise_scale_at_tmh"])
    early_eps = d["early_epochs"]
    late_eps = d["late_epochs"]

    useful = (acc_late > useful_lo) & (acc_late < useful_hi)
    best_idx = (np.where(useful, gap, -np.inf).argmax()
                if useful.any() else gap.argmax())
    print(f"Recommended:  c = {c_values[best_idx]:.2f}")
    print(f"  early accept = {acc_early[best_idx]:.4f}")
    print(f"  late  accept = {acc_late[best_idx]:.4f}")
    print(f"  gap          = {gap[best_idx]:+.4f}")

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    early_lo = int(np.min(early_eps))
    early_hi = int(np.max(early_eps))
    late_lo = int(np.min(late_eps))
    late_hi = int(np.max(late_eps))
    ax1.errorbar(c_values, acc_early, yerr=err_early,
                 label=f'Early epochs {early_lo}--{early_hi} '
                       f'($n={len(early_eps)}$)',
                 marker='o', capsize=3, color='C3')
    ax1.errorbar(c_values, acc_late, yerr=err_late,
                 label=f'Late epochs {late_lo}--{late_hi} '
                       f'($n={len(late_eps)}$)',
                 marker='s', capsize=3, color='C0')
    ax1.axvline(c_values[best_idx], ls='--', color='gray', alpha=0.6,
                label=f'Recommended $c={c_values[best_idx]:.2g}$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$c\;(h = c / L^3)$')
    ax1.set_ylabel('Acceptance rate')
    ax1.set_title(rf'3D: $L=64,\;\kappa=0.2,\;\sigma(t_{{\mathrm{{mh}}}})\approx{noise_scale_at_tmh:.2g}$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    colors = ['C2' if useful[i] else 'C7' for i in range(len(c_values))]
    ax2.bar(range(len(c_values)), gap,
            tick_label=[f'{c:.2g}' for c in c_values], color=colors)
    ax2.set_xlabel(r'$c$')
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"Saved {out_pdf}")
    plt.close()


def replot_tmh(npz_path, out_pdf, useful_lo, useful_hi):
    d = np.load(npz_path, allow_pickle=True)
    t_mh_values = d["t_mh_values"]
    noise_scale_values = d["noise_scale_values"]
    acc_early = d["acc_early"]
    acc_late = d["acc_late"]
    gap = d["gap"]
    sigma_ve = float(d["sigma_ve"])
    early_eps = d["early_epochs"]
    late_eps = d["late_epochs"]
    early_per_ep = d["early_per_ep"]
    late_per_ep = d["late_per_ep"]

    err_early = (early_per_ep.std(axis=0, ddof=1) / np.sqrt(len(early_eps))
                 if len(early_eps) > 1 else np.zeros_like(acc_early))
    err_late = (late_per_ep.std(axis=0, ddof=1) / np.sqrt(len(late_eps))
                if len(late_eps) > 1 else np.zeros_like(acc_late))

    useful = (acc_late > useful_lo) & (acc_late < useful_hi)
    best_idx = (np.where(useful, gap, -np.inf).argmax()
                if useful.any() else gap.argmax())
    print(f"Recommended t_mh = {t_mh_values[best_idx]:.1e}  "
          f"(sigma(t) = {noise_scale_values[best_idx]:.3e})")
    print(f"  early accept = {acc_early[best_idx]:.4f}")
    print(f"  late  accept = {acc_late[best_idx]:.4f}")
    print(f"  gap          = {gap[best_idx]:+.4f}")

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    early_lo = int(np.min(early_eps))
    early_hi = int(np.max(early_eps))
    late_lo = int(np.min(late_eps))
    late_hi = int(np.max(late_eps))
    ax1.errorbar(noise_scale_values, acc_early, yerr=err_early,
                 label=f'Early epochs {early_lo}--{early_hi} '
                       f'($n={len(early_eps)}$)',
                 marker='o', capsize=3, color='C3')
    ax1.errorbar(noise_scale_values, acc_late, yerr=err_late,
                 label=f'Late epochs {late_lo}--{late_hi} '
                       f'($n={len(late_eps)}$)',
                 marker='s', capsize=3, color='C0')
    ax1.axvline(noise_scale_values[best_idx], ls='--', color='gray', alpha=0.6,
                label=rf'Recommended $\sigma(t_{{\mathrm{{mh}}}})\approx'
                      rf'{noise_scale_values[best_idx]:.2g}$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'noise scale $\sigma(t_{\mathrm{mh}})$')
    ax1.set_ylabel('Acceptance rate')
    ax1.set_title(rf'3D: $L=64,\;\kappa=0.2$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    bar_labels = []
    for s in noise_scale_values:
        e = int(np.floor(np.log10(s)))
        m = s / 10**e
        bar_labels.append(rf'${m:.1f}\!\times\!10^{{{e}}}$')
    colors = ['C2' if useful[i] else 'C7' for i in range(len(noise_scale_values))]
    ax2.bar(range(len(noise_scale_values)), gap,
            tick_label=bar_labels, color=colors)
    ax2.set_xlabel(r'noise scale $\sigma(t_{\mathrm{mh}})$')
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), fontsize=9, rotation=30, ha='right')

    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"Saved {out_pdf}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=str, help="Path to npz produced by calibrate_*.py")
    p.add_argument("--out", type=str, default=None,
                   help="Output PDF path (default: same as npz with .pdf).")
    p.add_argument("--useful_lo", type=float, default=0.30)
    p.add_argument("--useful_hi", type=float, default=0.90)
    args = p.parse_args()

    npz_path = Path(args.npz)
    out_pdf = Path(args.out) if args.out else npz_path.with_suffix(".pdf")

    d = np.load(npz_path, allow_pickle=True)
    if "c_values" in d.files:
        replot_h(npz_path, out_pdf, args.useful_lo, args.useful_hi)
    elif "t_mh_values" in d.files:
        replot_tmh(npz_path, out_pdf, args.useful_lo, args.useful_hi)
    else:
        raise ValueError(f"Unrecognised npz keys: {d.files}")


if __name__ == "__main__":
    main()
