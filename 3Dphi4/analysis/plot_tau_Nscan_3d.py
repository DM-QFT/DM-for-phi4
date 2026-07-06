"""
Merge worker NPZs from fit_tau_from_Nscan_3d.py and plot:
  1. T_S(n) curves with fitted exponential saturation overlaid (one
     panel per epoch).
  2. tau_fit vs epoch (the extracted true integrated time).

Usage:
  python analysis/plot_tau_Nscan_3d.py \
      --inputs results/dm_mc_S_Nscan_L16_part_*.npz \
      --out_prefix results/dm_mc_S_Nscan_L16
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter


def model(n, tau, A):
    return 0.5 + A * tau * (1.0 - np.exp(-(n - 1) / tau))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    rows = []
    meta = None
    for path in args.inputs:
        d = np.load(path)
        if meta is None:
            meta = dict(L=int(d["L"]), k=float(d["k"]), l=float(d["l"]),
                        c=float(d["c"]), t_mh=float(d["t_mh"]),
                        num_chains=int(d["num_chains"]),
                        mh_steps=int(d["mh_steps"]))
        for ie, ep in enumerate(d["epochs"]):
            rows.append((int(ep), float(d["accept"][ie]),
                         float(d["tau_fit"][ie]),
                         float(d["A_fit"][ie]),
                         float(d["sigma_fit"][ie]),
                         d["rho_S"][ie], d["T_S"][ie]))
    rows.sort(key=lambda r: r[0])
    eps = np.array([r[0] for r in rows])
    accs = np.array([r[1] for r in rows])
    taus = np.array([r[2] for r in rows])
    As = np.array([r[3] for r in rows])
    sigmas = np.array([r[4] for r in rows])

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    title_str = (rf"3D $\phi^4$, $L={meta['L']},\;\kappa={meta['k']},\;"
                 rf"\lambda={meta['l']:g}$, $c={meta['c']}$")

    print(f"{'epoch':>6s}  {'accept':>10s}  {'tau_fit':>10s}  "
          f"{'sigma':>7s}  {'A':>5s}  {'T_S(N)':>9s}")
    print("-" * 60)
    for r in rows:
        ep, p, tau, A, sig, rho, T = r
        tau_str = f"{tau:>10.2f}" if np.isfinite(tau) else f"{'nan':>10s}"
        sig_str = f"{sig:>7.2f}" if np.isfinite(sig) else f"{'nan':>7s}"
        A_str = f"{A:>5.2f}" if np.isfinite(A) else f"{'nan':>5s}"
        print(f"{ep:>6d}  {p:>10.4g}  {tau_str}  {sig_str}  {A_str}  {T[-1]:>9.2f}")

    # Plot 1: T_S(n) curves with fits, one panel per epoch
    n_eps = len(rows)
    ncols = min(4, n_eps)
    nrows = (n_eps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows),
                             squeeze=False)
    n_arr = np.arange(1, meta["mh_steps"] + 1)
    for j, r in enumerate(rows):
        ax = axes[j // ncols, j % ncols]
        ep, p, tau, A, sig, rho, T = r
        ax.plot(n_arr, T, "C0-", lw=1.2, label="data $T_S(n)$")
        if np.isfinite(tau):
            ax.plot(n_arr, model(n_arr, tau, A), "C3--", lw=1.4,
                    label=rf"fit: $\tau={tau:.1f}\pm{sig:.1f}$, $A={A:.2f}$")
        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$T_S(n) = \frac{1}{2}+\sum\rho$")
        ax.set_title(rf"ep={ep}, accept={p:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    # Hide unused panels
    for j in range(n_eps, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title_str, y=1.0, fontsize=11)
    fig.tight_layout()
    out = Path(args.out_prefix + "_T_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")

    # Plot 2: tau_fit vs epoch (with N/2 reference, accept-rate overlay)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    finite = np.isfinite(taus)
    if finite.any():
        ax.errorbar(eps[finite], taus[finite], yerr=sigmas[finite],
                    marker="o", ms=6, capsize=3, color="C0",
                    label=r"$\tau_{\rm fit}$ (saturating-exp model)")
    ax.axhline(meta["mh_steps"], color="k", ls=":", lw=1, alpha=0.7,
               label=rf"chain length $N={meta['mh_steps']}$")
    ax.axhline(meta["mh_steps"] / 2, color="gray", ls=":", lw=1, alpha=0.7,
               label=rf"$N/2={meta['mh_steps']//2}$ (Sokal saturation floor)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\tau_{\mathrm{fit}}(S)$  [MH steps]")
    ax.set_title(title_str)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = Path(args.out_prefix + "_tau_fit.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Save merged data.
    save = dict(**meta,
                epochs=eps, accept=accs,
                tau_fit=taus, A_fit=As, sigma_fit=sigmas)
    np.savez(Path(args.out_prefix + ".npz"), **save)
    print(f"Saved {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
