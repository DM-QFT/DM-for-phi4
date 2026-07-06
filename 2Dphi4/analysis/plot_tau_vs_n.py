"""
Plot integrated autocorrelation time tau_int as a function of chain length n.

Reads the dm_mc_ess_2D_*.npz produced by dm_mc_ess_2d.py, and for each
truncation length n re-estimates tau_int from phi_bar[1:n+1] using the same
FFT-based ACF + Sokal automatic windowing (and a fixed-cutoff fallback).

A converged tau_int(n) plateaus once n >> tau_int. If it keeps growing,
the chain is too short.

Usage:
  python analysis/plot_tau_vs_n.py \
      --npz results/dm_mc_ess_2D_L32_k0.28_e7564.npz
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from dm_mc_ess_2d import acf_per_chain_fft, sokal_window, cutoff_window


def tau_curves(phi_bar, n_grid, sokal_c=5.0, acf_cutoff=0.05):
    """
    For each n in n_grid, estimate tau_int(n) using the first n post-step samples.
    phi_bar has shape (N+1, M); we use phi_bar[1:n+1].
    Returns arrays (n, tau_sokal, sokal_W, tau_se, tau_cutoff).
    """
    M = phi_bar.shape[1]
    rows = []
    for n in n_grid:
        series = phi_bar[1:n + 1]                      # (n, M)
        if series.shape[0] < 4:
            rows.append((n, np.nan, 0, np.nan, np.nan))
            continue
        rho = acf_per_chain_fft(series).mean(axis=1)   # average across chains
        tau_s, W, _ = sokal_window(rho, c=sokal_c)
        tau_c = cutoff_window(rho, cutoff=acf_cutoff)
        # Madras-Sokal s.e. on tau_int from total samples M*n.
        se = np.sqrt(2.0 * (2 * W + 1) / (M * n)) * tau_s
        rows.append((n, tau_s, W, se, tau_c))
    arr = np.array(rows, dtype=float)
    return arr  # columns: n, tau_sokal, sokal_W, tau_se, tau_cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to dm_mc_ess_2D_*.npz from dm_mc_ess_2d.py.")
    parser.add_argument("--n_min", type=int, default=20)
    parser.add_argument("--n_step", type=int, default=20,
                        help="Linear stride on n.")
    parser.add_argument("--sokal_c", type=float, default=5.0)
    parser.add_argument("--acf_cutoff", type=float, default=0.05)
    parser.add_argument("--out", type=str, default=None,
                        help="Output PDF path; default appends _tau_vs_n to the npz stem.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    d = np.load(npz_path)
    L = int(d["L"])
    k = float(d["k"])
    M = int(d["num_chains"])
    N = int(d["mh_steps"])
    c = float(d["args_c"])
    t_mh = float(d["args_t_mh"])
    early_ep = int(d["early_epoch"])
    late_ep = int(d["late_epoch"])

    n_grid = np.arange(args.n_min, N + 1, args.n_step, dtype=int)
    if n_grid[-1] != N:
        n_grid = np.append(n_grid, N)

    print(f"L={L}, k={k}, M={M}, N={N}, c={c}, t_mh={t_mh:g}")
    print(f"n grid: {n_grid[0]} .. {n_grid[-1]} (step={args.n_step}, "
          f"{len(n_grid)} points)")

    print("\n--- early ---")
    early = tau_curves(d["early_phi_bar"], n_grid,
                       sokal_c=args.sokal_c, acf_cutoff=args.acf_cutoff)
    print("\n--- late ---")
    late = tau_curves(d["late_phi_bar"], n_grid,
                      sokal_c=args.sokal_c, acf_cutoff=args.acf_cutoff)

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.4, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"early": "C3", "late": "C0"}

    for arr, label, ep in [(early, "early", early_ep), (late, "late", late_ep)]:
        col = colors[label]
        ax1.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 3],
                     marker="o", ms=3, capsize=2,
                     color=col, label=f"{label} (ep {ep}), Sokal")
        ax1.plot(arr[:, 0], arr[:, 4], "--", color=col, alpha=0.6,
                 label=f"{label}, cutoff $\\rho<{args.acf_cutoff}$")
        # Sokal self-consistency: dotted line n = sokal_c * tau (chain barely long enough).
        ax2.plot(arr[:, 0], arr[:, 0] / np.where(arr[:, 1] > 0, arr[:, 1], np.nan),
                 marker=".", ms=3, color=col, label=f"{label} (ep {ep})")

    ax1.set_xlabel("chain length $n$ (MH steps used)")
    ax1.set_ylabel(r"$\tau_{\mathrm{int}}(n)$")
    ax1.set_title(rf"$\tau_{{\mathrm{{int}}}}$ vs chain length, $L={L},\;\kappa={k}$, "
                  rf"$M={M}$ chains")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    ax2.axhline(args.sokal_c, color="k", ls=":", lw=1,
                label=rf"Sokal threshold $n/\tau={args.sokal_c:g}$")
    ax2.set_xlabel("chain length $n$")
    ax2.set_ylabel(r"$n / \tau_{\mathrm{int}}(n)$")
    ax2.set_title("Sokal self-consistency ratio")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(rf"DM-MALA, 2D $\phi^4$, $c={c},\;t_{{\mathrm{{mh}}}}={t_mh:g}$",
                 y=1.02, fontsize=12)
    fig.tight_layout()

    out = Path(args.out) if args.out else npz_path.with_name(
        npz_path.stem + "_tau_vs_n.pdf"
    )
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    # Save the curves alongside.
    out_npz = out.with_suffix(".npz")
    np.savez(out_npz, n_grid=n_grid,
             early=early, late=late,
             columns=np.array(["n", "tau_sokal", "sokal_W", "tau_se", "tau_cutoff"]),
             early_epoch=early_ep, late_epoch=late_ep,
             L=L, k=k, c=c, t_mh=t_mh, num_chains=M, mh_steps=N)
    print(f"Saved curves to {out_npz}")


if __name__ == "__main__":
    main()
