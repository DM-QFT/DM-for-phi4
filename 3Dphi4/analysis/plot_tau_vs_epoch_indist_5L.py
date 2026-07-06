"""
Plot tau_exp vs epoch for 5 L values (4, 8, 16, 32, 64) using
in-distribution single-L models, each starting from its first
non-zero-accept epoch.

Reads the already-fitted dm_mc_S_vs_epoch_3d_indist_tauexp.npz which
has the per-L (epoch, accept, tau_exp) data.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter


def main():
    here = Path(__file__).resolve().parents[1]
    res = here / "results"
    d = np.load(res / "dm_mc_S_vs_epoch_3d_indist_tauexp.npz")

    Ls = [4, 8, 16, 32, 64]

    # For each L, build (epoch, accept, tau_exp, sigma_exp).
    per_L = {}
    for L in Ls:
        epk = f"L{L}_epochs"
        if epk not in d.files:
            continue
        eps = d[epk]
        accs = d[f"L{L}_accept"]
        taus = d[f"L{L}_tau_exp"]
        sigmas = d[f"L{L}_sigma_exp"]
        per_L[L] = (eps, accs, taus, sigmas)

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    title_str = (r"3D $\phi^4$, $\kappa=0.1923,\;\lambda=0.9,\;c=1.0$  "
                 r"(in-distribution single-$L$ models)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")

    print(f"{'L':>4s}  {'epoch':>6s}  {'accept':>10s}  "
          f"{'tau_exp':>10s}  {'sigma':>7s}")
    print("-" * 55)
    for i, L in enumerate(Ls):
        if L not in per_L:
            continue
        eps, accs, taus, sigmas = per_L[L]
        # Start from first non-zero-accept epoch with finite tau_exp.
        valid = (accs > 0) & np.isfinite(taus)
        if not valid.any():
            print(f"  L={L}: no finite tau_exp data")
            continue
        first_idx = np.argmax(valid)
        eps_v = eps[first_idx:][valid[first_idx:]]
        accs_v = accs[first_idx:][valid[first_idx:]]
        taus_v = taus[first_idx:][valid[first_idx:]]
        sigmas_v = sigmas[first_idx:][valid[first_idx:]]
        for e, p, t, s in zip(eps_v, accs_v, taus_v, sigmas_v):
            print(f"{L:>4d}  {int(e):>6d}  {p:>10.4g}  "
                  f"{t:>10.2f}  {s:>7.2f}")
        col = cmap(i / max(1, len(Ls) - 1))
        ax.errorbar(eps_v, taus_v, yerr=sigmas_v,
                    marker="o", ms=6, capsize=2,
                    color=col, label=rf"$L={L}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0,
                   subs=(1.5, 2.5, 4.0, 6.0, 8.0, 9.0), numticks=60))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\tau_{\mathrm{exp}}(S)$  [MH steps]")
    ax.set_title(title_str)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = res / "tau_vs_epoch_indist_5L.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
