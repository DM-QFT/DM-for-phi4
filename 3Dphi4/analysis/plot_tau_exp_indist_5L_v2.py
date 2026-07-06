"""
Combine fit_tau_exp_Nscan output NPZs across 5 L values (in-distribution
single-L models), plot tau_exp vs epoch with one curve per L.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter


def main():
    here = Path(__file__).resolve().parents[1]
    res = here / "results"

    # L -> list of NPZ paths (fit_tau_exp_Nscan outputs, one per worker)
    sources = {
        4:  [res / "dm_mc_S_Nscan_indist_L4_v2_tauexp.npz"],
        8:  [res / "dm_mc_S_Nscan_indist_L8_v2_tauexp.npz"],
        16: [res / "dm_mc_S_Nscan_indist_L16_v2_tauexp.npz"],
        32: [res / "dm_mc_S_Nscan_indist_L32_v2_a_tauexp.npz",
             res / "dm_mc_S_Nscan_indist_L32_v2_b_tauexp.npz"],
        64: [res / "dm_mc_S_Nscan_indist_L64_v2_main_tauexp.npz",
             res / "dm_mc_S_Nscan_indist_L64_v2_late_tauexp.npz",
             res / "dm_mc_S_Nscan_indist_L64_v2_refine_tauexp.npz"],
    }

    per_L = {}
    for L, paths in sources.items():
        # Collect, prefer finite tau over NaN if duplicate epoch.
        ep_to_row = {}
        for path in paths:
            d = np.load(path)
            for ep, p, t, s in zip(d["epochs"], d["accept"],
                                   d["tau_exp"], d["sigma_exp"]):
                ep_i = int(ep)
                row = (ep_i, float(p), float(t), float(s))
                if ep_i not in ep_to_row or (
                    not np.isfinite(ep_to_row[ep_i][2]) and np.isfinite(t)
                ):
                    ep_to_row[ep_i] = row
        per_L[L] = sorted(ep_to_row.values(), key=lambda r: r[0])

    print(f"{'L':>4s}  {'epoch':>6s}  {'accept':>10s}  "
          f"{'tau_exp':>10s}  {'sigma':>7s}")
    print("-" * 55)
    for L in sorted(per_L.keys()):
        for ep, p, t, s in per_L[L]:
            ts = f"{t:>10.2f}" if np.isfinite(t) else f"{'nan':>10s}"
            ss = f"{s:>7.2f}" if np.isfinite(s) else f"{'nan':>7s}"
            print(f"{L:>4d}  {ep:>6d}  {p:>10.4g}  {ts}  {ss}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    title_str = (r"3D $\phi^4$, $\kappa=0.1923,\;\lambda=0.9,\;c=1.0$  "
                 r"(in-distribution single-$L$ models)")

    cmap = plt.get_cmap("viridis")
    Ls = sorted(per_L.keys())

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, L in enumerate(Ls):
        rows = per_L[L]
        eps = np.array([r[0] for r in rows])
        accs = np.array([r[1] for r in rows])
        taus = np.array([r[2] for r in rows])
        sigmas = np.array([r[3] for r in rows])
        valid = (accs > 0) & np.isfinite(taus)
        if not valid.any():
            continue
        col = cmap(i / max(1, len(Ls) - 1))
        ax.errorbar(eps[valid], taus[valid], yerr=sigmas[valid],
                    marker="o", ms=5, capsize=2,
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
    out = res / "tau_exp_indist_5L_v2.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")

    # Save combined NPZ
    save = {}
    for L in Ls:
        rows = per_L[L]
        save[f"L{L}_epochs"] = np.array([r[0] for r in rows])
        save[f"L{L}_accept"] = np.array([r[1] for r in rows])
        save[f"L{L}_tau_exp"] = np.array([r[2] for r in rows])
        save[f"L{L}_sigma_exp"] = np.array([r[3] for r in rows])
    np.savez(res / "tau_exp_indist_5L_v2.npz", **save)
    print(f"Saved {res/'tau_exp_indist_5L_v2.npz'}")


if __name__ == "__main__":
    main()
