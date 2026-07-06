"""
Plot tau_exp vs c for the late epochs (5214, 9111) at L=16, aggregating
results from the c=0.02, 0.1, 0.2 (Nscan) and c=1.0 (S_vs_epoch) fits.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter

LATE_EPS = [5214, 9111]


def load_tauexp_per_ep(npz_paths):
    """Read tau_exp NPZ from fit_tau_exp_Nscan or fit_tau_exp_3d outputs.
    Returns dict {epoch: tau_exp}."""
    out = {}
    for path in npz_paths:
        d = np.load(path)
        eps = d["epochs"] if "epochs" in d.files else None
        if eps is None:
            # fit_tau_exp_3d format: keys like L16_epochs / L16_tau_exp
            for L in d.get("Ls", [16]):
                ep_key = f"L{int(L)}_epochs"
                tau_key = f"L{int(L)}_tau_exp"
                if ep_key in d.files and tau_key in d.files:
                    for e, t in zip(d[ep_key], d[tau_key]):
                        out[int(e)] = float(t)
        else:
            taus = d["tau_exp"]
            for e, t in zip(eps, taus):
                out[int(e)] = float(t)
    return out


def main():
    here = Path(__file__).resolve().parents[1]            # 3Dphi4/
    res = here / "results"

    sources = {
        0.005: [res / "dm_mc_S_Nscan_L16_c0.005_tauexp.npz"],
        0.02:  [res / "dm_mc_S_Nscan_L16_c0.02_tauexp.npz"],
        0.1:   [res / "dm_mc_S_Nscan_L16_c0.1_tauexp.npz"],
        0.2:   [res / "dm_mc_S_Nscan_L16_c0.2_tauexp.npz"],
        0.5:   [res / "dm_mc_S_Nscan_L16_c0.5_tauexp.npz"],
        1.0:   [res / "dm_mc_S_Nscan_L16_c1.0_tauexp.npz"],
        2.0:   [res / "dm_mc_S_Nscan_L16_c2.0_tauexp.npz"],
        5.0:   [res / "dm_mc_S_Nscan_L16_c5.0_tauexp.npz"],
        10.0:  [res / "dm_mc_S_Nscan_L16_c10.0_tauexp.npz"],
    }

    # ep -> {c -> tau}
    table = {ep: {} for ep in LATE_EPS}
    for c, paths in sources.items():
        per_ep = load_tauexp_per_ep([str(p) for p in paths])
        for ep in LATE_EPS:
            if ep in per_ep:
                table[ep][c] = per_ep[ep]

    print(f"{'epoch':>6s}  " + "  ".join(f"c={c}".rjust(10) for c in sorted(sources)))
    print("-" * 60)
    for ep in LATE_EPS:
        line = f"{ep:>6d}"
        for c in sorted(sources):
            v = table[ep].get(c, np.nan)
            line += f"  {v:>10.2f}"
        print(line)

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, ep in enumerate(LATE_EPS):
        cs = sorted(table[ep].keys())
        taus = [table[ep][c] for c in cs]
        col = cmap(0.2 + 0.6 * i / max(1, len(LATE_EPS) - 1))
        ax.plot(cs, taus, "o-", ms=8, color=col, label=rf"ep$={ep}$")
        for c, t in zip(cs, taus):
            ax.annotate(f"{t:.0f}", xy=(c, t), xytext=(4, 4),
                        textcoords="offset points", fontsize=9, color=col)

    # Reference lines: tau ~ 1/c and tau ~ const
    cs_ref = np.geomspace(0.01, 2.0, 50)
    if 0.02 in table[LATE_EPS[0]]:
        tau_at_002 = table[LATE_EPS[0]][0.02]
        ax.plot(cs_ref, tau_at_002 * (0.02 / cs_ref), "k--", lw=1, alpha=0.4,
                label=r"reference: $\tau\propto 1/c$")
        ax.plot(cs_ref, np.full_like(cs_ref, tau_at_002), "k:", lw=1, alpha=0.4,
                label=r"reference: $\tau=\text{const}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0,
                   subs=(1.5, 2.5, 4.0, 6.0, 8.0, 9.0), numticks=60))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"$c$  ($h = c/L^3$)")
    ax.set_ylabel(r"$\tau_{\mathrm{exp}}(S)$  [MH steps]")
    ax.set_title(r"3D $\phi^4$, $L=16,\;\kappa=0.1923,\;\lambda=0.9$  "
                 r"(late-epoch $\tau_{\rm exp}$ vs $c$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = res / "tau_vs_c_late_3d_L16.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")

    # Save data
    np.savez(res / "tau_vs_c_late_3d_L16.npz",
             c_values=np.array(sorted(sources)),
             eps=np.array(LATE_EPS),
             tau_table=np.array([[table[ep].get(c, np.nan)
                                  for c in sorted(sources)]
                                 for ep in LATE_EPS]))
    print(f"Saved {res/'tau_vs_c_late_3d_L16.npz'}")


if __name__ == "__main__":
    main()
