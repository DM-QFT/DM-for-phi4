"""
Plot tau_exp vs epoch at c=1.0 starting from the first epoch with
non-zero acceptance (chain stops being frozen).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter


def main():
    here = Path(__file__).resolve().parents[1]
    res = here / "results"
    d = np.load(res / "dm_mc_S_Nscan_L16_c1.0_tauexp.npz")
    eps = d["epochs"]
    accs = d["accept"]
    taus = d["tau_exp"]
    sigmas = d["sigma_exp"]
    finite = np.isfinite(taus) & (accs > 0)
    eps_f = eps[finite]
    taus_f = taus[finite]
    sigmas_f = sigmas[finite]
    accs_f = accs[finite]

    print(f"Plotting from first non-zero-accept epoch: ep={eps_f[0]}")
    for e, p, t, s in zip(eps_f, accs_f, taus_f, sigmas_f):
        print(f"  ep={int(e):>5d}  accept={p:.4g}  tau={t:.2f}+/-{s:.2f}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(eps_f, taus_f, yerr=sigmas_f, marker="o", ms=7, capsize=3,
                color="C0", lw=1.5,
                label=r"single-$L$ model (L16)")
    for e, t, p in zip(eps_f, taus_f, accs_f):
        ax.annotate(f"acc={p:.3g}", xy=(e, t), xytext=(7, 7),
                    textcoords="offset points", fontsize=9, color="C0")

    # Multi-L model at ep=10000
    multi_path = res / "dm_mc_S_Nscan_L16_c1.0_multiL_ep10000_tauexp.npz"
    if multi_path.exists():
        dm = np.load(multi_path)
        em = dm["epochs"]
        am = dm["accept"]
        tm = dm["tau_exp"]
        sm = dm["sigma_exp"]
        m_finite = np.isfinite(tm) & (am > 0)
        ax.errorbar(em[m_finite], tm[m_finite], yerr=sm[m_finite],
                    marker="s", ms=10, capsize=3,
                    color="C3", lw=1.5,
                    label=r"multi-$L$ model (4-8-16-32) at $L=16$")
        for e, t, p in zip(em[m_finite], tm[m_finite], am[m_finite]):
            ax.annotate(f"acc={p:.3g}", xy=(e, t), xytext=(7, -14),
                        textcoords="offset points", fontsize=9, color="C3")

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
    ax.set_title(r"3D $\phi^4$, $L=16,\;\kappa=0.1923,\;\lambda=0.9,\;c=1.0$  "
                 r"(from first non-zero-accept epoch)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = res / "tau_vs_epoch_c1_3d_L16.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
