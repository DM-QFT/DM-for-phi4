"""
Fit single-exponential decay rho(tau) = A*exp(-tau/tau_exp) to the
EARLY-lag region of the chain-averaged ACF, using NPZs produced by
fit_tau_from_Nscan_3d.py. Fits use lags where rho is in [rho_min,
rho_max], avoiding the drift-contaminated negative tail at large lags.

Usage:
  python analysis/fit_tau_exp_Nscan.py \
      --inputs results/dm_mc_S_Nscan_L16_c0.02_part_*.npz \
      --out_prefix results/dm_mc_S_Nscan_L16_c0.02_tauexp
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.optimize import curve_fit


def fit_exp(rho, rho_min=0.1, rho_max=0.95, lag_min=1, max_lag=None):
    """rho(tau) = A * exp(-tau/tau_exp), fit on rho in [rho_min, rho_max]."""
    if max_lag is None:
        max_lag = len(rho)
    lags_all = np.arange(len(rho))
    valid = (lags_all >= lag_min) & (lags_all < max_lag) \
        & (rho > rho_min) & (rho < rho_max) & np.isfinite(rho)
    if valid.sum() < 4:
        return np.nan, np.nan, 0
    lags = lags_all[valid].astype(float)
    rs = rho[valid]
    # Log-linear seed.
    coef = np.polyfit(lags, np.log(rs), 1)
    if coef[0] >= 0:
        return np.nan, np.nan, int(valid.sum())
    tau = -1.0 / coef[0]
    A0 = np.exp(coef[1])
    try:
        popt, pcov = curve_fit(lambda t, A, tau: A * np.exp(-t / tau),
                               lags, rs, p0=(A0, tau), maxfev=2000)
        tau = float(popt[1])
        sigma_tau = float(np.sqrt(pcov[1, 1]))
    except Exception:
        sigma_tau = np.nan
    return tau, sigma_tau, int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--rho_min", type=float, default=0.1)
    parser.add_argument("--rho_max", type=float, default=0.95)
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
            rho = d["rho_S"][ie]
            tau, sigma, n_pts = fit_exp(rho, rho_min=args.rho_min,
                                        rho_max=args.rho_max)
            rows.append((int(ep), float(d["accept"][ie]),
                         tau, sigma, n_pts, rho))
    rows.sort(key=lambda r: r[0])

    print(f"{'epoch':>6s}  {'accept':>10s}  {'tau_exp':>10s}  "
          f"{'sigma':>8s}  {'npts':>5s}")
    print("-" * 50)
    for ep, p, tau, sig, n, _ in rows:
        ts = f"{tau:>10.2f}" if np.isfinite(tau) else f"{'nan':>10s}"
        ss = f"{sig:>8.2f}" if np.isfinite(sig) else f"{'nan':>8s}"
        print(f"{ep:>6d}  {p:>10.4g}  {ts}  {ss}  {n:>5d}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    title_str = (rf"3D $\phi^4$, $L={meta['L']},\;\kappa={meta['k']},\;"
                 rf"\lambda={meta['l']:g}$, $c={meta['c']}$  (early-lag exp fit)")

    eps = np.array([r[0] for r in rows])
    accs = np.array([r[1] for r in rows])
    taus = np.array([r[2] for r in rows])
    sigmas = np.array([r[3] for r in rows])

    # tau_exp vs epoch
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    finite = np.isfinite(taus)
    if finite.any():
        ax.errorbar(eps[finite], taus[finite], yerr=sigmas[finite],
                    marker="o", ms=5, capsize=2, color="C0",
                    label=r"$\tau_{\rm exp}$ (fit on $\rho\in[0.1,0.95]$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\tau_{\mathrm{exp}}(S)$  [MH steps]")
    ax.set_title(title_str)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = Path(args.out_prefix + "_tau_exp.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")

    # ACF curves with fits
    n_show = min(8, len(rows))
    show_idx = np.linspace(0, len(rows) - 1, n_show).astype(int)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("plasma")
    for j, idx in enumerate(show_idx):
        ep, p, tau, sig, n, rho = rows[idx]
        max_lag = min(1500, len(rho))
        lags = np.arange(max_lag)
        col = cmap(j / max(1, n_show - 1))
        ax.plot(lags, rho[:max_lag], color=col, lw=1.0,
                label=rf"ep={ep}, acc={p:.3f}, $\tau={tau:.0f}$"
                if np.isfinite(tau) else rf"ep={ep}, acc={p:.3f}")
        if np.isfinite(tau):
            ax.plot(lags, np.exp(-lags / tau), color=col, ls=":", lw=0.9, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"lag $\tau$ (MH steps)")
    ax.set_ylabel(r"$\bar\rho_S(\tau)$")
    ax.set_title(title_str + " — ACF + exp fit (dotted)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    out = Path(args.out_prefix + "_acf_fits.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Save NPZ
    save = dict(**meta,
                epochs=eps, accept=accs,
                tau_exp=taus, sigma_exp=sigmas)
    np.savez(Path(args.out_prefix + ".npz"), **save)
    print(f"Saved {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
