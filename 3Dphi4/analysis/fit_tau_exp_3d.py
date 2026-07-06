"""
Fit a single-exponential decay rho(tau) = A*exp(-tau/tau_exp) to the
chain-averaged ACF at each (L, epoch), as an alternative tau estimate
that does not saturate when the chain is shorter than the true tau_int.

Reads NPZs produced by dm_mc_S_vs_epoch_3d.py (which saved rho_S[epoch, lag])
and writes (L, epoch, tau_exp, tau_se) plus a comparison plot of
tau_exp vs tau_sokal.

The fit is done on lags where rho is in [rho_min, rho_max] and where the
fluctuations are still resolved; this avoids both the noisy long-tail
region and the lag-0 anchor that's exactly 1.

Usage:
  python analysis/fit_tau_exp_3d.py \
      --inputs results/dm_mc_S_vs_epoch_3d_indist_L*.npz \
      --out_prefix results/dm_mc_S_vs_epoch_3d_indist_tauexp
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.optimize import curve_fit


def fit_exp(rho, rho_min=0.1, rho_max=0.95, lag_min=1, max_lag=None):
    """
    Fit rho(tau) = A * exp(-tau / tau_exp) using rho values within
    [rho_min, rho_max], from lag = lag_min upward.
    Returns (tau_exp, sigma_tau, n_pts_used).
    """
    if max_lag is None:
        max_lag = len(rho)
    lags_all = np.arange(len(rho))
    # window: lag >= lag_min AND rho_min <= rho <= rho_max (and finite)
    valid = (lags_all >= lag_min) & (lags_all < max_lag) \
        & (rho > rho_min) & (rho < rho_max) & np.isfinite(rho)
    if valid.sum() < 4:
        return np.nan, np.nan, 0
    lags = lags_all[valid].astype(float)
    rs = rho[valid]
    # Use log-linear fit with polyfit for cheap, robust estimate; then refine
    # with curve_fit if it converges.
    log_rs = np.log(rs)
    slope, intercept, residuals, rank, _ = np.linalg.lstsq(
        np.vstack([lags, np.ones_like(lags)]).T, log_rs, rcond=None)[:5] \
        if False else (None,) * 5  # placeholder
    # Simpler: polyfit
    coef, cov = np.polyfit(lags, log_rs, 1, cov=True)
    slope = coef[0]
    if slope >= 0:
        return np.nan, np.nan, int(valid.sum())
    tau = -1.0 / slope
    # Standard error via cov; var(slope) -> var(tau) by delta method.
    var_slope = float(cov[0, 0])
    sigma_tau = float(np.sqrt(var_slope) / slope ** 2)

    # Optional refinement with curve_fit (nonlinear, weights ~ 1).
    try:
        A0 = np.exp(coef[1])
        popt, pcov = curve_fit(lambda t, A, tau: A * np.exp(-t / tau),
                               lags, rs, p0=(A0, tau),
                               maxfev=2000)
        tau = float(popt[1])
        sigma_tau = float(np.sqrt(pcov[1, 1]))
    except Exception:
        pass

    return tau, sigma_tau, int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--rho_min", type=float, default=0.1)
    parser.add_argument("--rho_max", type=float, default=0.95)
    parser.add_argument("--lag_min", type=int, default=1)
    args = parser.parse_args()

    per_L = {}
    meta_common = None
    for path in args.inputs:
        d = np.load(path)
        L = int(d["L"])
        rows_for_L = per_L.setdefault(L, [])
        for ie, ep in enumerate(d["epochs"]):
            rho_S = d["rho_S"][ie]
            tau_exp, sigma_tau, n_pts = fit_exp(
                rho_S, rho_min=args.rho_min, rho_max=args.rho_max,
                lag_min=args.lag_min)
            rows_for_L.append((int(ep),
                               float(d["accept"][ie]),
                               float(d["tau_S"][ie]),     # Sokal
                               float(d["se_S"][ie]),
                               int(d["W_S"][ie]),
                               tau_exp, sigma_tau, n_pts,
                               rho_S))
        if meta_common is None:
            meta_common = dict(k=float(d["k"]), l=float(d["l"]),
                               c=float(d["c"]), t_mh=float(d["t_mh"]),
                               num_chains=int(d["num_chains"]),
                               mh_steps=int(d["mh_steps"]))

    Ls = sorted(per_L.keys())
    print(f"L values: {Ls}")
    for L in Ls:
        per_L[L].sort(key=lambda r: r[0])

    print(f"\n{'L':>4s}  {'epoch':>6s}  {'accept':>7s}  "
          f"{'tau_sokal':>10s}  {'W':>4s}  {'tau_exp':>10s}  "
          f"{'sigma':>7s}  {'npts':>4s}")
    print("-" * 70)
    for L in Ls:
        for r in per_L[L]:
            ep, acc, tau_s, _se_s, W, tau_e, sig_e, n = r[:8]
            tau_e_str = f"{tau_e:>10.2f}" if np.isfinite(tau_e) else f"{'nan':>10s}"
            sig_e_str = f"{sig_e:>7.2f}" if np.isfinite(sig_e) else f"{'nan':>7s}"
            print(f"{L:>4d}  {ep:>6d}  {acc:>7.3f}  {tau_s:>10.2f}  "
                  f"{W:>4d}  {tau_e_str}  {sig_e_str}  {n:>4d}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    title_str = (rf"3D $\phi^4$, "
                 rf"$\kappa={meta_common['k']},\;\lambda={meta_common['l']:g}$")

    cmap = plt.get_cmap("viridis")

    # Plot 1: tau_exp vs epoch (one curve per L)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, L in enumerate(Ls):
        color = cmap(i / max(1, len(Ls) - 1))
        rows = per_L[L]
        eps = np.array([r[0] for r in rows])
        tau_exp = np.array([r[5] for r in rows])
        sig_exp = np.array([r[6] for r in rows])
        finite = np.isfinite(tau_exp)
        ax.errorbar(eps[finite], tau_exp[finite],
                    yerr=sig_exp[finite],
                    marker="o", ms=4, capsize=2,
                    color=color, label=rf"$L={L}$")
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
    ax.set_ylabel(r"$\tau_{\mathrm{exp}}(S)$  [MH steps, single-exp fit]")
    ax.set_title(title_str)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = Path(args.out_prefix + "_tau_exp_S.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")

    # Plot 2: tau_exp vs tau_sokal scatter — only L's with fitted data
    Ls_with_fit = [L for L in Ls
                   if any(np.isfinite(r[5]) for r in per_L[L])]
    if Ls_with_fit:
        fig, axes = plt.subplots(1, len(Ls_with_fit),
                                 figsize=(3.5 * len(Ls_with_fit), 4.0),
                                 sharey=False)
        if len(Ls_with_fit) == 1:
            axes = [axes]
        for ax, L in zip(axes, Ls_with_fit):
            rows = per_L[L]
            tau_s = np.array([r[2] for r in rows])
            tau_e = np.array([r[5] for r in rows])
            finite = np.isfinite(tau_e) & np.isfinite(tau_s) \
                & (tau_s > 0) & (tau_e > 0)
            if finite.any():
                ax.scatter(tau_s[finite], tau_e[finite], color="C0", s=20)
                mn = min(tau_s[finite].min(), tau_e[finite].min())
                mx = max(tau_s[finite].max(), tau_e[finite].max())
                ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.6, label=r"$y=x$")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(loc="best", fontsize=9)
            ax.set_xlabel(r"$\tau_{\mathrm{Sokal}}$")
            ax.set_ylabel(r"$\tau_{\mathrm{exp}}$")
            ax.set_title(rf"$L={L}$")
            ax.grid(alpha=0.3)
        fig.suptitle(title_str + r" — Sokal vs single-exp fit", y=1.02)
        fig.tight_layout()
        out = Path(args.out_prefix + "_sokal_vs_exp.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    # Plot 3: ACF curves with fitted exponential overlay (for one L, all epochs)
    # Pick L=16 as representative
    L_show = 16 if 16 in Ls else Ls[0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rows = per_L[L_show]
    cmap2 = plt.get_cmap("plasma")
    n_show = min(8, len(rows))
    show_idx = np.linspace(0, len(rows) - 1, n_show).astype(int)
    for j, idx in enumerate(show_idx):
        r = rows[idx]
        ep = r[0]
        rho_S = r[8]
        tau_e = r[5]
        max_lag = min(80, len(rho_S))
        lags = np.arange(max_lag)
        col = cmap2(j / max(1, n_show - 1))
        ax.plot(lags, rho_S[:max_lag], color=col, lw=1.2,
                label=rf"ep={ep}, $\tau_{{\rm exp}}={tau_e:.1f}$")
        if np.isfinite(tau_e):
            ax.plot(lags, np.exp(-lags / tau_e), color=col, ls=":", lw=0.9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"lag $\tau$ (MH steps)")
    ax.set_ylabel(r"$\bar\rho_S(\tau)$")
    ax.set_title(title_str + rf",  $L={L_show}$, ACF + exp fit (dotted)")
    ax.set_xlim(0, max_lag)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    out = Path(args.out_prefix + f"_acf_fits_L{L_show}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Save combined NPZ
    save = dict(**meta_common)
    for L in Ls:
        rows = per_L[L]
        save[f"L{L}_epochs"] = np.array([r[0] for r in rows])
        save[f"L{L}_accept"] = np.array([r[1] for r in rows])
        save[f"L{L}_tau_sokal"] = np.array([r[2] for r in rows])
        save[f"L{L}_tau_exp"] = np.array([r[5] for r in rows])
        save[f"L{L}_sigma_exp"] = np.array([r[6] for r in rows])
        save[f"L{L}_npts"] = np.array([r[7] for r in rows])
    np.savez(Path(args.out_prefix + ".npz"), **save)
    print(f"Saved {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
