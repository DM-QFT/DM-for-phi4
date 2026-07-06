"""
Estimate tau_int(S) for the early-epoch (frozen-chain) regime where
direct ACF fitting fails because rho(tau) is identically 1.

Method:
  1. At late epochs where the single-exp fit converges, define
       N_unrej(L, ep) = tau_exp(L, ep) * p_accept(L, ep)
     -- the number of accepted moves needed to decorrelate S.
  2. Average N_unrej over late epochs to get a per-L constant.
  3. For ANY epoch, estimate
       tau_phys(ep) = N_unrej(L) / p_accept(ep).
     -- when p_accept > 0 from data, this is a direct estimate.
     -- when p_accept = 0 exactly (no chain ever accepted), use the
        upper bound p_accept < 1/(M*N) and report the LOWER BOUND
        tau_phys >= N_unrej * M * N.

This is a heuristic: it assumes each accepted move contributes the same
amount of decorrelation regardless of training stage. It is NOT a
rigorous estimator, but it lets you put physical numbers on the
"chain-frozen" plateau where direct fitting gives nothing.

Usage:
  python analysis/extrapolate_tau_early_3d.py \
      --inputs results/dm_mc_S_vs_epoch_3d_indist_L*.npz \
      --out_prefix results/dm_mc_S_vs_epoch_3d_indist_tauphys
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.optimize import curve_fit


def fit_exp(rho, rho_min=0.1, rho_max=0.95, lag_min=1):
    lags_all = np.arange(len(rho))
    valid = (lags_all >= lag_min) & (rho > rho_min) & (rho < rho_max) \
        & np.isfinite(rho)
    if valid.sum() < 4:
        return np.nan
    lags = lags_all[valid].astype(float)
    rs = rho[valid]
    coef = np.polyfit(lags, np.log(rs), 1)
    if coef[0] >= 0:
        return np.nan
    tau = -1.0 / coef[0]
    try:
        popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau),
                            lags, rs, p0=(np.exp(coef[1]), tau), maxfev=2000)
        tau = float(popt[1])
    except Exception:
        pass
    return tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--late_accept_min", type=float, default=0.1,
                        help="Use epochs with accept > this for N_unrej calibration.")
    args = parser.parse_args()

    per_L = {}
    meta_common = None
    for path in args.inputs:
        d = np.load(path)
        L = int(d["L"])
        rows = per_L.setdefault(L, [])
        for ie, ep in enumerate(d["epochs"]):
            rho_S = d["rho_S"][ie]
            tau_exp = fit_exp(rho_S)
            rows.append((int(ep),
                         float(d["accept"][ie]),
                         float(d["tau_S"][ie]),     # Sokal
                         tau_exp))
        if meta_common is None:
            meta_common = dict(k=float(d["k"]), l=float(d["l"]),
                               c=float(d["c"]), t_mh=float(d["t_mh"]),
                               num_chains=int(d["num_chains"]),
                               mh_steps=int(d["mh_steps"]))

    Ls = sorted(per_L.keys())
    M = meta_common["num_chains"]
    N = meta_common["mh_steps"]
    accept_floor = 1.0 / (M * N)        # smallest non-trivially-resolvable accept

    # 1) Calibrate N_unrej per L from late epochs.
    N_unrej_per_L = {}
    for L in Ls:
        rows = per_L[L]
        late = [r for r in rows
                if r[1] > args.late_accept_min and np.isfinite(r[3])]
        if not late:
            N_unrej_per_L[L] = np.nan
            continue
        prods = np.array([r[3] * r[1] for r in late])
        N_unrej_per_L[L] = float(np.median(prods))

    print(f"Calibration (M={M}, N={N}, accept floor = 1/(M*N) = {accept_floor:.2e}):")
    for L in Ls:
        if np.isnan(N_unrej_per_L[L]):
            print(f"  L={L}: no late-epoch data, N_unrej undetermined")
        else:
            print(f"  L={L}: N_unrej = median(tau_exp * accept) = {N_unrej_per_L[L]:.2f}")

    # 2) Compute tau_phys at every epoch.
    print(f"\n{'L':>4s}  {'epoch':>6s}  {'accept':>10s}  {'tau_sokal':>10s}  "
          f"{'tau_exp':>10s}  {'tau_phys':>11s}")
    print("-" * 70)
    for L in Ls:
        N_unrej = N_unrej_per_L[L]
        for ep, p, tau_s, tau_e in per_L[L]:
            if not np.isfinite(N_unrej):
                tau_p = np.nan
                lb_flag = ""
            elif p > 0:
                tau_p = N_unrej / p
                lb_flag = ""
            else:
                # No accept observed across M*N proposals -> p < 1/(M*N).
                # tau_phys > N_unrej / (1/(M*N)) = N_unrej * M * N.
                tau_p = N_unrej * M * N
                lb_flag = " (>=)"
            tau_e_str = f"{tau_e:>10.2f}" if np.isfinite(tau_e) else f"{'nan':>10s}"
            print(f"{L:>4d}  {ep:>6d}  {p:>10.4g}  {tau_s:>10.2f}  "
                  f"{tau_e_str}  {tau_p:>11.3g}{lb_flag}")

    # 3) Plot tau_phys vs epoch (one curve per L), using log-log.
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    title_str = (rf"3D $\phi^4$, $\kappa={meta_common['k']},\;"
                 rf"\lambda={meta_common['l']:g}$ "
                 rf"(extrapolated $\tau_{{\rm phys}} = N_{{\rm unrej}} / p_{{\rm accept}}$)")
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, L in enumerate(Ls):
        N_unrej = N_unrej_per_L[L]
        if not np.isfinite(N_unrej):
            continue
        color = cmap(i / max(1, len(Ls) - 1))
        rows = per_L[L]
        eps = np.array([r[0] for r in rows])
        accs = np.array([r[1] for r in rows])
        taus = np.where(accs > 0, N_unrej / np.where(accs > 0, accs, 1.0),
                        N_unrej * M * N)
        # Distinguish: solid markers for measured (accept>0), open for lower bound.
        meas = accs > 0
        if meas.any():
            ax.plot(eps[meas], taus[meas], "o-", color=color, ms=5,
                    label=rf"$L={L}$ (measured)")
        if (~meas).any():
            ax.plot(eps[~meas], taus[~meas], "v", color=color, ms=8,
                    mfc="none", mew=1.5,
                    label=rf"$L={L}$ (lower bound, $p<1/MN$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\tau_{\mathrm{phys}}(S)$  [MH steps]")
    ax.set_title(title_str)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out = Path(args.out_prefix + "_tau_phys.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")

    # 4) Plot tau_exp vs accept_rate (calibration check).
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, L in enumerate(Ls):
        color = cmap(i / max(1, len(Ls) - 1))
        rows = per_L[L]
        accs = np.array([r[1] for r in rows])
        tau_e = np.array([r[3] for r in rows])
        finite = np.isfinite(tau_e) & (accs > 0)
        if finite.any():
            ax.plot(accs[finite], tau_e[finite], "o", color=color,
                    ms=5, label=rf"$L={L}$")
    # Reference: tau = N_unrej_avg / p
    if any(np.isfinite(v) for v in N_unrej_per_L.values()):
        avg_N = np.nanmean([N_unrej_per_L[L] for L in Ls])
        ps = np.geomspace(0.01, 1.0, 50)
        ax.plot(ps, avg_N / ps, "k--", lw=1, alpha=0.6,
                label=rf"$\tau = {avg_N:.1f}/p$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=30))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"$p_{\mathrm{accept}}$")
    ax.set_ylabel(r"$\tau_{\mathrm{exp}}(S)$")
    ax.set_title(title_str + r"  (calibration)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out = Path(args.out_prefix + "_calibration.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Save merged data.
    save = dict(**meta_common, accept_floor=accept_floor,
                N_unrej_per_L=np.array([N_unrej_per_L[L] for L in Ls]),
                Ls=np.array(Ls))
    for L in Ls:
        rows = per_L[L]
        save[f"L{L}_epochs"] = np.array([r[0] for r in rows])
        save[f"L{L}_accept"] = np.array([r[1] for r in rows])
        save[f"L{L}_tau_sokal"] = np.array([r[2] for r in rows])
        save[f"L{L}_tau_exp"] = np.array([r[3] for r in rows])
        N_unrej = N_unrej_per_L[L]
        accs = np.array([r[1] for r in rows])
        if np.isfinite(N_unrej):
            taus = np.where(accs > 0, N_unrej / np.where(accs > 0, accs, 1.0),
                            N_unrej * M * N)
        else:
            taus = np.full_like(accs, np.nan)
        save[f"L{L}_tau_phys"] = taus
        save[f"L{L}_lb_flag"] = (accs == 0).astype(int)
    np.savez(Path(args.out_prefix + ".npz"), **save)
    print(f"Saved {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
