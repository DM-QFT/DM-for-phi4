"""
Plot the normalised autocorrelation of |M| = |phi-bar| over short MC lags,
mirroring the style of the reference Fig.~8 (DM-MC vs HMC vs MC).

Reads the dm_mc_ess_2D_*.npz produced by dm_mc_ess_2d.py and recomputes
the ACF using the absolute magnetisation as the observable. |M| is Z2-
symmetric so it does not see the (intractable) Z2 tunnelling that makes
the bare phi-bar autocorrelation grow without bound when the chain is
trapped in one broken-phase well.

Optionally overlays HMC training-data ACF (treating consecutive HMC cfgs
as a single chain of length N_train, observable |phi-bar|).

Usage:
  python analysis/plot_acf_M.py \
      --npz results/dm_mc_ess_2D_L32_k0.28_e7564.npz \
      --max_lag 11 \
      --hmc_data trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_32^2.jld2
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))

import h5py
import matplotlib.pyplot as plt
import numpy as np

from dm_mc_ess_2d import acf_per_chain_fft


def acf_single_chain(series, max_lag):
    """ACF of a 1-D series, normalised so rho(0)=1."""
    x = series - series.mean()
    var = (x * x).mean()
    if var <= 0:
        return np.ones(max_lag + 1)
    out = np.empty(max_lag + 1)
    n = len(x)
    out[0] = 1.0
    for tau in range(1, max_lag + 1):
        if tau >= n:
            out[tau] = 0.0
        else:
            out[tau] = (x[:n - tau] * x[tau:]).mean() / var
    return out


def acf_M_from_dm_chains(phi_bar, max_lag):
    """
    Per-chain ACF of |phi-bar| averaged across chains, with std-error across chains.
    phi_bar: (N+1, M).  Returns (rho_mean, rho_sem) of length max_lag+1.
    """
    series = np.abs(phi_bar[1:])                       # (N, M)
    rho = acf_per_chain_fft(series)[: max_lag + 1]     # (max_lag+1, M)
    rho_mean = rho.mean(axis=1)
    rho_sem = rho.std(axis=1, ddof=1) / np.sqrt(rho.shape[1])
    return rho_mean, rho_sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="dm_mc_ess_2D_*.npz from dm_mc_ess_2d.py")
    parser.add_argument("--max_lag", type=int, default=11)
    parser.add_argument("--hmc_data", type=str, default=None,
                        help="HMC training data .jld2 to overlay an HMC reference curve.")
    parser.add_argument("--hmc_max_samples", type=int, default=4096,
                        help="Cap on consecutive HMC cfgs used for the reference ACF.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    d = np.load(npz_path)
    L = int(d["L"])
    k = float(d["k"])
    M = int(d["num_chains"])
    N = int(d["mh_steps"])
    early_ep = int(d["early_epoch"])
    late_ep = int(d["late_epoch"])
    c = float(d["args_c"])
    t_mh = float(d["args_t_mh"])

    lags = np.arange(args.max_lag + 1)
    curves = []

    rho_e, sem_e = acf_M_from_dm_chains(d["early_phi_bar"], args.max_lag)
    curves.append(("DM-MC (early, ep " + str(early_ep) + ")", lags, rho_e, sem_e,
                   "C3", "^"))
    rho_l, sem_l = acf_M_from_dm_chains(d["late_phi_bar"], args.max_lag)
    curves.append(("DM-MC (late, ep " + str(late_ep) + ")", lags, rho_l, sem_l,
                   "C1", "o"))

    if args.hmc_data is not None:
        hmc_path = Path(args.hmc_data)
        if not hmc_path.is_absolute():
            hmc_path = HERE.parents[1] / hmc_path
        with h5py.File(hmc_path, "r") as f:
            cfgs = np.array(f["cfgs"])
        n_take = min(args.hmc_max_samples, cfgs.shape[0])
        cfgs = cfgs[:n_take]
        # Magnetisation per cfg in the original (unnormalised) field; |M| is the obs.
        M_series = np.abs(cfgs.reshape(n_take, -1).mean(axis=1))
        rho_hmc = acf_single_chain(M_series, args.max_lag)
        curves.append((f"HMC training data (n={n_take})", lags, rho_hmc, None,
                       "C2", "s"))

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, x, y, yerr, color, marker in curves:
        if yerr is None:
            ax.plot(x, y, marker=marker, ms=6, color=color, ls="-.", label=label)
        else:
            ax.errorbar(x, y, yerr=yerr, marker=marker, ms=6, color=color,
                        ls="-.", capsize=2, label=label)

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Monte-Carlo Time-Step")
    ax.set_ylabel(r"$\bar C_{|M|}(t)$")
    ax.set_xlim(-0.2, args.max_lag + 0.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(lags)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(rf"Normalised ACF of $|M|$,  $L={L},\;\kappa={k},\;\lambda={float(d['l']):g}$  "
                 rf"(DM-MC: $c={c},\;t_{{\mathrm{{mh}}}}={t_mh:g},\;M={M},\;N={N}$)")

    fig.tight_layout()
    out = Path(args.out) if args.out else npz_path.with_name(
        npz_path.stem + "_acf_M.pdf"
    )
    fig.savefig(out)
    plt.close(fig)

    # Save curves alongside the figure.
    np.savez(out.with_suffix(".npz"),
             lags=lags,
             early_rho=rho_e, early_sem=sem_e,
             late_rho=rho_l, late_sem=sem_l,
             hmc_rho=(curves[-1][2] if args.hmc_data else None),
             early_epoch=early_ep, late_epoch=late_ep,
             L=L, k=k, c=c, t_mh=t_mh, num_chains=M, mh_steps=N)
    print(f"Saved figure to {out}")
    print(f"Saved curves to {out.with_suffix('.npz')}")

    print("\n" + "=" * 56)
    print(f"{'lag':>4s}  {'DM early':>10s}  {'DM late':>10s}", end="")
    if args.hmc_data:
        print(f"  {'HMC train':>10s}", end="")
    print()
    print("=" * 56)
    for i, t in enumerate(lags):
        print(f"{t:>4d}  {rho_e[i]:>10.4f}  {rho_l[i]:>10.4f}", end="")
        if args.hmc_data:
            print(f"  {curves[-1][2][i]:>10.4f}", end="")
        print()


if __name__ == "__main__":
    main()
