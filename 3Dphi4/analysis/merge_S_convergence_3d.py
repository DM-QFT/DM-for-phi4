"""
Merge worker NPZs from dm_mc_S_convergence_3d.py (each worker = different
seed, M chains each). Concatenate chains across workers, recompute the
chain-averaged ACF and tau_int(n) curve at the higher chain count.

Usage:
  python analysis/merge_S_convergence_3d.py \
      --inputs results/dm_mc_S_convergence_3d_L16_ep1_c1.0_N2000_part_*.npz \
      --out results/dm_mc_S_convergence_3d_L16_ep1_c1.0_N2000_merged.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dm_mc_tau_vs_epoch_3d import acf_per_chain_fft, sokal_window


def tau_curve(series, n_grid, sokal_c=5.0):
    taus = np.empty(len(n_grid))
    Ws = np.empty(len(n_grid), dtype=int)
    for i, n in enumerate(n_grid):
        seg = series[:n]
        var = seg.var(axis=0)
        if (var <= 0).all():
            taus[i] = n / 2.0
            Ws[i] = n - 1
            continue
        rho_pc = acf_per_chain_fft(seg)
        bad = ~np.isfinite(rho_pc).all(axis=0)
        if bad.any():
            rho_pc[:, bad] = 1.0
        rho_mean = rho_pc.mean(axis=1)
        tau, W = sokal_window(rho_mean, c=sokal_c)
        taus[i] = tau
        Ws[i] = W
    return taus, Ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_step_grid", type=int, default=20)
    args = parser.parse_args()

    S_list, M_list, acc_list = [], [], []
    meta = None
    for path in args.inputs:
        d = np.load(path)
        S_list.append(d["S_traj"])      # (N+1, M)
        M_list.append(d["M_traj"])
        acc_list.append(float(d["accept_rate"]))
        if meta is None:
            meta = dict(L=int(d["L"]), k=float(d["k"]), l=float(d["l"]),
                        epoch=int(d["epoch"]), c=float(d["c"]),
                        h=float(d["h"]), t_mh=float(d["t_mh"]),
                        mh_steps=int(d["mh_steps"]),
                        num_chains_per_worker=int(d["num_chains"]))

    S_traj = np.concatenate(S_list, axis=1)        # (N+1, M_total)
    M_traj = np.concatenate(M_list, axis=1)
    M_total = S_traj.shape[1]
    N = meta["mh_steps"]
    accept_overall = float(np.mean(acc_list))
    print(f"Merged {len(args.inputs)} workers -> M_total={M_total} chains, "
          f"N={N}, accept_avg={accept_overall:.4f}")

    n_grid = np.arange(args.n_step_grid, N + 1, args.n_step_grid)
    if n_grid[-1] != N:
        n_grid = np.append(n_grid, N)

    S_series = S_traj[1:]
    M_series = np.abs(M_traj[1:])
    taus_S, Ws_S = tau_curve(S_series, n_grid)
    taus_M, Ws_M = tau_curve(M_series, n_grid)
    rho_S = acf_per_chain_fft(S_series).mean(axis=1)
    rho_M = acf_per_chain_fft(M_series).mean(axis=1)

    print(f"\n  tau_int(S) at N={N}:  {taus_S[-1]:.3f}  (W={Ws_S[-1]})")
    print(f"  tau_int(|M|) at N={N}: {taus_M[-1]:.3f}  (W={Ws_M[-1]})")

    # Chain-level diagnostics
    var_S_inter = S_traj[1:].mean(axis=0).var()
    var_S_intra = S_traj[1:].var(axis=0).mean()
    print(f"\n  S inter-chain var (across-chain time-mean): {var_S_inter:.4e}")
    print(f"  S intra-chain var (mean within-chain var):  {var_S_intra:.4e}")
    drift = S_traj[-1] - S_traj[0]
    print(f"  S drift S[N]-S[0]:  mean={drift.mean():.3f}, "
          f"max|.|={np.abs(drift).max():.3f}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_traj, ax_acf, ax_tau = axes

    title_str = (rf"3D $\phi^4$ multi-L, $L={meta['L']}$, ep={meta['epoch']}, "
                 rf"$c={meta['c']},\;h={meta['h']:.2e}$, "
                 rf"$M_\text{{total}}={M_total}$ chains $\times\;N={N}$, "
                 rf"accept={accept_overall:.3f}")

    show_chains = min(8, M_total)
    for m in range(show_chains):
        ax_traj.plot(S_traj[:, m], lw=0.7, alpha=0.7)
    ax_traj.set_xlabel("MH step")
    ax_traj.set_ylabel(r"$S(\phi)$ per chain")
    ax_traj.set_title(rf"Action trace ({show_chains} chains shown)")
    ax_traj.grid(alpha=0.3)

    N_acf = len(rho_S)
    max_lag_show = min(N_acf - 1, max(50, 5 * max(Ws_S[-1], Ws_M[-1])))
    lags = np.arange(max_lag_show + 1)
    ax_acf.plot(lags, rho_S[: max_lag_show + 1], color="C0", label=r"$S$")
    ax_acf.plot(lags, rho_M[: max_lag_show + 1], color="C3", label=r"$|M|$")
    ax_acf.axhline(0, color="k", lw=0.5)
    ax_acf.set_xlabel(r"lag $\tau$ (MH steps)")
    ax_acf.set_ylabel(r"$\bar\rho(\tau)$")
    ax_acf.set_title("Chain-averaged ACF")
    ax_acf.legend()
    ax_acf.grid(alpha=0.3)

    ax_tau.plot(n_grid, taus_S, "o-", ms=4, color="C0",
                label=rf"$S$,  $\tau(N)={taus_S[-1]:.2f}$, $W={Ws_S[-1]}$")
    ax_tau.plot(n_grid, taus_M, "o-", ms=4, color="C3",
                label=rf"$|M|$,  $\tau(N)={taus_M[-1]:.2f}$, $W={Ws_M[-1]}$")
    ax_tau.set_xlabel("chain length $n$")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(n)$")
    ax_tau.set_yscale("log")
    ax_tau.set_title(r"$\tau_{\mathrm{int}}(n)$ convergence")
    ax_tau.grid(alpha=0.3, which="both")
    ax_tau.legend(fontsize=9)

    fig.suptitle(title_str, y=1.02, fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    np.savez(out.with_suffix(".npz"),
             S_traj=S_traj, M_traj=M_traj,
             n_grid=n_grid,
             taus_S=taus_S, Ws_S=Ws_S, rho_S=rho_S,
             taus_M=taus_M, Ws_M=Ws_M, rho_M=rho_M,
             accept_overall=accept_overall, **meta)
    print(f"Saved data to {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
