"""
Diagnose tau_int of the action S(phi) under DM-MALA at a single
(L, epoch, c). Records S along the chain and reports the Sokal
estimator both at the full chain length and as a function of
truncation n, with the |M| ACF for reference.

Usage:
  python analysis/dm_mc_S_convergence_3d.py \
      --device cuda:3 --L 16 --epoch 1 --c 1.0 --mh_steps 500
"""

import argparse
import functools
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 3Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import matplotlib.pyplot as plt
import numpy as np
import torch

from dm_mc_tau_vs_epoch_3d import (
    find_epoch_checkpoints, load_model, load_reference_data,
    acf_per_chain_fft, sokal_window, cutoff_window,
)
from sample_phi4 import phi4_action_3d


@torch.no_grad()
def run_mala_record_S_and_M(model, action_fn, x0, h, t_mh, n_steps):
    """MALA chain that records S(x) and phi-bar(x) at every step."""
    device = x0.device
    M = x0.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x0.dtype)
    sqrt_2h = torch.sqrt(2 * h_t)

    spatial_dims = tuple(range(1, x0.ndim))

    def score(z):
        return model(z, torch.full((z.shape[0],), t_mh,
                                   device=device, dtype=z.dtype))

    S_traj = np.empty((n_steps + 1, M), dtype=np.float64)
    M_traj = np.empty((n_steps + 1, M), dtype=np.float64)
    accept_log = np.empty((n_steps, M), dtype=np.float32)

    x = x0.clone()
    S_traj[0] = action_fn(x).cpu().numpy()
    M_traj[0] = x.mean(dim=spatial_dims).cpu().numpy()

    for step in range(n_steps):
        score_x = score(x)
        noise = torch.randn_like(x)
        y = x + h_t * score_x + sqrt_2h * noise
        score_y = score(y)
        drift = h_t * (score_x + score_y)
        log_q = -(0.25 / h_t) * torch.sum((sqrt_2h * noise + drift) ** 2,
                                          dim=spatial_dims) \
                + 0.5 * torch.sum(noise ** 2, dim=spatial_dims)
        Sx = action_fn(x)
        Sy = action_fn(y)
        log_pi = Sx - Sy
        accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
        accept = torch.rand(M, device=device) < accept_prob
        x = torch.where(accept[(slice(None),) + (None,) * (x.ndim - 1)], y, x)
        accept_log[step] = accept.cpu().numpy().astype(np.float32)
        S_traj[step + 1] = action_fn(x).cpu().numpy()
        M_traj[step + 1] = x.mean(dim=spatial_dims).cpu().numpy()

    return S_traj, M_traj, accept_log


def tau_curve(series, n_grid, sokal_c=5.0):
    """For each n in n_grid, compute Sokal tau_int using series[:n]."""
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
    parser.add_argument("--device", required=True)
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=64)
    parser.add_argument("--mh_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=float, default=0.1923)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--data_root", type=str, default="trainingdata")
    parser.add_argument("--n_step_grid", type=int, default=20)
    parser.add_argument("--out_tag", type=str, default=None,
                        help="Override output tag. Default = "
                             "L{L}_ep{epoch}_c{c}_N{N}_seed{seed}.")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    import os
    os.chdir(str(HERE.parents[1]))

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = HERE.parents[1] / args.run_dir
    ckpts = find_epoch_checkpoints(run_dir / "models")
    available = sorted(ckpts.keys())
    ep = min(available, key=lambda e: abs(e - args.epoch))
    print(f"Using ckpt epoch={ep}")

    ref_ckpt = torch.load(ckpts[ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min")
    norm_max = hparams.get("norm_max")
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt
    print(f"pooled norm = [{norm_min:.4f}, {norm_max:.4f}], sigma_VE={sigma_ve}")

    data_path = (Path(args.data_root) /
                 f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2")
    x_ref = load_reference_data(str(data_path), args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    print(f"x_ref {tuple(x_ref.shape)}")

    h = args.c / args.L ** 3
    print(f"L={args.L}, c={args.c}, h={h:.3e}, M={args.num_chains}, "
          f"N={args.mh_steps}, t_mh={args.t_mh:g}")

    model, action_fn, _ = load_model(ckpts[ep], args.device, args.k, args.l)

    torch.manual_seed(args.seed)
    t0 = time.time()
    S_traj, M_traj, accept_log = run_mala_record_S_and_M(
        model, action_fn, x_ref, h, args.t_mh, args.mh_steps,
    )
    print(f"chain finished in {time.time()-t0:.1f}s, "
          f"overall accept = {accept_log.mean():.4f}")

    print(f"\n  S(t=0) (HMC starting cfgs):  "
          f"mean={S_traj[0].mean():.3f}, std={S_traj[0].std():.3f}")
    print(f"  S(t=N) (after {args.mh_steps} MH steps): "
          f"mean={S_traj[-1].mean():.3f}, std={S_traj[-1].std():.3f}")
    print(f"  S over full chain   : "
          f"mean={S_traj[1:].mean():.3f}, std={S_traj[1:].std():.3f}")
    chain_drift = (S_traj[-1] - S_traj[0])
    print(f"  per-chain drift S[N]-S[0]: "
          f"mean={chain_drift.mean():.3f}, max|.|={np.abs(chain_drift).max():.3f}")

    n_grid = np.arange(args.n_step_grid, args.mh_steps + 1, args.n_step_grid)
    if n_grid[-1] != args.mh_steps:
        n_grid = np.append(n_grid, args.mh_steps)

    S_series = S_traj[1:]
    M_series = np.abs(M_traj[1:])
    taus_S, Ws_S = tau_curve(S_series, n_grid)
    taus_M, Ws_M = tau_curve(M_series, n_grid)
    print(f"\n  tau_int(S) at N={args.mh_steps}:  {taus_S[-1]:.3f}  "
          f"(W={Ws_S[-1]})")
    print(f"  tau_int(|M|) at N={args.mh_steps}: {taus_M[-1]:.3f}  "
          f"(W={Ws_M[-1]})")

    rho_S = acf_per_chain_fft(S_series).mean(axis=1)
    rho_M = acf_per_chain_fft(M_series).mean(axis=1)

    out_dir = HERE.parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    tag = (args.out_tag or
           f"L{args.L}_ep{ep}_c{args.c}_N{args.mh_steps}_seed{args.seed}")
    out_pdf = out_dir / f"dm_mc_S_convergence_3d_{tag}.pdf"
    out_npz = out_dir / f"dm_mc_S_convergence_3d_{tag}.npz"

    np.savez(out_npz,
             L=args.L, k=args.k, l=args.l, epoch=ep,
             c=args.c, h=h, t_mh=args.t_mh,
             num_chains=args.num_chains, mh_steps=args.mh_steps,
             accept_rate=float(accept_log.mean()),
             S_traj=S_traj, M_traj=M_traj, accept_log=accept_log,
             n_grid=n_grid,
             taus_S=taus_S, Ws_S=Ws_S, rho_S=rho_S,
             taus_M=taus_M, Ws_M=Ws_M, rho_M=rho_M)
    print(f"Saved data to {out_npz}")

    if args.no_plot:
        return

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_traj, ax_acf, ax_tau = axes

    title_str = (rf"3D $\phi^4$ multi-L, $L={args.L}$, ep={ep}, "
                 rf"$c={args.c},\;h={h:.2e},\;t_{{\rm mh}}={args.t_mh:g}$, "
                 rf"$M={args.num_chains}\!\times\!N={args.mh_steps}$, "
                 rf"accept={accept_log.mean():.3f}")

    show_chains = min(8, args.num_chains)
    for m in range(show_chains):
        ax_traj.plot(S_traj[:, m], lw=0.7, alpha=0.7)
    ax_traj.set_xlabel("MH step")
    ax_traj.set_ylabel(r"$S(\phi)$ (per chain)")
    ax_traj.set_title(rf"Action trace ({show_chains} chains)")
    ax_traj.grid(alpha=0.3)

    # ACF: rho_S/rho_M have length N; use lags 0..min(N-1, ...).
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
                label=rf"$S$,  $\tau_{{\rm int}}(N)={taus_S[-1]:.2f}$, $W={Ws_S[-1]}$")
    ax_tau.plot(n_grid, taus_M, "o-", ms=4, color="C3",
                label=rf"$|M|$,  $\tau_{{\rm int}}(N)={taus_M[-1]:.2f}$, $W={Ws_M[-1]}$")
    ax_tau.set_xlabel("chain length $n$ (MH steps)")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(n)$")
    ax_tau.set_yscale("log")
    ax_tau.set_title(r"$\tau_{\mathrm{int}}(n)$ convergence")
    ax_tau.grid(alpha=0.3, which="both")
    ax_tau.legend(fontsize=9)

    fig.suptitle(title_str, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved figure to {out_pdf}")


if __name__ == "__main__":
    main()
