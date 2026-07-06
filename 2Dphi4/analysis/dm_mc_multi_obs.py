"""
Run a MALA chain (DM-driven) at a fixed (c, t_mh) and record several
observables along the chain to compare their ACFs and tau_int.

Observables:
  |M| = |phi-bar|                         global, Z2-symmetric, slow
  phi_site                                 single-site, local, fast
  phi^2_avg = <phi^2>_x                    global, Z2-symmetric, mixed
  S(phi)                                   total action, global mixed
  G(k_min) = |phi-tilde(k_min)|^2          long-wavelength Fourier mode
  G(k_max) = |phi-tilde(k_max)|^2          short-wavelength Fourier mode

If all curves saturate at ~N/12, tau_int is estimator-limited (chain too
short). If short-wavelength observables decorrelate in O(1) steps while
|M| stays slow, the slow part is physical.

Usage (from 2Dphi4/):
  python analysis/dm_mc_multi_obs.py \
      --device cuda:3 --c 0.5 --mh_steps 500
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 2Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import matplotlib.pyplot as plt
import numpy as np
import torch

from dm_mc_ess_2d import (
    find_epoch_checkpoints, load_model, load_reference_data,
    acf_per_chain_fft, sokal_window, cutoff_window,
)


def compute_observables(x, action_fn, k_min_idx, k_max_idx):
    """
    Compute per-chain observables.
    x: (M, 1, L, L) torch tensor (normalised space)
    Returns dict of length-M numpy arrays.
    """
    spatial = (-2, -1)
    phi = x.squeeze(1)                                  # (M, L, L)
    M_field = phi.mean(dim=spatial)                     # signed magnetisation
    phi_sq = (phi ** 2).mean(dim=spatial)
    phi_site = phi[..., 0, 0]
    S = action_fn(x)
    phi_k = torch.fft.fftn(phi, dim=spatial, norm="ortho")
    G = phi_k.real ** 2 + phi_k.imag ** 2               # (M, L, L)
    Gk_min = G[..., k_min_idx[0], k_min_idx[1]]
    Gk_max = G[..., k_max_idx[0], k_max_idx[1]]

    return {
        "absM":      M_field.abs().detach().cpu().numpy(),
        "phi_site":  phi_site.detach().cpu().numpy(),
        "phi_sq":    phi_sq.detach().cpu().numpy(),
        "S":         S.detach().cpu().numpy(),
        "G_kmin":    Gk_min.detach().cpu().numpy(),
        "G_kmax":    Gk_max.detach().cpu().numpy(),
    }


@torch.no_grad()
def run_mala_with_obs(model, action_fn, x0, h, t_mh, n_steps, L,
                     log_every=100):
    """MALA chain that records multiple observables at every step."""
    device = x0.device
    M = x0.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x0.dtype)
    sqrt_2h = torch.sqrt(2 * h_t)

    def score(z):
        return model(z, torch.full((z.shape[0],), t_mh,
                                   device=device, dtype=z.dtype))

    spatial_dims = tuple(range(1, x0.ndim))

    # k indices for Fourier observables.
    k_min_idx = (1, 0)
    k_max_idx = (L // 2, L // 2)

    # Pre-allocate observable storage.
    obs_names = ["absM", "phi_site", "phi_sq", "S", "G_kmin", "G_kmax"]
    obs_traj = {name: np.empty((n_steps + 1, M), dtype=np.float64)
                for name in obs_names}

    def record(t, x):
        oc = compute_observables(x, action_fn, k_min_idx, k_max_idx)
        for name in obs_names:
            obs_traj[name][t] = oc[name]

    record(0, x0)

    x = x0.clone()
    accept_log = np.empty((n_steps, M), dtype=np.float32)

    for step in range(n_steps):
        score_x = score(x)
        noise = torch.randn_like(x)
        y = x + h_t * score_x + sqrt_2h * noise
        score_y = score(y)
        drift = h_t * (score_x + score_y)
        log_q = -(0.25 / h_t) * torch.sum((sqrt_2h * noise + drift) ** 2,
                                          dim=spatial_dims) \
                + 0.5 * torch.sum(noise ** 2, dim=spatial_dims)
        log_pi = action_fn(x) - action_fn(y)
        accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
        accept = torch.rand(M, device=device) < accept_prob
        x = torch.where(accept[(slice(None),) + (None,) * (x.ndim - 1)], y, x)
        accept_log[step] = accept.detach().cpu().numpy().astype(np.float32)
        record(step + 1, x)

        if log_every and ((step + 1) % log_every == 0 or step == 0):
            print(f"  step {step+1:>5d}/{n_steps}  "
                  f"cumulative accept = {accept_log[: step + 1].mean():.4f}")

    return obs_traj, accept_log, k_min_idx, k_max_idx


def estimate_tau(series, sokal_c=5.0, acf_cutoff=0.05):
    """
    series: (n_steps, M). Returns dict with rho_mean, rho_sem, tau_sokal,
    sokal_W, tau_se, tau_cutoff.
    """
    var = series.var(axis=0)
    if (var <= 0).all():
        T = series.shape[0]
        return dict(rho=np.ones(T), rho_sem=np.zeros(T),
                    tau_sokal=T / 2.0, sokal_W=T - 1,
                    tau_se=0.0, tau_cutoff=T / 2.0)
    rho_pc = acf_per_chain_fft(series)
    bad = ~np.isfinite(rho_pc).all(axis=0)
    if bad.any():
        rho_pc[:, bad] = 1.0
    rho_mean = rho_pc.mean(axis=1)
    rho_sem = rho_pc.std(axis=1, ddof=1) / np.sqrt(rho_pc.shape[1])
    tau_s, W, _ = sokal_window(rho_mean, c=sokal_c)
    tau_cut = cutoff_window(rho_mean, cutoff=acf_cutoff)
    M, N = series.shape[1], series.shape[0]
    tau_se = float(np.sqrt(2.0 * (2 * W + 1) / (M * N)) * tau_s)
    return dict(rho=rho_mean, rho_sem=rho_sem,
                tau_sokal=tau_s, sokal_W=W,
                tau_se=tau_se, tau_cutoff=tau_cut)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_L32_k0.28_l0.022_ncsnpp_sigma180")
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--k", type=float, default=0.28)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=7564)
    parser.add_argument("--c", type=float, default=0.5,
                        help="MALA step coefficient, h = c/L^2.")
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lag_plot", type=int, default=80)
    parser.add_argument("--out_tag", type=str, default=None)
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

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"

    ref_ckpt = torch.load(ckpts[ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt
    print(f"sigma_VE={sigma_ve}, norm range [{norm_min:.3f}, {norm_max:.3f}]")

    x_ref = load_reference_data(args.data_path, args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    print(f"x_ref: {tuple(x_ref.shape)}")

    model, action_fn, _ = load_model(ckpts[ep], args.device)

    h = args.c / args.L ** 2
    print(f"\nMALA: c={args.c}, h={h:.3e}, t_mh={args.t_mh:g}, "
          f"M={args.num_chains}, N={args.mh_steps}")

    torch.manual_seed(args.seed)
    t0 = time.time()
    obs_traj, accept_log, k_min_idx, k_max_idx = run_mala_with_obs(
        model, action_fn, x_ref, h, args.t_mh, args.mh_steps, args.L,
        log_every=100,
    )
    print(f"\nChain finished in {time.time()-t0:.1f}s, "
          f"overall accept = {accept_log.mean():.4f}")

    obs_labels = {
        "absM":     r"$|M|$",
        "phi_site": rf"$\phi_{{x_0}}$ (single site)",
        "phi_sq":   r"$\langle \phi^2\rangle$",
        "S":        r"$S(\phi)$",
        "G_kmin":   rf"$G(k_{{\min}})$, $k=({k_min_idx[0]},{k_min_idx[1]})$",
        "G_kmax":   rf"$G(k_{{\max}})$, $k=({k_max_idx[0]},{k_max_idx[1]})$",
    }
    obs_colors = {
        "absM": "C0", "phi_site": "C1", "phi_sq": "C2",
        "S": "C3", "G_kmin": "C4", "G_kmax": "C5",
    }

    print(f"\n{'observable':>16s}  {'tau_sokal':>10s}  {'tau_se':>7s}  "
          f"{'sokal_W':>8s}  {'tau_cut':>8s}")
    print("-" * 60)

    results = {}
    for name in obs_traj:
        series = obs_traj[name][1:]                    # drop initial (training data) value
        r = estimate_tau(series)
        results[name] = r
        print(f"{name:>16s}  {r['tau_sokal']:10.3f}  {r['tau_se']:7.2f}  "
              f"{int(r['sokal_W']):>8d}  {r['tau_cutoff']:8.3f}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9,
        "lines.linewidth": 1.5, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_short, ax_long = axes

    L_short = min(args.max_lag_plot, args.mh_steps)
    L_long = args.mh_steps

    for name, r in results.items():
        col = obs_colors[name]
        lbl = (f"{obs_labels[name]}: "
               rf"$\tau_{{\rm int}}={r['tau_sokal']:.1f}\pm{r['tau_se']:.1f}$, "
               rf"$W={int(r['sokal_W'])}$")
        x_short = np.arange(L_short + 1)
        ax_short.plot(x_short, r["rho"][: L_short + 1], "-", color=col, label=lbl)
        ax_long.plot(np.arange(L_long), r["rho"][: L_long], "-", color=col,
                     alpha=0.85)

    ax_short.axhline(0, color="k", lw=0.5)
    ax_short.axhline(0.05, color="k", lw=0.5, ls=":", alpha=0.5)
    ax_short.set_xlabel("Monte-Carlo Time-Step")
    ax_short.set_ylabel(r"$\bar C_\mathcal{O}(t)$")
    ax_short.set_xlim(-0.2, L_short + 0.2)
    ax_short.set_ylim(-0.1, 1.05)
    ax_short.grid(alpha=0.3)
    ax_short.set_title(f"Short-lag ACF (lag 0..{L_short})")
    ax_short.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax_long.axhline(0, color="k", lw=0.5)
    ax_long.set_xlabel("Monte-Carlo Time-Step")
    ax_long.set_ylabel(r"$\bar C_\mathcal{O}(t)$")
    ax_long.set_ylim(-0.2, 1.05)
    ax_long.grid(alpha=0.3)
    ax_long.set_title(f"Full ACF (lag 0..{L_long})")

    fig.suptitle(
        rf"DM-MALA, 2D $\phi^4$, $L={args.L},\;\kappa={args.k},\;\lambda={args.l:g}$, "
        rf"ckpt ep {ep},  $c={args.c},\;t_{{\rm mh}}={args.t_mh:g}$, "
        rf"$M={args.num_chains}\times N={args.mh_steps}$, "
        rf"accept = {accept_log.mean():.3f}",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()

    tag = args.out_tag or f"ep{ep}_c{args.c}"
    out_dir = HERE.parents[1] / "results"
    out_pdf = out_dir / f"dm_mc_multi_obs_L{args.L}_k{args.k}_{tag}.pdf"
    out_npz = out_dir / f"dm_mc_multi_obs_L{args.L}_k{args.k}_{tag}.npz"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nSaved figure to {out_pdf}")

    save = dict(c=args.c, h=h, t_mh=args.t_mh, L=args.L, k=args.k, l=args.l,
                epoch=ep, num_chains=args.num_chains, mh_steps=args.mh_steps,
                accept_rate=float(accept_log.mean()),
                k_min_idx=np.array(k_min_idx), k_max_idx=np.array(k_max_idx))
    for name in obs_traj:
        save[f"{name}_traj"] = obs_traj[name]
        for key, val in results[name].items():
            save[f"{name}_{key}"] = val
    np.savez(out_npz, **save)
    print(f"Saved data to {out_npz}")


if __name__ == "__main__":
    main()
