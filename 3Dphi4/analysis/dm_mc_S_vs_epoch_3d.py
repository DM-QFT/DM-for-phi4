"""
Track tau_int(S) and tau_int(|M|) vs training epoch under DM-MALA at a
single L. Records both observables along the chain so we can compare
the action (collective, mixed) with the magnetisation (Z2-symmetric,
slow) as a quality metric.

Designed to be split across GPUs by giving disjoint --epochs lists.

Usage (one worker per device):
  python analysis/dm_mc_S_vs_epoch_3d.py \
      --device cuda:0 --L 16 --c 1.0 \
      --epochs 1,2,4,7 \
      --num_chains 256 --mh_steps 1000 \
      --out_npz results/dm_mc_S_vs_epoch_3d_L16_part_a.npz
"""

import argparse
import functools
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 3Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import numpy as np
import torch

from dm_mc_tau_vs_epoch_3d import (
    find_epoch_checkpoints, load_model, load_reference_data,
    acf_per_chain_fft, sokal_window, cutoff_window,
)
from sample_phi4 import phi4_action_3d


@torch.no_grad()
def run_mala_record_S_M(model, action_fn, x0, h, t_mh, n_steps):
    """MALA chain recording S(x) and phi-bar(x) per step."""
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


def estimate_tau(series, sokal_c=5.0, acf_cutoff=0.05):
    var = series.var(axis=0)
    if (var <= 0).all():
        T = series.shape[0]
        return T / 2.0, T - 1, 0.0, T / 2.0
    rho_pc = acf_per_chain_fft(series)
    bad = ~np.isfinite(rho_pc).all(axis=0)
    if bad.any():
        rho_pc[:, bad] = 1.0
    rho_mean = rho_pc.mean(axis=1)
    tau_s, W = sokal_window(rho_mean, c=sokal_c)
    tau_cut = cutoff_window(rho_mean, cutoff=acf_cutoff)
    M, N = series.shape[1], series.shape[0]
    tau_se = float(np.sqrt(2.0 * (2 * W + 1) / (M * N)) * tau_s)
    return tau_s, W, tau_se, tau_cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--epochs", type=str, required=True,
                        help="comma-separated epoch list")
    parser.add_argument("--k", type=float, default=0.1923)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--data_root", type=str, default="trainingdata")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_npz", required=True)
    args = parser.parse_args()

    import os
    os.chdir(str(HERE.parents[1]))

    epochs_req = [int(e) for e in args.epochs.split(",") if e.strip()]
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = HERE.parents[1] / args.run_dir
    ckpts = find_epoch_checkpoints(run_dir / "models")
    available = sorted(ckpts.keys())
    epochs = sorted(set(min(available, key=lambda e: abs(e - tgt)) for tgt in epochs_req))
    print(f"[{args.device}] {len(epochs)} epochs: {epochs}")

    ref_ckpt = torch.load(ckpts[epochs[0]], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min")
    norm_max = hparams.get("norm_max")
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt
    print(f"[{args.device}] pooled norm = [{norm_min:.4f}, {norm_max:.4f}], "
          f"sigma_VE = {sigma_ve}")

    data_path = (Path(args.data_root) /
                 f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2")
    x_ref = load_reference_data(str(data_path), args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    print(f"[{args.device}] x_ref {tuple(x_ref.shape)}")

    h = args.c / args.L ** 3
    print(f"[{args.device}] L={args.L}, c={args.c}, h={h:.3e}, "
          f"M={args.num_chains}, N={args.mh_steps}\n")

    n_e = len(epochs)
    res_tau_S = np.full(n_e, np.nan)
    res_se_S = np.full(n_e, np.nan)
    res_W_S = np.full(n_e, np.nan)
    res_cut_S = np.full(n_e, np.nan)
    res_tau_M = np.full(n_e, np.nan)
    res_se_M = np.full(n_e, np.nan)
    res_W_M = np.full(n_e, np.nan)
    res_cut_M = np.full(n_e, np.nan)
    res_acc = np.full(n_e, np.nan)
    rho_S_all = np.full((n_e, args.mh_steps), np.nan)
    rho_M_all = np.full((n_e, args.mh_steps), np.nan)

    torch.manual_seed(args.seed)
    for ie, ep in enumerate(epochs):
        t0 = time.time()
        model, _, _ = load_model(ckpts[ep], args.device, args.k, args.l)
        action_fn = functools.partial(phi4_action_3d, k=args.k, l=args.l,
                                      phi_min=norm_min, phi_max=norm_max)
        S_traj, M_traj, accept_log = run_mala_record_S_M(
            model, action_fn, x_ref, h, args.t_mh, args.mh_steps,
        )
        accept_rate = float(accept_log.mean())

        S_series = S_traj[1:]
        M_series = np.abs(M_traj[1:])
        tau_S, W_S, se_S, cut_S = estimate_tau(S_series)
        tau_M, W_M, se_M, cut_M = estimate_tau(M_series)
        rho_S_all[ie] = acf_per_chain_fft(S_series).mean(axis=1) \
            if S_series.var(axis=0).any() else np.ones(args.mh_steps)
        rho_M_all[ie] = acf_per_chain_fft(M_series).mean(axis=1) \
            if M_series.var(axis=0).any() else np.ones(args.mh_steps)

        res_tau_S[ie], res_se_S[ie], res_W_S[ie], res_cut_S[ie] = tau_S, se_S, W_S, cut_S
        res_tau_M[ie], res_se_M[ie], res_W_M[ie], res_cut_M[ie] = tau_M, se_M, W_M, cut_M
        res_acc[ie] = accept_rate
        print(f"[{args.device}] ep={ep:>5d}  acc={accept_rate:.3f}  "
              f"tau_S={tau_S:7.2f}+/-{se_S:5.2f} W_S={int(W_S):>4d}  "
              f"tau_|M|={tau_M:7.2f}+/-{se_M:5.2f} W_M={int(W_M):>4d}  "
              f"({time.time()-t0:.1f}s)")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = Path(args.out_npz)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        out,
        epochs=np.array(epochs),
        tau_S=res_tau_S, se_S=res_se_S, W_S=res_W_S, cut_S=res_cut_S,
        tau_M=res_tau_M, se_M=res_se_M, W_M=res_W_M, cut_M=res_cut_M,
        accept=res_acc,
        rho_S=rho_S_all, rho_M=rho_M_all,
        c=args.c, t_mh=args.t_mh, k=args.k, l=args.l, L=args.L,
        num_chains=args.num_chains, mh_steps=args.mh_steps,
        norm_min=norm_min, norm_max=norm_max, sigma_ve=sigma_ve,
    )
    print(f"[{args.device}] Saved -> {out}")


if __name__ == "__main__":
    main()
