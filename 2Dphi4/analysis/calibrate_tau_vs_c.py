"""
For a fixed late ckpt, sweep MALA step coefficient c (with h = c/L^2) and
measure tau_int of |M| = |phi-bar|. Designed to be launched in parallel
on multiple GPUs by partitioning the c list across workers.

The starting cfgs (256 configs from HMC training data) are the same for
every c, so curves are directly comparable.

Usage (one worker per device):
  python analysis/calibrate_tau_vs_c.py \
      --device cuda:2 \
      --c_values 0.01,0.05,0.1,0.2,1.0 \
      --out_npz results/calib_tau_c_part_a.npz \
      --mh_steps 500
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 2Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import numpy as np
import torch

from dm_mc_ess_2d import (
    find_epoch_checkpoints, load_model, load_reference_data,
    run_mala_chain, acf_per_chain_fft, sokal_window, cutoff_window,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--c_values", type=str, required=True,
                        help="comma-separated list, e.g. 0.01,0.05,0.1")
    parser.add_argument("--out_npz", type=str, required=True)
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_L32_k0.28_l0.022_ncsnpp_sigma180")
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--k", type=float, default=0.28)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=7564,
                        help="Single ckpt to test on (closest available).")
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sokal_c", type=float, default=5.0)
    parser.add_argument("--acf_cutoff", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    import os
    os.chdir(str(HERE.parents[1]))

    c_list = [float(x) for x in args.c_values.split(",") if x.strip()]
    print(f"[{args.device}] c values: {c_list}")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = HERE.parents[1] / args.run_dir
    ckpts = find_epoch_checkpoints(run_dir / "models")
    available = sorted(ckpts.keys())
    ep = min(available, key=lambda e: abs(e - args.epoch))
    print(f"[{args.device}] Using ckpt epoch={ep}")

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"

    ref_ckpt = torch.load(ckpts[ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt

    x_ref = load_reference_data(args.data_path, args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    print(f"[{args.device}] x_ref shape {tuple(x_ref.shape)}, sigma_VE={sigma_ve}")

    model, action_fn, _ = load_model(ckpts[ep], args.device)

    L = args.L
    M = args.num_chains
    N = args.mh_steps

    n_c = len(c_list)
    res_tau_sokal = np.full(n_c, np.nan)
    res_tau_se = np.full(n_c, np.nan)
    res_W = np.full(n_c, np.nan)
    res_tau_cutoff = np.full(n_c, np.nan)
    res_accept = np.full(n_c, np.nan)
    res_rho = np.full((n_c, N), np.nan)

    torch.manual_seed(args.seed)
    for i, c in enumerate(c_list):
        h = c / L ** 2
        t0 = time.time()
        phi_bar, accept_log = run_mala_chain(
            model, action_fn, x_ref, h, args.t_mh, N,
            batch_size=args.batch_size, log_every=0,
        )
        dt = time.time() - t0
        accept_rate = float(accept_log.mean())

        series = np.abs(phi_bar[1:])               # |M| timeseries (N, M)
        # Guard against constant series (acf would be NaN).
        var_per_chain = series.var(axis=0)
        if (var_per_chain <= 0).all():
            rho = np.ones(N)
            tau_s, W, _ = N / 2.0, N - 1, np.full(N - 1, np.nan)
            tau_cut = N / 2.0
            tau_se = 0.0
        else:
            rho_pc = acf_per_chain_fft(series)
            # Replace any NaN per-chain rho (constant chain) with 1 to be safe.
            bad = ~np.isfinite(rho_pc).all(axis=0)
            if bad.any():
                rho_pc[:, bad] = 1.0
            rho = rho_pc.mean(axis=1)
            tau_s, W, _ = sokal_window(rho, c=args.sokal_c)
            tau_cut = cutoff_window(rho, cutoff=args.acf_cutoff)
            tau_se = float(np.sqrt(2.0 * (2 * W + 1) / (M * N)) * tau_s)

        res_tau_sokal[i] = tau_s
        res_tau_se[i] = tau_se
        res_W[i] = W
        res_tau_cutoff[i] = tau_cut
        res_accept[i] = accept_rate
        res_rho[i] = rho

        print(f"[{args.device}] c={c:>6.3f}  h={h:.2e}  accept={accept_rate:.3f}  "
              f"tau_sokal={tau_s:7.2f}  W={int(W):>4d}  tau_cut={tau_cut:7.2f}  "
              f"({dt:.1f}s)")

    out = Path(args.out_npz)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(out,
             c_values=np.array(c_list),
             tau_sokal=res_tau_sokal, tau_se=res_tau_se,
             sokal_W=res_W, tau_cutoff=res_tau_cutoff,
             accept=res_accept, rho=res_rho,
             L=L, k=args.k, l=args.l, t_mh=args.t_mh,
             epoch=ep, num_chains=M, mh_steps=N,
             sigma_ve=sigma_ve)
    print(f"[{args.device}] Saved -> {out}")


if __name__ == "__main__":
    main()
