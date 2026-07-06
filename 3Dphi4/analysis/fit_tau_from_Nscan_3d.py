"""
Run MALA at L=16 for selected (early-to-late) epochs and record S_traj
fully. For each epoch, compute the cumulative ACF integral

    T_S(n) = 1/2 + sum_{t=1}^{n-1} rho_S(t)

and fit the saturating-exponential model

    T_S(n; tau, A) = 1/2 + A * tau * (1 - exp(-(n-1)/tau))

to extract tau_true. This works for early epochs where Sokal saturates
(linear T_S(n) growth) provided the chain has SOME mixing -- the fit
exploits the slope/curvature of the cumulative integral, not the
absolute Sokal-windowed value.

Usage (one worker per device):
  python analysis/fit_tau_from_Nscan_3d.py \
      --device cuda:0 --L 16 \
      --epochs 1,200 \
      --num_chains 128 --mh_steps 3000 \
      --out_npz results/dm_mc_S_Nscan_L16_part_a.npz
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
from scipy.optimize import curve_fit

from dm_mc_tau_vs_epoch_3d import (
    find_epoch_checkpoints, load_model, load_reference_data,
    acf_per_chain_fft,
)
from dm_mc_S_vs_epoch_3d import run_mala_record_S_M
from sample_phi4 import phi4_action_3d


def cumulative_T(rho, n_max=None):
    """T(n) = 1/2 + sum_{t=1..n-1} rho(t)  for n = 1..n_max."""
    n_max = n_max or len(rho)
    cum = np.cumsum(rho[:n_max])
    T = 0.5 + cum - rho[0]      # subtract rho[0] (=1) so sum starts at t=1
    T[0] = 0.5                  # T(1) = 1/2 (just the lag-0 prefactor)
    return T


def fit_saturating(n, T, n_min=10, n_max=None):
    """
    Fit T(n) = 0.5 + A*tau*(1 - exp(-(n-1)/tau)) for n in [n_min, n_max].
    Returns (tau, A, sigma_tau) or NaNs on failure.
    """
    if n_max is None:
        n_max = len(T)
    mask = (n >= n_min) & (n <= n_max) & np.isfinite(T)
    if mask.sum() < 6:
        return np.nan, np.nan, np.nan
    nf = n[mask].astype(float)
    Tf = T[mask]

    def model(x, tau, A):
        return 0.5 + A * tau * (1.0 - np.exp(-(x - 1) / tau))

    # Initial guess: A=1, tau = 2*(T[-1]-T[0]) (rough)
    tau0 = max(10.0, (Tf[-1] - Tf[0]))
    try:
        popt, pcov = curve_fit(model, nf, Tf, p0=(tau0, 1.0),
                               bounds=([0.1, 0.01], [1e9, 5.0]),
                               maxfev=5000)
        tau, A = float(popt[0]), float(popt[1])
        sigma_tau = float(np.sqrt(pcov[0, 0]))
    except Exception:
        return np.nan, np.nan, np.nan
    return tau, A, sigma_tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_3d_L16_k0.1923_l0.9_ncsnpp_sigma280")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--epochs", type=str, required=True)
    parser.add_argument("--k", type=float, default=0.1923)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--data_root", type=str, default="trainingdata")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=128)
    parser.add_argument("--mh_steps", type=int, default=3000)
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
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt

    data_path = (Path(args.data_root) /
                 f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2")
    x_ref = load_reference_data(str(data_path), args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    h = args.c / args.L ** 3

    n_e = len(epochs)
    N = args.mh_steps
    M = args.num_chains

    rho_S_all = np.zeros((n_e, N))
    T_S_all = np.zeros((n_e, N))
    accept_all = np.zeros(n_e)
    tau_fit = np.full(n_e, np.nan)
    A_fit = np.full(n_e, np.nan)
    sigma_fit = np.full(n_e, np.nan)

    torch.manual_seed(args.seed)
    for ie, ep in enumerate(epochs):
        t0 = time.time()
        model, _, _ = load_model(ckpts[ep], args.device, args.k, args.l)
        action_fn = functools.partial(phi4_action_3d, k=args.k, l=args.l,
                                      phi_min=norm_min, phi_max=norm_max)
        S_traj, M_traj, accept_log = run_mala_record_S_M(
            model, action_fn, x_ref, h, args.t_mh, N,
        )
        accept_rate = float(accept_log.mean())
        accept_all[ie] = accept_rate

        S_series = S_traj[1:]                     # (N, M)
        if S_series.var(axis=0).any():
            rho_pc = acf_per_chain_fft(S_series)
            bad = ~np.isfinite(rho_pc).all(axis=0)
            if bad.any():
                rho_pc[:, bad] = 1.0
            rho = rho_pc.mean(axis=1)
        else:
            rho = np.ones(N)
        rho_S_all[ie] = rho
        T = cumulative_T(rho)
        T_S_all[ie] = T

        n_arr = np.arange(1, N + 1)
        tau, A, sigma = fit_saturating(n_arr, T, n_min=10)
        tau_fit[ie] = tau
        A_fit[ie] = A
        sigma_fit[ie] = sigma

        if np.isfinite(tau):
            tau_str = f"tau={tau:9.2f}+/-{sigma:6.2f} A={A:5.2f}"
        else:
            tau_str = "fit failed"
        print(f"[{args.device}] ep={ep:>5d}  acc={accept_rate:.4f}  "
              f"T(N)={T[-1]:7.2f}  {tau_str}  ({time.time()-t0:.1f}s)")

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = Path(args.out_npz)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        out,
        epochs=np.array(epochs),
        accept=accept_all,
        rho_S=rho_S_all,
        T_S=T_S_all,
        tau_fit=tau_fit, A_fit=A_fit, sigma_fit=sigma_fit,
        L=args.L, k=args.k, l=args.l,
        c=args.c, t_mh=args.t_mh,
        num_chains=M, mh_steps=N,
    )
    print(f"[{args.device}] Saved -> {out}")


if __name__ == "__main__":
    main()
