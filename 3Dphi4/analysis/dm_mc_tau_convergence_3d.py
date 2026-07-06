"""
Diagnose whether the early-epoch tau_int values reported by
dm_mc_tau_vs_epoch_3d.py (which often hit N/2 saturation) are
chain-length artefacts or real.

Run a longer chain (default N=500) at a fixed L for a small set of
selected epochs, and for each epoch plot tau_int(n) for n increasing
from a small initial chunk up to N. A "good" estimate plateaus once
n >> tau_int; a saturating curve indicates the chain is too short.

Usage (from 3Dphi4/):
  python analysis/dm_mc_tau_convergence_3d.py \
      --device cuda:3 --L 16 \
      --epochs 1,54,104,200,385,10000 \
      --num_chains 64 --mh_steps 500
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
    run_mala_chain_3d, acf_per_chain_fft, sokal_window, cutoff_window,
)
from sample_phi4 import phi4_action_3d


def tau_int_at_n(series_full, n, sokal_c=5.0):
    series = series_full[:n]
    var = series.var(axis=0)
    if (var <= 0).all():
        return n / 2.0, n - 1
    rho_pc = acf_per_chain_fft(series)
    bad = ~np.isfinite(rho_pc).all(axis=0)
    if bad.any():
        rho_pc[:, bad] = 1.0
    rho_mean = rho_pc.mean(axis=1)
    return sokal_window(rho_mean, c=sokal_c)


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
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=64)
    parser.add_argument("--mh_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_step_grid", type=int, default=20,
                        help="stride on n for the convergence curve.")
    parser.add_argument("--out_tag", type=str, default=None)
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip per-worker plot; only save NPZ.")
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
    print(f"Epochs: {epochs}")

    ref_ckpt = torch.load(ckpts[epochs[0]], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min")
    norm_max = hparams.get("norm_max")
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt
    print(f"pooled norm = [{norm_min:.4f}, {norm_max:.4f}], sigma_VE = {sigma_ve}")

    data_path = (Path(args.data_root) /
                 f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2")
    x_ref = load_reference_data(str(data_path), args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    print(f"x_ref {tuple(x_ref.shape)}")

    h = args.c / args.L ** 3
    print(f"L={args.L}, h={h:.3e}, M={args.num_chains}, N={args.mh_steps}")

    n_grid = np.arange(args.n_step_grid, args.mh_steps + 1, args.n_step_grid)
    if n_grid[-1] != args.mh_steps:
        n_grid = np.append(n_grid, args.mh_steps)
    print(f"n_grid: {n_grid[0]}..{n_grid[-1]} (step={args.n_step_grid}, "
          f"{len(n_grid)} pts)\n")

    results = {}
    torch.manual_seed(args.seed)
    for ep in epochs:
        t0 = time.time()
        model, _, _ = load_model(ckpts[ep], args.device, args.k, args.l)
        action_fn = functools.partial(phi4_action_3d, k=args.k, l=args.l,
                                      phi_min=norm_min, phi_max=norm_max)
        phi_bar, accept_log = run_mala_chain_3d(
            model, action_fn, x_ref, h, args.t_mh, args.mh_steps,
        )
        accept_rate = float(accept_log.mean())
        absM = np.abs(phi_bar[1:])                # (N, M)

        taus = np.empty(len(n_grid))
        Ws = np.empty(len(n_grid), dtype=int)
        for i, n in enumerate(n_grid):
            tau, W = tau_int_at_n(absM, n)
            taus[i] = tau
            Ws[i] = W

        results[ep] = dict(
            accept=accept_rate, taus=taus, Ws=Ws,
            phi_bar=phi_bar, n_grid=n_grid,
        )
        print(f"ep={ep:>5d}  accept={accept_rate:.3f}  "
              f"tau(N={args.mh_steps})={taus[-1]:6.2f}  W={Ws[-1]:>4d}  "
              f"({time.time()-t0:.1f}s)")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_tau, ax_ratio = axes

    for i, ep in enumerate(epochs):
        col = cmap(i / max(1, len(epochs) - 1))
        r = results[ep]
        lbl = rf"ep={ep}  (accept={r['accept']:.3f})"
        ax_tau.plot(r["n_grid"], r["taus"], marker="o", ms=4, color=col, label=lbl)
        ratio = r["n_grid"] / r["taus"]
        ax_ratio.plot(r["n_grid"], ratio, marker="o", ms=4, color=col, label=lbl)

    ax_tau.set_xlabel("chain length $n$ (MH steps used)")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(|M|)$")
    ax_tau.set_yscale("log")
    ax_tau.set_title(rf"$L={args.L}$,  $\tau_{{\rm int}}(n)$ convergence")
    ax_tau.grid(alpha=0.3, which="both")
    ax_tau.legend(fontsize=9, loc="best")

    ax_ratio.axhline(5.0, color="k", lw=1, ls=":",
                     label=r"Sokal threshold $n/\tau\geq 5$")
    ax_ratio.set_xlabel("chain length $n$")
    ax_ratio.set_ylabel(r"$n / \tau_{\mathrm{int}}(n)$")
    ax_ratio.set_title("Sokal self-consistency ratio")
    ax_ratio.grid(alpha=0.3, which="both")
    ax_ratio.legend(fontsize=9, loc="best")

    fig.tight_layout()
    tag = args.out_tag or f"L{args.L}_N{args.mh_steps}"
    out_dir = HERE.parents[1] / "results"
    out_pdf = out_dir / f"dm_mc_tau_convergence_3d_{tag}.pdf"
    out_npz = out_dir / f"dm_mc_tau_convergence_3d_{tag}.npz"
    if not args.no_plot:
        fig.savefig(out_pdf)
        print(f"\nSaved figure to {out_pdf}")
    plt.close(fig)

    save = dict(
        L=args.L, k=args.k, l=args.l, c=args.c, t_mh=args.t_mh,
        num_chains=args.num_chains, mh_steps=args.mh_steps,
        n_grid=n_grid, epochs=np.array(epochs),
    )
    for ep in epochs:
        r = results[ep]
        save[f"ep{ep}_accept"] = r["accept"]
        save[f"ep{ep}_taus"] = r["taus"]
        save[f"ep{ep}_Ws"] = r["Ws"]
        save[f"ep{ep}_phi_bar"] = r["phi_bar"]
    np.savez(out_npz, **save)
    print(f"Saved data to {out_npz}")


if __name__ == "__main__":
    main()
