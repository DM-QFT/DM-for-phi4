"""β_conf(t) along reverse SDE — swept over training epochs.

Runs `analysis/beta_conf_along_reverse.py`'s reverse-SDE + β̂ machinery
on a geometric sweep of checkpoints, and produces

  (i) β̂(t→eps) − 1   vs   epoch    (asymptotic temperature convergence)
  (ii) full β̂(t) curves overlaid across a few representative epochs.

Forward-noised HMC curve (same across epochs) is recomputed on each epoch's
t grid for a per-epoch DM−HMC residual.

Run from `2Dphi4/`:
    python analysis/beta_conf_vs_epoch_2d.py \
        --L 64 --k 0.2705 --l 0.022 --output_suffix _sigma250 \
        --epochs 1 5 22 79 235 703 1795 4581 7318 10000 \
        --num_steps 2000 --schedule log --num_samples 256 \
        --record_every 20 --hmc_n 1024 --device cuda:3 --seed 0
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(HERE.parents[1]))

from diffusion_lightning import DiffusionModel, marginal_prob_std
from networks import NCSNpp2D
from phi4_action import phi4_beta_conf

from beta_conf_along_reverse import (
    load_hmc_2d, reverse_sde_with_beta, forward_noised_hmc_curve,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--k", type=float, default=0.2705)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--output_suffix", type=str, default="_sigma250")
    p.add_argument("--epochs", type=int, nargs="+",
                   default=[1, 5, 22, 79, 235, 703, 1795, 4581, 7318, 10000])
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="log")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--record_every", type=int, default=20)
    p.add_argument("--hmc_path", type=str, default=None)
    p.add_argument("--hmc_n", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.output_suffix}")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # ---- load HMC once ----
    if args.hmc_path is None:
        args.hmc_path = (f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}"
                         f"_{args.L}^2.jld2")
    hmc_path = Path(args.hmc_path)
    print(f"HMC: {hmc_path.name}  using {args.hmc_n} cfgs")

    # ---- read sigma_model + norm range from the latest ckpt (they're invariant) ----
    sample_ckpt = sorted(run_dir.glob(f"models/*{args.epochs[-1]}*.ckpt"))
    if not sample_ckpt:
        raise FileNotFoundError(f"no ckpt for ep={args.epochs[-1]}")
    ck = torch.load(sample_ckpt[-1], map_location="cpu", weights_only=False)
    hparams = ck.get("hyper_parameters", {})
    sigma_model = hparams.get("sigma", 250.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"sigma_model={sigma_model}  norm=[{norm_min:.4f}, {norm_max:.4f}]")
    del ck

    hmc_cfgs = load_hmc_2d(hmc_path, args.hmc_n)
    hmc_norm = 2.0 * (hmc_cfgs - norm_min) / (norm_max - norm_min) - 1.0
    hmc_norm_t = torch.from_numpy(hmc_norm).unsqueeze(1).float().to(args.device)

    # ---- sweep ----
    per_ep = {}
    final_summary = []           # rows: (ep, β̂_DM_final, sem_DM, β̂_HMC_final, sem_HMC)
    for ep in args.epochs:
        ck_paths = sorted(run_dir.glob(f"models/*{ep}*.ckpt"))
        if not ck_paths:
            print(f"[skip] ep={ep}: no checkpoint")
            continue
        ck_path = ck_paths[-1]

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        mp_std = functools.partial(marginal_prob_std, sigma=sigma_model)
        score_model = NCSNpp2D(mp_std)
        model = DiffusionModel.load_from_checkpoint(ck_path, score_model=score_model)
        model = model.to(args.device).eval()
        assert model.spatial_dims == 2

        dm = reverse_sde_with_beta(
            model, args.num_samples, args.num_steps, args.schedule, args.eps,
            args.k, args.l, norm_min, norm_max, record_every=args.record_every,
        )
        hmc_fwd = forward_noised_hmc_curve(
            hmc_norm_t, dm["t"], dm["sigma_t"],
            args.k, args.l, norm_min, norm_max, seed=args.seed + 1,
            spatial_dims=2,
        )

        per_ep[ep] = dict(dm=dm, hmc=hmc_fwd)
        final_summary.append((
            ep, dm["mean"][-1], dm["sem"][-1],
            hmc_fwd["mean"][-1], hmc_fwd["sem"][-1],
        ))
        print(f"ep={ep:>6d}   β̂_DM(t={dm['t'][-1]:.0e}) = "
              f"{dm['mean'][-1]:.4f} ± {dm['sem'][-1]:.4f}   "
              f"β̂_HMC = {hmc_fwd['mean'][-1]:.4f} ± {hmc_fwd['sem'][-1]:.4f}   "
              f"Δ = {dm['mean'][-1] - hmc_fwd['mean'][-1]:+.4f}")

        del model
        torch.cuda.empty_cache()

    # ---- save ----
    summary = np.array(final_summary)            # (n_ep, 5)
    out_dir = Path("results/beta_conf_traj")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"vs_epoch_L{args.L}_k{args.k}_l{args.l}_N{args.num_samples}"
    npz_path = out_dir / f"beta_conf_{tag}.npz"

    save_dict = dict(
        epochs=summary[:, 0].astype(int),
        dm_final=summary[:, 1], dm_sem=summary[:, 2],
        hmc_final=summary[:, 3], hmc_sem=summary[:, 4],
        sigma_model=sigma_model, norm_min=norm_min, norm_max=norm_max,
        k=args.k, l=args.l, L=args.L,
    )
    # also embed per-epoch full curves
    for ep, d in per_ep.items():
        save_dict[f"t_{ep}"]        = d["dm"]["t"]
        save_dict[f"sigma_t_{ep}"]  = d["dm"]["sigma_t"]
        save_dict[f"dm_mean_{ep}"]  = d["dm"]["mean"]
        save_dict[f"dm_sem_{ep}"]   = d["dm"]["sem"]
        save_dict[f"hmc_mean_{ep}"] = d["hmc"]["mean"]
        save_dict[f"hmc_sem_{ep}"]  = d["hmc"]["sem"]
    np.savez(npz_path, **save_dict)
    print(f"saved npz → {npz_path}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (a) final β̂ at t=eps vs epoch
    ax = axes[0]
    ax.set_xscale("log")
    eps_x = summary[:, 0].astype(float)
    ax.errorbar(eps_x, summary[:, 1], yerr=summary[:, 2],
                fmt="o-", color="C3", lw=1.4, capsize=2,
                label=r"DM  $\langle\hat\beta(t\!\to\!\epsilon)\rangle$")
    ax.errorbar(eps_x, summary[:, 3], yerr=summary[:, 4],
                fmt="s--", color="C0", lw=1.2, capsize=2,
                label="HMC + fwd noise")
    ax.axhline(1.0, ls=":", color="k", lw=0.8, label="target β=1")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\langle\hat\beta(x_{t=\epsilon})\rangle$")
    ax.set_title(f"L={args.L}  κ={args.k}  λ={args.l}   asymptotic β̂ vs epoch")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # (b) full β̂(t) curves overlaid for selected epochs
    ax = axes[1]
    ax.set_xscale("log")
    n_show = min(6, len(per_ep))
    eps_list = list(per_ep.keys())
    show_idx = np.linspace(0, len(eps_list) - 1, n_show).astype(int)
    cmap = plt.get_cmap("viridis")
    for j, idx in enumerate(show_idx):
        ep = eps_list[idx]
        d = per_ep[ep]["dm"]
        ax.plot(d["t"], d["mean"], "-", lw=1.2,
                color=cmap(j / max(n_show - 1, 1)), label=f"ep={ep}")
    # HMC reference from the last epoch (same t grid; ground-truth marginal)
    d_hmc = per_ep[eps_list[-1]]["hmc"]
    ax.plot(d_hmc["t"], d_hmc["mean"], "k--", lw=1.4, label="HMC + fwd noise")
    ax.axhline(1.0, ls=":", color="k", lw=0.6)
    ax.set_xlabel("t  (reverse time)")
    ax.set_ylabel(r"$\langle\hat\beta(x_t)\rangle$")
    ax.set_title("β̂(t) across training")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    pdf_path = out_dir / f"beta_conf_{tag}.pdf"
    fig.savefig(pdf_path)
    print(f"saved figure → {pdf_path}")


if __name__ == "__main__":
    main()
