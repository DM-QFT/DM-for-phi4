"""β_conf(t) forward (HMC+noise, exact marginal) vs reverse (DM) for the
multi-L run phi4_Lmulti8-16-32-64_k0.28_l0.022_ncsnpp, one curve pair per L.

Run from repo root:
    python betaconfig/beta_conf_fwd_rev_multiL.py --device cuda
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]           # betaconfig/analysis/ -> repo root
sys.path.append(str(ROOT))
sys.path.append(str(HERE))

from diffusion_lightning import DiffusionModel, marginal_prob_std
from networks import NCSNpp2D
from beta_conf_along_reverse import (
    load_hmc_2d, reverse_sde_with_beta, forward_noised_hmc_curve,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ls", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--run_dir", type=str,
                   default="2Dphi4/runs/phi4_Lmulti8-16-32-64_k0.28_l0.022_ncsnpp")
    p.add_argument("--ep", type=int, default=10000)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="log")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--base_samples", type=int, default=512,
                   help="reverse-SDE batch at L=64; scaled by (64/L)^2 for equal dof")
    p.add_argument("--record_every", type=int, default=4)
    p.add_argument("--hmc_n", type=int, default=10240)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first_term_only", action="store_true")
    args = p.parse_args()

    ckpt_path = ROOT / args.run_dir / "models" / f"epoch={args.ep}.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    sigma_model, norm_min, norm_max = hp["sigma"], hp["norm_min"], hp["norm_max"]
    print(f"ckpt: {ckpt_path.name}  sigma={sigma_model}  "
          f"norm=[{norm_min:.4f}, {norm_max:.4f}]")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma_model)
    score_model = NCSNpp2D(marginal_prob_std_fn)
    model = DiffusionModel.load_from_checkpoint(
        ckpt_path, score_model=score_model, map_location=args.device,
        weights_only=False)
    model = model.to(args.device).eval()

    out_dir = HERE.parent / "results" / "beta_conf_traj_multiL"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_common = (f"k{args.k}_l{args.l}_ep{args.ep}_steps{args.num_steps}"
                  f"_{args.schedule}" + ("_term1" if args.first_term_only else ""))

    results = {}
    for L in args.Ls:
        n_samp = max(1, round(args.base_samples * (64 / L) ** 2))
        torch.manual_seed(args.seed + L)
        torch.cuda.manual_seed_all(args.seed + L)
        model.L = L
        print(f"\n=== L={L}: reverse SDE, batch={n_samp} ===")
        dm = reverse_sde_with_beta(
            model, n_samp, args.num_steps, args.schedule, args.eps,
            args.k, args.l, norm_min, norm_max,
            record_every=args.record_every,
            first_term_only=args.first_term_only)
        print(f"  final β̂ = {dm['mean'][-1]:.4f} ± {dm['sem'][-1]:.4f}")

        hmc_path = (ROOT / "2Dphi4" / "trainingdata" /
                    f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{L}^2.jld2")
        hmc = load_hmc_2d(hmc_path, args.hmc_n)
        hmc_norm = 2.0 * (hmc - norm_min) / (norm_max - norm_min) - 1.0
        hmc_t = torch.from_numpy(hmc_norm).unsqueeze(1).float().to(args.device)
        fwd = forward_noised_hmc_curve(
            hmc_t, dm["t"], dm["sigma_t"], args.k, args.l, norm_min, norm_max,
            seed=args.seed + L + 1, spatial_dims=2,
            first_term_only=args.first_term_only)
        print(f"  fwd at smallest t: β̂ = {fwd['mean'][-1]:.4f} ± {fwd['sem'][-1]:.4f}")

        np.savez(out_dir / f"beta_conf_fwd_rev_L{L}_{tag_common}.npz",
                 t=dm["t"], sigma_t=dm["sigma_t"],
                 dm_mean=dm["mean"], dm_sem=dm["sem"],
                 hmc_mean=fwd["mean"], hmc_sem=fwd["sem"],
                 L=L, k=args.k, l=args.l, sigma_model=sigma_model,
                 norm_min=norm_min, norm_max=norm_max,
                 n_samp=n_samp, hmc_n=args.hmc_n)
        results[L] = (dm, fwd)

    # ---------- combined figure ----------
    colors = {L: c for L, c in zip(args.Ls, ["C0", "C1", "C2", "C3", "C4"])}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax = axes[0]
    ax.set_xscale("log")
    for L, (dm, fwd) in results.items():
        ax.plot(dm["t"], dm["mean"], "-", color=colors[L], lw=1.5,
                label=f"L={L} DM reverse")
        ax.plot(fwd["t"], fwd["mean"], "--", color=colors[L], lw=1.2,
                label=f"L={L} HMC+noise")
    ax.axhline(1.0, ls=":", color="k", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle\hat\beta(x_t)\rangle$")
    ax.set_title("full trajectory")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.set_xscale("log")
    for L, (dm, fwd) in results.items():
        m = dm["t"] < 1e-2
        ax.errorbar(dm["t"][m], dm["mean"][m], yerr=dm["sem"][m], fmt="o-",
                    ms=2, lw=1.0, color=colors[L], label=f"L={L} DM", alpha=0.9)
        mh = fwd["t"] < 1e-2
        ax.errorbar(fwd["t"][mh], fwd["mean"][mh], yerr=fwd["sem"][mh],
                    fmt="s--", ms=2, lw=0.8, color=colors[L], alpha=0.5)
    ax.axhline(1.0, ls=":", color="k", lw=0.8)
    ax.set_ylim(0.85, 1.1)
    ax.set_xlabel("t")
    ax.set_title("late stage (t < 1e-2); dashed = HMC+noise")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.set_xscale("log")
    for L, (dm, fwd) in results.items():
        diff = dm["mean"] - fwd["mean"]
        err = np.hypot(dm["sem"], fwd["sem"])
        ax.errorbar(dm["sigma_t"], diff, yerr=err, fmt="o-", ms=2, lw=1.0,
                    color=colors[L], label=f"L={L}", alpha=0.85)
    ax.axhline(0.0, ls=":", color="k", lw=0.8)
    ax.set_xlabel(r"$\sigma_t$")
    ax.set_ylabel(r"$\langle\hat\beta\rangle_{\rm DM} - \langle\hat\beta\rangle_{\rm HMC+noise}$")
    ax.set_title("reverse − forward (model error)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"multi-L  κ={args.k}  λ={args.l}  ep={args.ep}  "
                 f"σ={sigma_model}  {args.schedule} {args.num_steps} steps")
    fig.tight_layout()
    l_tag = "-".join(str(L) for L in args.Ls)
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"beta_conf_fwd_rev_L{l_tag}_{tag_common}.{ext}", dpi=160)
    print(f"\nsaved figure → {out_dir}/beta_conf_fwd_rev_L{l_tag}_{tag_common}.pdf")


if __name__ == "__main__":
    main()
