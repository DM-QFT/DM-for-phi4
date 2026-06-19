"""
Generate samples from a trained EDM diffusion model on 2D phi4.

Usage:
    python sample_phi4_edm.py --L 32 --k 0.28 --sigma_data 0.30 \
        --ep 5000 --num_samples 1024 --device cuda:3
"""

import sys
sys.path.append("..")

import os
import re
import argparse
from pathlib import Path

import h5py
import torch
import numpy as np

from networks import NCSNpp2D
from edm_lightning import EDMDiffusionModel, trivial_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .ckpt; if not set, picks newest matching --ep")
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma_data", type=float, default=None,
                   help="Used to locate the run directory; will be overridden "
                        "by value stored in ckpt hparams")
    p.add_argument("--ep", type=str, default=None,
                   help="Substring to match the checkpoint epoch (e.g. '5000')")
    p.add_argument("--num_samples", type=int, default=1024,
                   help="Samples per Heun pass")
    p.add_argument("--n_repeats", type=int, default=1,
                   help="Number of independent passes to concatenate")
    p.add_argument("--num_steps", type=int, default=18,
                   help="Number of Heun steps (Karras default = 18)")
    p.add_argument("--rho", type=float, default=7.0,
                   help="sigma schedule curvature (Karras default = 7)")
    p.add_argument("--S_churn", type=float, default=0.0,
                   help="Stochasticity (0 = deterministic ODE)")
    p.add_argument("--sigma_min", type=float, default=None,
                   help="Override sigma_min from ckpt (Karras default = 0.002)")
    p.add_argument("--sigma_max", type=float, default=None,
                   help="Override sigma_max from ckpt (Karras default = 80)")
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_suffix", type=str, default="")
    args = p.parse_args()

    if args.checkpoint is None:
        # Locate run dir; user can override --sigma_data to disambiguate
        sd_str = f"_sd{args.sigma_data}" if args.sigma_data is not None else ""
        run_dir = f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_edm{sd_str}{args.output_suffix}"
        models_dir = Path(f"{run_dir}/models")
        if not models_dir.exists():
            # Fallback: glob for any matching directory
            cand = sorted(Path("runs").glob(f"phi4_L{args.L}_k{args.k}_l{args.l}_edm*"))
            if not cand:
                raise FileNotFoundError(f"No EDM run dir under runs/ matching "
                                        f"L={args.L} k={args.k}")
            run_dir = str(cand[-1])
            models_dir = Path(f"{run_dir}/models")
            print(f"Auto-located run dir: {run_dir}")
        pattern = f"*{args.ep}*.ckpt" if args.ep else "*.ckpt"
        ckpts = sorted(models_dir.glob(pattern))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {models_dir} matching {pattern}")
        args.checkpoint = str(ckpts[-1])
    else:
        run_dir = str(Path(args.checkpoint).parent.parent)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Run dir:    {run_dir}")

    # Load hparams from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    sigma_data = hp.get("sigma_data", args.sigma_data)
    # If user explicitly passes --L different from ckpt's L, allow cross-L
    # generation (NCSNpp2D is fully-convolutional + periodic so weights work
    # at any spatial size). Otherwise default to ckpt-stored L.
    ckpt_L = hp.get("L", args.L)
    L = args.L if args.L != ckpt_L else ckpt_L
    if L != ckpt_L:
        print(f"[CROSS-L] ckpt was trained at L={ckpt_L}, sampling at L={L}")
    P_mean = hp.get("P_mean", -1.2)
    P_std  = hp.get("P_std", 1.2)
    sigma_min = args.sigma_min if args.sigma_min is not None else hp.get("sigma_min", 0.002)
    sigma_max = args.sigma_max if args.sigma_max is not None else hp.get("sigma_max", 80.0)
    norm_min = hp.get("norm_min")
    norm_max = hp.get("norm_max")
    print(f"L={L}, sigma_data={sigma_data}, sigma_min={sigma_min}, sigma_max={sigma_max}")
    print(f"norm: [{norm_min}, {norm_max}]")

    # Build model (must match training config; trivial_std bypass)
    score_network = NCSNpp2D(trivial_std, channels=[16, 32, 64, 128])
    model = EDMDiffusionModel.load_from_checkpoint(
        args.checkpoint,
        score_network=score_network,
        sigma_data=sigma_data,
        L=L,
        P_mean=P_mean, P_std=P_std,
        sigma_min=sigma_min, sigma_max=sigma_max,
        norm_min=norm_min, norm_max=norm_max,
        map_location=args.device, weights_only=False,
    )
    model = model.to(args.device).eval()

    # Sample
    print(f"Sampling: num_steps={args.num_steps}, rho={args.rho}, "
          f"S_churn={args.S_churn}, n_repeats={args.n_repeats}")
    reps = []
    for i in range(args.n_repeats):
        if args.seed is not None:
            s = args.seed + i
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        x = model.sample_heun(
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            rho=args.rho,
            S_churn=args.S_churn,
        )
        reps.append(x)
    samples_norm = torch.cat(reps, dim=0)
    print(f"Generated {samples_norm.shape[0]} samples, shape={tuple(samples_norm.shape)}")

    # Denormalize back to physical scale
    samples_phys = model.denormalize(samples_norm).detach().cpu().numpy()

    # Save
    out_dir = Path(run_dir) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_tag = re.search(r"epoch=(\d+)", args.checkpoint)
    ep_tag = ep_tag.group(1) if ep_tag else "unk"
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    churn_tag = f"_Sc{args.S_churn:g}" if args.S_churn > 0 else ""
    smin_tag = f"_smin{sigma_min:g}" if args.sigma_min is not None else ""
    # Cross-L tag: include L only when generating at a size different from ckpt's L
    cross_L_tag = f"_L{L}" if L != ckpt_L else ""
    out_path = out_dir / (f"samples_edm_ep{ep_tag}_N{samples_phys.shape[0]}"
                          f"_steps{args.num_steps}{churn_tag}{smin_tag}{cross_L_tag}{seed_tag}.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("cfgs", data=samples_phys[:, 0])  # (N, L, L)
        f.attrs["L"] = int(L)
        f.attrs["k"] = float(args.k)
        f.attrs["lam"] = float(args.l)
        f.attrs["sigma_data"] = float(sigma_data)
        f.attrs["num_steps"] = int(args.num_steps)
        f.attrs["rho"] = float(args.rho)
        f.attrs["S_churn"] = float(args.S_churn)
        f.attrs["checkpoint"] = str(args.checkpoint)
    print(f"Saved samples to: {out_path}")

    # Quick diagnostics
    print()
    print("=== quick diagnostics ===")
    phi_mean = samples_phys.mean()
    phi_abs_mag = np.abs(samples_phys.mean(axis=(1, 2, 3))).mean()
    phi2 = (samples_phys ** 2).mean()
    print(f"  <phi>            = {phi_mean:+.4f}")
    print(f"  <|magnetisation|> = {phi_abs_mag:.4f}    (per-config |mean phi|)")
    print(f"  <phi^2>          = {phi2:.4f}")


if __name__ == "__main__":
    main()
