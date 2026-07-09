"""
Generate samples from a trained 2D Yukawa scalar-field diffusion model.

Usage:
    python sample_yukawa.py --g 0.1 --ep 1000
    python sample_yukawa.py --g 0.1 --method ode --num_steps 200
"""

import sys
sys.path.append("../..")

import os
import re
import functools
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import ScoreNet, ScoreNetUNetPeriodic, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--method", type=str, default="em",
                        choices=["em", "ode", "pc"])
    parser.add_argument("--ep", type=str, default=None)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--g", type=float, default=0.1)
    parser.add_argument("--plot_grid", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["scorenet", "unet", "ncsnpp"],
                        help="Network architecture: scorenet | unet | ncsnpp")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix on the training output dir (e.g. '_sigma50')")
    parser.add_argument("--schedule", type=str, default="log",
                        choices=["log", "linear"],
                        help="Reverse-time step schedule")
    parser.add_argument("--ode_method", type=str, default="dpm2",
                        choices=["dpm1", "dpm2", "dpm3", "rk45"],
                        help="ODE solver when --method=ode (default dpm2)")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="Number of independent sampling passes to concatenate")
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, seeds torch/cuda RNG before each sampling pass. "
                             "Fixes both the initial noise x_T and all per-step noise "
                             "in SDE reverse integration, giving identical trajectories "
                             "across calls. Repeats use seed, seed+1, ... (deterministic).")
    args = parser.parse_args()

    run_dir = f"runs/yukawa_L{args.L}_g{args.g}_{args.network}{args.output_suffix}"

    # Get checkpoint
    if args.checkpoint is None:
        ckpts = sorted(Path(f"{run_dir}/models").glob(f"*{args.ep}*.ckpt"))
        args.checkpoint = str(ckpts[-1]) if ckpts else None
    print(f"Checkpoint: {args.checkpoint}")

    # Load hparams from checkpoint first
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 50.0)
    L = hparams.get("L", args.L)
    norm_min = hparams["norm_min"]
    norm_max = hparams["norm_max"]
    print(f"norm_min: {norm_min}, norm_max: {norm_max}")
    output = f"{run_dir}/data/"
    if not os.path.exists(output):
        os.makedirs(output)
    output = os.path.join(output, "samples")
    # Parse g from checkpoint path (directory or filename)
    g_match = re.search(r'_g([\d.]+?)_', str(args.checkpoint))
    g = float(g_match.group(1)) if g_match else args.g
    print(f"L={L}, g={g}, sigma={sigma}")

    # Load model with correct sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn)
    elif args.network == "scorenet":
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)
    model = DiffusionModel.load_from_checkpoint(args.checkpoint, score_model=score_model, map_location=args.device, weights_only=False)
    model = model.to(args.device).eval()

    # Sample
    print(f"Sampling ({args.method.upper()})  num_steps={args.num_steps}  n_repeats={args.n_repeats}  num_samples/rep={args.num_samples}")
    if args.seed is not None:
        print(f"Seed: {args.seed} (fixed IC + reverse-step noise)")

    def _seed_for(rep_idx):
        if args.seed is not None:
            s = args.seed + rep_idx
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)

    if args.method == "em":
        reps = []
        for i in range(args.n_repeats):
            _seed_for(i)
            reps.append(model.sample(args.num_samples, args.num_steps, schedule=args.schedule))
        samples = torch.concatenate(reps, axis=0)
    elif args.method == "ode":
        reps = []
        for i in range(args.n_repeats):
            _seed_for(i)
            reps.append(model.sample_ode(args.num_samples, args.num_steps,
                                         schedule=args.schedule, method=args.ode_method))
        samples = torch.concatenate(reps, axis=0)
    else:  # pc
        reps = []
        for i in range(args.n_repeats):
            _seed_for(i)
            reps.append(model.sample_pc(args.num_samples, args.num_steps))
        samples = torch.concatenate(reps, axis=0)

    # Samples are in normalized [-1, 1] range, shape (num_samples, 1, L, L)
    samples_norm = samples[:, 0].cpu().numpy()

    # Renormalize to original range
    samples_renorm = (samples_norm + 1) / 2 * (norm_max - norm_min) + norm_min

    # Save as (L, L, num_samples*n_repeats)
    samples_out = samples_renorm.transpose(1, 2, 0)
    tag = f"{args.method}_{args.schedule}_steps{args.num_steps}_{args.ep}"
    np.save(f"{output}_{tag}.npy", samples_out)
    print(f"Saved samples to {output}_{tag}.npy, shape: {samples_out.shape}")

    # Plot grid (only first plot_grid^2 samples)
    n = args.plot_grid
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, 0].cpu().numpy(), cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output}_{tag}.png", dpi=150)
    print(f"Saved plot to {output}_{tag}.png")


if __name__ == "__main__":
    main()
