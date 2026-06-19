"""
Post-hoc EMA reconstruction (Karras 2024 EDM2 style, simplified).

Given a set of training snapshots saved at log-scale epochs, reconstruct
an EMA with arbitrary effective decay by weighted averaging.

Two weighting schemes are supported:
  * "power": w_i propto (epoch_i / T_final)^gamma. Power-EMA from Karras.
  * "exp":   w_i propto exp((epoch_i - T_final) / tau). Standard EMA with
             time-constant tau (in epoch units).

The script can either (a) write the reconstructed EMA into a new ckpt's
`ema_state_dict` (for later use with sample_phi4_edm2.py), or (b) sweep
multiple gammas/taus and report which gives best validation quality.

Usage (recommended workflow):
  # Build reconstructed EMA at several gammas in a new run-dir/post_hoc_ema/
  python posthoc_ema.py build \\
      --run_dir runs/phi4_Lmulti8-16-32-64_k0.2705_l0.022_edm2_sd0.281 \\
      --gammas 1 3 7 16 32

  # Sample from each post-hoc ckpt and let user pick best.
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


def list_ckpts(model_dir: Path):
    """Return sorted list of (epoch_int, path) for all epoch=N.ckpt files."""
    out = []
    for p in model_dir.glob("epoch=*.ckpt"):
        m = re.match(r"epoch=(\d+)\.ckpt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def compute_weights(epochs, gamma=None, tau=None):
    """Build normalized weights for each snapshot.

    Args:
        epochs: sorted list of training epoch indices
        gamma: power-EMA parameter (>0; larger = sharper EMA)
        tau: exponential EMA time-constant (epochs; larger = wider window)

    Exactly one of (gamma, tau) must be specified.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    T = epochs[-1]
    if gamma is not None:
        # Power EMA: w_i ∝ (epoch_i / T)^gamma
        w = (epochs / T) ** gamma
    elif tau is not None:
        w = np.exp((epochs - T) / tau)
    else:
        raise ValueError("specify gamma OR tau")
    w = w / w.sum()
    return w


def weighted_average_state_dicts(ckpt_paths, weights, key_prefix="state_dict",
                                 filter_prefix="precond.net"):
    """Read each ckpt's `key_prefix` section and compute a weighted average.

    Only averages tensor parameters; non-tensors (e.g. metadata) are copied
    from the last ckpt.

    Args:
        ckpt_paths: list of Path objects
        weights: same length as ckpt_paths, sums to 1
        key_prefix: which section of the ckpt to access (default 'state_dict')
        filter_prefix: only average keys starting with this prefix; non-matching
            keys are copied from the LAST ckpt unchanged.

    Returns the averaged state_dict (dict[str -> Tensor]).
    """
    assert abs(weights.sum() - 1.0) < 1e-6, f"weights sum = {weights.sum()}"
    weights = torch.as_tensor(weights, dtype=torch.float64)
    avg = None
    for i, (p, w) in enumerate(zip(ckpt_paths, weights)):
        c = torch.load(p, map_location="cpu", weights_only=False)
        sd = c[key_prefix]
        if avg is None:
            # Initialize with structure from last ckpt for non-tensor fields,
            # but zero out the tensors we will accumulate.
            avg = {}
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and (filter_prefix in k):
                    avg[k] = torch.zeros_like(v, dtype=torch.float64)
                else:
                    avg[k] = v   # copy from first ckpt; will overwrite below
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and (filter_prefix in k):
                avg[k].add_(v.to(torch.float64) * w.item())
        # Always overwrite non-tensor / non-matching with the LATEST snapshot's
        if i == len(ckpt_paths) - 1:
            for k, v in sd.items():
                if not (isinstance(v, torch.Tensor) and (filter_prefix in k)):
                    avg[k] = v
    # Cast back to fp32
    for k in list(avg.keys()):
        if isinstance(avg[k], torch.Tensor) and avg[k].dtype == torch.float64:
            avg[k] = avg[k].to(torch.float32)
    return avg


def build_posthoc_ema_ckpt(template_ckpt_path: Path, new_state_dict: dict,
                           out_path: Path):
    """Take an existing ckpt as template, replace its state_dict (and clear
    ema_state_dict so sample script uses these weights directly via the
    `ema.average_parameters()` context), save as new ckpt.

    To make the new weights take effect during sampling, we copy them into the
    'shadow_params' of ema_state_dict — that's what `ema.average_parameters()`
    swaps in at sample time.
    """
    c = torch.load(template_ckpt_path, map_location="cpu", weights_only=False)
    # Replace state_dict with averaged weights
    c["state_dict"] = new_state_dict

    # Also update ema_state_dict's shadow_params so sampling picks them up.
    # NOTE: shadow_params only tracks trainable params (excludes buffers like
    # MPGaussianFourierProjection's freqs/phases). We exclude '.freqs' and
    # '.phases' keys to match.
    if "ema_state_dict" in c and "shadow_params" in c["ema_state_dict"]:
        sp = c["ema_state_dict"]["shadow_params"]
        net_keys = [k for k in new_state_dict.keys()
                    if ("precond.net" in k)
                    and (not k.endswith(".freqs"))
                    and (not k.endswith(".phases"))]
        assert len(sp) == len(net_keys), (
            f"shadow_params has {len(sp)} entries but trainable-param keys "
            f"in state_dict count {len(net_keys)} — order mismatch?")
        for i, k in enumerate(net_keys):
            sp[i] = new_state_dict[k].clone()
        c["ema_state_dict"]["shadow_params"] = sp

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(c, out_path)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Reconstruct post-hoc EMA ckpts")
    p_build.add_argument("--run_dir", type=str, required=True)
    p_build.add_argument("--gammas", type=float, nargs="+",
                         default=[1, 3, 7, 16, 32, 64],
                         help="Power-EMA gammas to build")
    p_build.add_argument("--taus", type=float, nargs="*", default=[],
                         help="Exp-EMA time-constants (epochs) to also build")
    p_build.add_argument("--out_subdir", type=str, default="post_hoc_ema")

    p_inspect = sub.add_parser("inspect", help="Show ckpts and weights")
    p_inspect.add_argument("--run_dir", type=str, required=True)
    p_inspect.add_argument("--gamma", type=float, default=7)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "models"
    ckpts = list_ckpts(model_dir)
    if not ckpts:
        sys.exit(f"No ckpts in {model_dir}")
    epochs = [e for e, _ in ckpts]
    paths = [p for _, p in ckpts]
    print(f"Found {len(ckpts)} ckpts: epochs {epochs[0]}..{epochs[-1]}")

    if args.cmd == "inspect":
        w = compute_weights(epochs, gamma=args.gamma)
        print(f"\nPower-EMA gamma={args.gamma} weights:")
        for e, wi in zip(epochs, w):
            bar = "█" * int(wi * 200)
            print(f"  epoch={e:>5}  w={wi:.4f}  {bar}")
        return

    # build
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    template = paths[-1]

    for gamma in args.gammas:
        w = compute_weights(epochs, gamma=gamma)
        print(f"\nBuilding gamma={gamma}: weight on last ckpt = {w[-1]:.4f}, "
              f"sum-of-weights on last 5 = {w[-5:].sum():.4f}")
        avg = weighted_average_state_dicts(paths, w)
        out_path = out_dir / f"posthoc_gamma{gamma:g}.ckpt"
        build_posthoc_ema_ckpt(template, avg, out_path)
        print(f"  saved {out_path}")

    for tau in args.taus:
        w = compute_weights(epochs, tau=tau)
        print(f"\nBuilding tau={tau} epochs: weight on last ckpt = {w[-1]:.4f}, "
              f"sum-of-weights on last 5 = {w[-5:].sum():.4f}")
        avg = weighted_average_state_dicts(paths, w)
        out_path = out_dir / f"posthoc_tau{tau:g}.ckpt"
        build_posthoc_ema_ckpt(template, avg, out_path)
        print(f"  saved {out_path}")

    print(f"\nDone. Use these ckpts with:")
    print(f"  python sample_phi4_edm2.py --L 32 ... \\")
    print(f"    --checkpoint {out_dir}/posthoc_gamma7.ckpt")


if __name__ == "__main__":
    main()
