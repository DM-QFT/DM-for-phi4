"""
MALA acceptance-rate diagnostic for 3D phi^4 score models (Sec. IV B of DM.tex).

Given a trained checkpoint and a target lattice size L, measure the single-step
Metropolis-adjusted Langevin acceptance rate of a proposal that uses the score
network as drift, evaluated on HMC reference configurations:

    h = c / L^3,   phi' = phi + h s_theta(phi, t_mh) + sqrt(2h) eta,
    accept with probability  min(1, exp[S(phi) - S(phi') + log(q-ratio)]).

Setting model.L = L makes the same code cover both in-distribution evaluation
(L is the training size) and cross-L evaluation (a multi-L model at an unseen L,
e.g. the {4,8,16,32} model at L=64). A single MH step keeps the trials
independent. Use EMA weights (--ema) and c=0.2, t_mh=1e-4 as in the paper.

Run from 3Dphi4/. The three 3D acceptance results in DM.tex are reproduced by
looping this script over the relevant runs, e.g.:

    # tab:acc_early_late  (single-L; loop L over 4 8 16 32 64 externally)
    python "analysis/acceptance rate/acceptance.py" --run runs/phi4_3d_L16_k0.1923_l0.9_ncsnpp_sigma280 \
        --L 16 --k 0.1923 --epochs 50 9111 --ema

    # tab:acc_multiL_3d   (cross-L at unseen L=64, SEM over 5 seeds)
    python "analysis/acceptance rate/acceptance.py" --run runs/phi4_3d_Lmulti4-8-16-32_k0.2_l0.9_ncsnpp_sigma400 \
        --L 64 --k 0.2 --epochs 50 10000 --ema --n_seeds 5

    # fig:acc_vs_epoch_3d (all checkpoints vs epoch -> CSV for plotting)
    python "analysis/acceptance rate/acceptance.py" --run runs/phi4_3d_L64_k0.1923_l0.9_ncsnpp_sigma2048 \
        --L 64 --k 0.1923 --ema --out_csv acc_vs_epoch_L64_k0.1923.csv
"""

import sys
import os
import re
import io
import csv
import functools
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]                   # 3Dphi4/
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ.parent))     # repo root
os.chdir(PROJ)

import h5py
import torch
import numpy as np

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from sample_phi4 import phi4_action_3d


def load_model(ckpt_path, L, k, l, device, ema):
    """Load a 3D score model; set model.L=L (covers cross-L); optionally apply EMA."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    sigma = hp.get("sigma", 150.0)
    norm_min = hp.get("norm_min") or -6.22
    norm_max = hp.get("norm_max") or 6.19

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):        # strip torch.compile prefix
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)
    score_model = NCSNpp3D(functools.partial(marginal_prob_std, sigma=sigma))
    model = DiffusionModel3D.load_from_checkpoint(buf, score_model=score_model)
    model.L = L
    model = model.to(device).eval()
    if ema:
        model.ema.to(device)
        model.ema.copy_to()
    action_fn = functools.partial(phi4_action_3d, k=k, l=l, phi_min=norm_min, phi_max=norm_max)
    return model, action_fn, norm_min, norm_max, sigma


def load_reference(k, l, L, n, norm_min, norm_max, device, seed):
    """Draw n HMC cfgs at (k,l,L), map to [-1,1], return (n,1,L,L,L) on device."""
    path = Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l={l}_{L}^3.jld2")
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.default_rng(seed).choice(cfgs.shape[0], size=min(n, cfgs.shape[0]), replace=False)
    x = ((cfgs[idx] - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return x.unsqueeze(1) if x.ndim == 4 else x


def acceptance_rate(model, action_fn, x, h, t_mh, batch_size):
    """One MALA step from each config in x. Return (mean acceptance, binomial SEM)."""
    dev = model.device
    h_t = torch.tensor(h, device=dev, dtype=x.dtype)

    def score(phi):
        return torch.cat([
            model(phi[i:i + batch_size],
                  torch.full((phi[i:i + batch_size].shape[0],), t_mh, device=dev, dtype=phi.dtype))
            for i in range(0, phi.shape[0], batch_size)
        ])

    with torch.no_grad():
        s_x = score(x)
        eta = torch.randn_like(x)
        y = x + h_t * s_x + torch.sqrt(2 * h_t) * eta
        s_y = score(y)
        drift = h_t * (s_x + s_y)
        log_q = -(0.25 / h_t) * ((torch.sqrt(2 * h_t) * eta + drift) ** 2).sum(dim=(1, 2, 3, 4)) \
                + 0.5 * (eta ** 2).sum(dim=(1, 2, 3, 4))
        alpha = torch.exp(action_fn(x) - action_fn(y) + log_q).clamp(max=1.0)
        accept = (torch.rand(x.shape[0], device=dev) < alpha).float()
    N = accept.shape[0]
    return accept.mean().item(), (accept.std() / N ** 0.5).item()


def select_checkpoints(args):
    """{epoch: path} from --ckpt (one) or --run (+ optional --epochs nearest match).

    Handles both single-L ('epoch=epoch=N.ckpt') and multi-L ('epoch=N.ckpt')
    naming via the trailing-number regex.
    """
    def epoch_of(p):
        m = re.search(r'epoch=(\d+)$', Path(p).stem)
        return int(m.group(1)) if m else -1
    if args.ckpt:
        return {epoch_of(args.ckpt): Path(args.ckpt)}
    all_ck = {epoch_of(f): f
              for f in sorted((Path(args.run) / "models").glob("epoch=*.ckpt"))
              if '-v' not in f.stem}
    all_ck = dict(sorted(all_ck.items()))
    if args.epochs:
        picked = {min(all_ck, key=lambda e: abs(e - t)) for t in args.epochs}
        return {e: all_ck[e] for e in sorted(picked)}
    return all_ck


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory; uses checkpoints under <run>/models")
    src.add_argument("--ckpt", help="a single checkpoint file")
    ap.add_argument("--L", type=int, required=True, help="lattice size to evaluate (sets model.L)")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--l", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, nargs="+", default=None,
                    help="with --run: epochs to evaluate (nearest available); default all")
    ap.add_argument("--c", type=float, default=0.2, help="step size h = c / L^3")
    ap.add_argument("--t_mh", type=float, default=1e-4)
    ap.add_argument("--num_samples", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=None, help="default scales down with L")
    ap.add_argument("--ema", action="store_true", help="use EMA-averaged weights (as in DM.tex)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_seeds", type=int, default=1, help=">1 reports the SEM across seeds")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    h = args.c / args.L ** 3
    bs = args.batch_size or (16 if args.L >= 64 else 64 if args.L >= 32 else 128)
    ckpts = select_checkpoints(args)
    seeds = range(args.seed, args.seed + max(1, args.n_seeds))
    print(f"L={args.L} k={args.k} l={args.l} | h={h:.2e} t_mh={args.t_mh:.0e} "
          f"N={args.num_samples} ema={args.ema} bs={bs} n_seeds={args.n_seeds}")

    rows = []
    for ep in sorted(ckpts):
        model, action_fn, nmin, nmax, sigma = load_model(
            ckpts[ep], args.L, args.k, args.l, args.device, args.ema)
        per_seed, bin_sem = [], float("nan")
        for s in seeds:
            torch.manual_seed(s)
            x = load_reference(args.k, args.l, args.L, args.num_samples, nmin, nmax, args.device, s)
            mean, bin_sem = acceptance_rate(model, action_fn, x, h, args.t_mh, bs)
            per_seed.append(mean)
        per_seed = np.array(per_seed)
        if len(per_seed) == 1:
            acc, sem = per_seed[0], bin_sem
        else:
            acc, sem = per_seed.mean(), per_seed.std(ddof=1) / len(per_seed) ** 0.5
        rows.append((ep, sigma, acc, sem))
        print(f"  epoch={ep:>6}  sigma={sigma:>5.0f}  accept = {acc:.4f} ({sem:.4f})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# L", args.L, "k", args.k, "c", args.c, "t_mh", args.t_mh,
                        "N", args.num_samples, "ema", args.ema, "n_seeds", args.n_seeds])
            w.writerow(["epoch", "sigma", "acc_mean", "acc_sem"])
            for ep, sigma, acc, sem in rows:
                w.writerow([ep, f"{sigma:.0f}", f"{acc:.6f}", f"{sem:.6f}"])
        print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
