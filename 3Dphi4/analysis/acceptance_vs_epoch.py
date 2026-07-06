"""
Sweep MALA acceptance rate vs training epoch for one 3D phi^4 run.
Mirror of 2Dphi4/acceptance_rate/acceptance_vs_epoch.py.

Usage:
    python analysis/acceptance_vs_epoch.py --device cuda:0 --L 64 --k 0.2 \
        --run_dir runs/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760 \
        --c 0.2 --t_mh 1e-4 --num_samples 256 --batch_size 512 --ema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import re
import io
import functools
import argparse
from pathlib import Path

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from phi4_action import phi4_action as _phi4_action_nd


def find_epoch_checkpoints(model_dir):
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=*.ckpt")):
        if '-v' in f.stem:
            continue
        m = re.search(r'epoch=(\d+)$', f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def build_run_context(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19

    path_str = str(ckpt_path)
    k = float(re.search(r'_k([\d.]+)', path_str).group(1))
    l = float(re.search(r'_l([\d.]+)', path_str).group(1))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp3D(marginal_prob_std_fn)

    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel3D.load_from_checkpoint(
        buf, score_model=score_model, map_location=device, weights_only=False
    )
    model = model.to(device).eval()
    if hasattr(model, 'ema'):
        model.ema.to(device)
    action_fn = functools.partial(
        _phi4_action_nd, k=k, l=l, phi_min=norm_min, phi_max=norm_max,
        spatial_dims=3,
    )
    return model, action_fn, norm_min, norm_max, sigma


def swap_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", {})
    model_keys = set(model.state_dict().keys())
    ckpt_has_orig = any('._orig_mod.' in k for k in sd)
    model_has_orig = any('._orig_mod.' in k for k in model_keys)
    if ckpt_has_orig and not model_has_orig:
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    elif not ckpt_has_orig and model_has_orig:
        sd = {k.replace('score_model.', 'score_model._orig_mod.', 1): v
              for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    if 'ema_state_dict' in ckpt and hasattr(model, 'ema'):
        model.ema.load_state_dict(ckpt['ema_state_dict'])
    del ckpt


def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    return x


def one_acceptance(model, action_fn, h, t_mh, x_ref, mh_steps, batch_size):
    device = model.device
    num_samples = x_ref.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x_ref.dtype)
    x = x_ref.clone()
    total_accept = torch.zeros(num_samples, device=device)
    with torch.no_grad():
        for _ in range(mh_steps):
            score_x = torch.cat([
                model(x[i:i+batch_size],
                      torch.ones(min(batch_size, num_samples - i), device=device) * t_mh)
                for i in range(0, num_samples, batch_size)
            ])
            noise = torch.randn_like(x)
            y = x + h_t * score_x + torch.sqrt(2 * h_t) * noise
            score_y = torch.cat([
                model(y[i:i+batch_size],
                      torch.ones(min(batch_size, num_samples - i), device=device) * t_mh)
                for i in range(0, num_samples, batch_size)
            ])
            drift = h_t * (score_x + score_y)
            log_q = -(0.25 / h_t) * torch.sum(
                (torch.sqrt(2 * h_t) * noise + drift) ** 2, dim=(1, 2, 3, 4)
            ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3, 4))
            log_pi = action_fn(x) - action_fn(y)
            accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
            accept = torch.rand(num_samples, device=device) < accept_prob
            x = torch.where(accept[:, None, None, None, None], y, x)
            total_accept += accept.float()
    acc = total_accept / mh_steps
    return acc.mean().item(), (acc.std() / acc.shape[0] ** 0.5).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch_window", type=str, default="0-1000000")
    parser.add_argument("--max_ckpts", type=int, default=0,
                        help="Cap on number of ckpts; <=0 uses all in window.")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--mh_steps", type=int, default=1)
    parser.add_argument("--c", type=float, default=0.2,
                        help="Step size coefficient h = c / L^3")
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA-averaged weights; output filename gets "
                             "'_ema' suffix.")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_tag", type=str, default=None)
    args = parser.parse_args()

    def _parse_window(w):
        a, b = w.split("-")
        return int(a), int(b)
    ep_lo, ep_hi = _parse_window(args.epoch_window)

    if args.run_dir:
        model_dir = Path(args.run_dir)
    else:
        model_dir = Path(f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_{args.network}")
    ckpts = find_epoch_checkpoints(model_dir / "models")
    all_epochs = sorted(ckpts.keys())
    if not all_epochs:
        raise RuntimeError(f"No ckpts found in {model_dir / 'models'}")

    in_range = [e for e in all_epochs if ep_lo <= e <= ep_hi]
    if args.max_ckpts > 0 and len(in_range) > args.max_ckpts:
        idx = np.unique(np.round(np.linspace(0, len(in_range) - 1,
                                             args.max_ckpts)).astype(int))
        in_range = [in_range[i] for i in idx]
    if not in_range:
        raise RuntimeError(
            f"No ckpts in epoch window {args.epoch_window} "
            f"(have {len(all_epochs)} spanning {all_epochs[0]}-{all_epochs[-1]})."
        )
    print(f"Run dir: {model_dir}")
    print(f"Sweeping {len(in_range)} epochs: {in_range}")

    print(f"Building model from {ckpts[in_range[0]]}")
    model, action_fn, norm_min, norm_max, sigma_ve = build_run_context(
        ckpts[in_range[0]], args.device
    )
    print(f"norm range: [{norm_min}, {norm_max}],  VE sigma = {sigma_ve}")

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"
    print(f"Reference data: {args.data_path}")
    np.random.seed(42)
    x_ref = load_reference_data(
        args.data_path, args.num_samples, norm_min, norm_max, args.device
    )
    print(f"Loaded {x_ref.shape[0]} reference configs, shape {x_ref.shape}")

    L = args.L
    h = args.c / (L ** 3)
    print(f"\nc = {args.c},  h = {h:.2e},  t_mh = {args.t_mh:.2e}\n")

    accs = np.zeros(len(in_range))
    sems = np.zeros(len(in_range))
    for i, ep in enumerate(in_range):
        if ckpts[ep] != ckpts[in_range[0]]:
            swap_weights(model, ckpts[ep])
        if args.ema:
            model.ema.copy_to()
        acc, sem = one_acceptance(model, action_fn, h, args.t_mh, x_ref,
                                  args.mh_steps, args.batch_size)
        accs[i] = acc
        sems[i] = sem
        print(f"  epoch {ep:>6d}  ->  accept = {acc:.4f} +- {sem:.4f}")

    # --- plot ---
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep_arr = np.array(in_range)
    ax.errorbar(ep_arr, accs, yerr=sems, marker='o', capsize=3, color='C0',
                label=rf'$\kappa={args.k}$')
    ax.set_xscale('log')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Acceptance rate')
    ax.set_title(rf'3D: $L={L},\;\kappa={args.k},\;c={args.c},\;t_{{\mathrm{{mh}}}}={args.t_mh:.0e}$')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()

    out_dir = Path(args.out_dir) if args.out_dir else Path(SCRIPT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    if args.ema:
        tag += "_ema"
    out_pdf = out_dir / f"acceptance_vs_epoch_3d_L{L}_k{args.k}{tag}.pdf"
    fig.savefig(out_pdf)
    np.savez(str(out_pdf).replace(".pdf", ".npz"),
             epochs=ep_arr, acc=accs, sem=sems,
             c=args.c, h=h, t_mh=args.t_mh, sigma_ve=sigma_ve,
             k=args.k, L=L, run_dir=str(model_dir))
    print(f"\nSaved plot to {out_pdf}")
    print(f"Saved data to {str(out_pdf).replace('.pdf', '.npz')}")
    plt.close()


if __name__ == "__main__":
    main()
