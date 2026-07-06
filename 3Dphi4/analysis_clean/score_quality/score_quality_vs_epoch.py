"""
Direct score-quality metrics vs training epoch for one 3D phi^4 run.

Consolidated, cleaned version of 3Dphi4/score_quality_analysis/
score_quality_vs_epoch.py.  The numerics are byte-for-byte identical to that
script, so it reproduces

    3Dphi4/score_quality_analysis/dat/
        score_quality_vs_epoch_3d_L64_k0.1923_sigma2048_ema.dat

(85 epochs, 7 columns).  Only the path handling, docstrings and the
matplotlib backend were cleaned up; the model loading, EMA handling and the
metric definitions are unchanged.

For each saved checkpoint in [--epoch_window] the model is loaded by
swap_weights() (no network reconstruction after the first ckpt), the EMA
weights are copied in when --ema is set, and the following per-sample metrics
are computed in torch.no_grad() against the analytic phi^4 score -dS/dx_norm:

    cos_sim   = <s_model, -dS/dx_norm> / (|s_model| |dS|)
    rel_mse   = ||s_model - (-dS)||^2 / ||dS||^2
    mag_ratio = |s_model| / |dS|

At small t_eval the model score approximates -dS/dx exactly (VE-SDE limit);
the paper uses t_eval = 1e-4 (sigma(t_eval) ~ 0.01 in [-1,1] space).  Each
metric is reported as (mean, SEM = std / sqrt(N)) over the N reference cfgs.

Outputs a quick-look 3-panel PDF and an .npz; run export_score_quality_dat.py
on the .npz to emit the whitespace-delimited .dat for Mathematica / gnuplot.

Usage (run from 3Dphi4/; reproduces the k=0.1923 .dat above):
    python analysis_clean/score_quality/score_quality_vs_epoch.py \
        --device cuda:0 --L 64 --k 0.1923 --l 0.9 \
        --run_dir runs/phi4_3d_L64_k0.1923_l0.9_ncsnpp_sigma2048 \
        --num_samples 64 --batch_size 64 --t_eval 1e-4 --ema \
        --out_dir analysis/results --out_tag sigma2048
"""

import sys
import os
import re
import io
import functools
import argparse
from pathlib import Path

# analysis_clean/score_quality/ is two levels below 3Dphi4/.
ROOT3D = Path(__file__).resolve().parents[2]    # 3Dphi4/
REPO = Path(__file__).resolve().parents[3]       # repo root
sys.path.insert(0, str(ROOT3D))                  # networks_3d, diffusion_lightning_3d
sys.path.insert(0, str(REPO))                    # phi4_action
os.chdir(ROOT3D)                                 # resolve runs/, trainingdata/ from 3Dphi4/

import h5py
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from phi4_action import phi4_grad_S as _phi4_grad_S_nd


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
    grad_S_fn = functools.partial(
        _phi4_grad_S_nd, k=k, l=l, phi_min=norm_min, phi_max=norm_max,
        spatial_dims=3,
    )
    return model, grad_S_fn, norm_min, norm_max, sigma


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
    if cfgs.shape[0] < cfgs.shape[-1]:
        cfgs = cfgs.transpose(3, 0, 1, 2)
    idx = np.random.default_rng(42).choice(cfgs.shape[0], size=num_samples,
                                           replace=False)
    cfgs_norm = ((cfgs[idx] - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    return x


def one_score_quality(model, grad_S_fn, x_ref, t_eval, batch_size):
    """Per-metric (mean, SEM); 3D needs chunking to fit memory."""
    device = model.device
    N = x_ref.shape[0]
    s_chunks = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            bt = torch.ones(min(batch_size, N - i), device=device) * t_eval
            s_chunks.append(model(x_ref[i:i+batch_size], bt))
        s_model = torch.cat(s_chunks)
        s_true = -grad_S_fn(x_ref)
        s_m = s_model.reshape(N, -1)
        s_t = s_true.reshape(N, -1)
        dot = (s_m * s_t).sum(dim=1)
        norm_m = s_m.norm(dim=1)
        norm_t = s_t.norm(dim=1)
        cos_sim = dot / (norm_m * norm_t + 1e-12)
        rel_mse = (s_m - s_t).pow(2).sum(dim=1) / (s_t.pow(2).sum(dim=1) + 1e-12)
        mag_ratio = norm_m / (norm_t + 1e-12)
    out = {}
    for name, t in [('cos_sim', cos_sim), ('rel_mse', rel_mse), ('mag_ratio', mag_ratio)]:
        out[name] = (t.mean().item(), (t.std() / N ** 0.5).item())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch_window", type=str, default="0-1000000")
    parser.add_argument("--max_ckpts", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--t_eval", type=float, default=1e-4)
    parser.add_argument("--ema", action="store_true")
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
            f"No ckpts in epoch window {args.epoch_window}."
        )
    print(f"Run dir: {model_dir}")
    print(f"Sweeping {len(in_range)} epochs: {in_range}")

    print(f"Building model from {ckpts[in_range[0]]}")
    model, grad_S_fn, norm_min, norm_max, sigma_ve = build_run_context(
        ckpts[in_range[0]], args.device
    )
    print(f"norm range: [{norm_min}, {norm_max}],  VE sigma = {sigma_ve}")

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"
    print(f"Reference data: {args.data_path}")
    x_ref = load_reference_data(
        args.data_path, args.num_samples, norm_min, norm_max, args.device
    )
    print(f"Loaded {x_ref.shape[0]} reference configs, shape {x_ref.shape}")
    print(f"\nt_eval = {args.t_eval:.2e}\n")

    L = args.L
    cos = np.zeros(len(in_range)); cos_sem = np.zeros(len(in_range))
    mse = np.zeros(len(in_range)); mse_sem = np.zeros(len(in_range))
    mag = np.zeros(len(in_range)); mag_sem = np.zeros(len(in_range))
    for i, ep in enumerate(in_range):
        if ckpts[ep] != ckpts[in_range[0]]:
            swap_weights(model, ckpts[ep])
        if args.ema:
            model.ema.copy_to()
        m = one_score_quality(model, grad_S_fn, x_ref, args.t_eval, args.batch_size)
        cos[i], cos_sem[i] = m['cos_sim']
        mse[i], mse_sem[i] = m['rel_mse']
        mag[i], mag_sem[i] = m['mag_ratio']
        print(f"  epoch {ep:>6d}  cos={cos[i]:.4f}  relMSE={mse[i]:.4f}  mag={mag[i]:.4f}")

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'lines.linewidth': 1.2, 'lines.markersize': 4,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ep_arr = np.array(in_range)
    for ax, (vals, errs, lbl) in zip(axes, [
        (cos, cos_sem, 'Cosine similarity'),
        (mse, mse_sem, 'Relative MSE'),
        (mag, mag_sem, 'Magnitude ratio'),
    ]):
        ax.errorbar(ep_arr, vals, yerr=errs, marker='o', capsize=2, color='C0')
        ax.set_xscale('log')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
    axes[2].axhline(1.0, color='gray', ls='--', lw=0.8)
    fig.suptitle(rf'3D: $L={L},\;\kappa={args.k},\;t_{{\mathrm{{eval}}}}={args.t_eval:.0e}$',
                 y=1.02)
    fig.tight_layout()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    if args.ema:
        tag += "_ema"
    out_pdf = out_dir / f"score_quality_vs_epoch_3d_L{L}_k{args.k}{tag}.pdf"
    fig.savefig(out_pdf)
    np.savez(str(out_pdf).replace(".pdf", ".npz"),
             epochs=ep_arr,
             cos_sim=cos, cos_sim_sem=cos_sem,
             rel_mse=mse, rel_mse_sem=mse_sem,
             mag_ratio=mag, mag_ratio_sem=mag_sem,
             t_eval=args.t_eval, sigma_ve=sigma_ve,
             k=args.k, L=L, run_dir=str(model_dir))
    print(f"\nSaved plot to {out_pdf}")
    print(f"Saved data to {str(out_pdf).replace('.pdf', '.npz')}")
    plt.close()


if __name__ == "__main__":
    main()
