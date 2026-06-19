"""
Calibrate t_mh by sweeping evaluation time on early- vs late-epoch windows.

Replaces the previous single-checkpoint diagnostic (epoch=49 vs epoch=9999)
with window averaging analogous to the 3D script: an early window (default
``--early_window 1-49``) and a late window (default ``--late_window
6000-10000``).

Usage:
    python acceptance_rate/calibrate_t_mh.py --device cuda:0 \
        --L 128 --k 0.28 --c 0.2 \
        --run_dir runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import re
import functools
import argparse
from pathlib import Path

import io

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
# Import phi4_action from the root module directly, NOT via sample_phi4 — the
# latter pulls in scipy at module level which isn't installed in nenv2 on Miyabi.
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
    """Build DiffusionModel + action_fn once per run. See calibrate_step_size.py
    for the rationale: sigma, norm range, network architecture are constant
    within a run, so we only construct once."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19

    path_str = str(ckpt_path)
    k = float(re.search(r'_k([\d.]+)', path_str).group(1))
    l = float(re.search(r'_l([\d.]+)', path_str).group(1))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)

    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel.load_from_checkpoint(
        buf, score_model=score_model, map_location=device, weights_only=False
    )
    model = model.to(device).eval()
    if hasattr(model, 'ema'):
        model.ema.to(device)
    action_fn = functools.partial(
        _phi4_action_nd, k=k, l=l, phi_min=norm_min, phi_max=norm_max,
        spatial_dims=2,
    )
    return model, action_fn, norm_min, norm_max, sigma


def swap_weights(model, ckpt_path):
    """Replace weights of an existing DiffusionModel in-place."""
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
    """Load training data, normalise to [-1, 1], return (num_samples, 1, L, L)."""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def sweep_t_mh(model, action_fn, t_mh_values, x_ref, mh_steps, h):
    """Run MALA at fixed step size, sweeping t_mh."""
    results = []
    device = model.device
    num_samples = x_ref.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x_ref.dtype)

    for t_mh in t_mh_values:
        x = x_ref.clone()
        batch_t = torch.ones(num_samples, device=device) * t_mh
        total_accept = torch.zeros(num_samples, device=device)

        with torch.no_grad():
            for _ in range(mh_steps):
                score_x = model(x, batch_t)
                noise = torch.randn_like(x)
                y = x + h_t * score_x + torch.sqrt(2 * h_t) * noise

                score_y = model(y, batch_t)
                drift = h_t * (score_x + score_y)
                log_q = -(0.25 / h_t) * torch.sum(
                    (torch.sqrt(2 * h_t) * noise + drift) ** 2, dim=(1, 2, 3)
                ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3))
                log_pi = action_fn(x) - action_fn(y)

                accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
                accept = torch.rand(num_samples, device=device) < accept_prob
                x = torch.where(accept[:, None, None, None], y, x)
                total_accept += accept.float()

        acc = total_accept / mh_steps
        sem = (acc.std() / acc.shape[0] ** 0.5).item()
        results.append((acc.mean().item(), sem))
        print(f"    t_mh = {t_mh:.1e}  ->  accept = {acc.mean().item():.4f} "
              f"+- {sem:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Override the model directory (default auto-builds "
                             "from L/k/l/network without sigma suffix).")
    parser.add_argument("--early_window", type=str, default="1-49",
                        help="Early-epoch window for averaging, e.g. '1-49'.")
    parser.add_argument("--late_window", type=str, default="6000-10000",
                        help="Late-epoch window for averaging, e.g. '6000-10000'.")
    parser.add_argument("--max_early_ckpts", type=int, default=12,
                        help="Cap on number of early ckpts; <=0 uses all "
                             "ckpts in the window.")
    parser.add_argument("--max_late_ckpts", type=int, default=12,
                        help="Cap on number of late ckpts; <=0 uses all "
                             "ckpts in the window.")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=1)
    parser.add_argument("--c", type=float, default=0.2,
                        help="Fixed step size coefficient h = c / L^2")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA-averaged weights for the score forward "
                             "pass. Output filename gets an '_ema' suffix.")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write the .pdf/.npz (default: SCRIPT_DIR)")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="Tag added to output filenames, e.g. 'sigma640'.")
    args = parser.parse_args()

    def _parse_window(w):
        a, b = w.split("-")
        return int(a), int(b)
    early_lo, early_hi = _parse_window(args.early_window)
    late_lo, late_hi = _parse_window(args.late_window)

    if args.run_dir:
        model_dir = Path(args.run_dir)
    else:
        model_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}")
    ckpts = find_epoch_checkpoints(model_dir / "models")
    all_epochs = sorted(ckpts.keys())
    if not all_epochs:
        raise RuntimeError(f"No ckpts found in {model_dir / 'models'}")

    def _select(epochs, lo, hi, cap):
        in_range = [e for e in epochs if lo <= e <= hi]
        if not in_range:
            return []
        if cap <= 0 or len(in_range) <= cap:
            return in_range
        idx = np.unique(np.round(np.linspace(0, len(in_range) - 1, cap)).astype(int))
        return [in_range[i] for i in idx]
    early_eps = _select(all_epochs, early_lo, early_hi, args.max_early_ckpts)
    late_eps = _select(all_epochs, late_lo, late_hi, args.max_late_ckpts)
    if not early_eps or not late_eps:
        raise RuntimeError(
            f"No ckpts in early={early_eps} or late={late_eps} windows "
            f"(have {len(all_epochs)} ckpts spanning {all_epochs[0]}-{all_epochs[-1]})."
        )
    print(f"Run dir: {model_dir}")
    print(f"Early window {args.early_window}: {early_eps}")
    print(f"Late  window {args.late_window}: {late_eps}")

    # Build model once; swap weights per epoch (see calibrate_step_size.py).
    print(f"Building model from {ckpts[early_eps[0]]}")
    model, action_fn, norm_min, norm_max, sigma_ve = build_run_context(
        ckpts[early_eps[0]], args.device
    )
    print(f"norm range: [{norm_min}, {norm_max}],  VE sigma = {sigma_ve}")

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    print(f"Reference data: {args.data_path}")
    np.random.seed(42)
    x_ref = load_reference_data(
        args.data_path, args.num_samples, norm_min, norm_max, args.device
    )
    print(f"Loaded {x_ref.shape[0]} reference configs, shape {x_ref.shape}")

    L = args.L
    h = args.c / (L ** 2)
    t_mh_values = np.array([1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1])

    print(f"\nFixed step size: c = {args.c}, h = {h:.2e}")
    print(f"Sweeping t_mh over {t_mh_values.tolist()}\n")

    def _sweep_window(name, eps):
        per_ep = np.zeros((len(eps), len(t_mh_values)))
        for i, ep in enumerate(eps):
            print(f"\n--- {name}  epoch {ep}  ({i+1}/{len(eps)}) ---")
            if ckpts[ep] != ckpts[early_eps[0]]:
                swap_weights(model, ckpts[ep])
            if args.ema:
                model.ema.copy_to()
            res = sweep_t_mh(model, action_fn, t_mh_values, x_ref,
                             args.mh_steps, h)
            per_ep[i] = [r[0] for r in res]
        return per_ep

    print(f"{'='*50}\nEarly window\n{'='*50}")
    early_per_ep = _sweep_window("EARLY", early_eps)
    print(f"\n{'='*50}\nLate window\n{'='*50}")
    late_per_ep = _sweep_window("LATE", late_eps)

    acc_early = early_per_ep.mean(axis=0)
    acc_late = late_per_ep.mean(axis=0)
    err_early = (early_per_ep.std(axis=0, ddof=1) / np.sqrt(len(early_eps))
                 if len(early_eps) > 1 else np.zeros_like(acc_early))
    err_late = (late_per_ep.std(axis=0, ddof=1) / np.sqrt(len(late_eps))
                if len(late_eps) > 1 else np.zeros_like(acc_late))
    gap = acc_late - acc_early

    print(f"\n{'='*60}")
    print(f"{'t_mh':>10s}  {'early':>7s}  {'late':>7s}  {'gap':>7s}")
    print(f"{'='*60}")
    for i, t in enumerate(t_mh_values):
        print(f"{t:10.1e}  {acc_early[i]:7.4f}  {acc_late[i]:7.4f}  {gap[i]:+7.4f}")

    useful = (acc_late > 0.3) & (acc_late < 0.85)
    if useful.any():
        best_idx = np.where(useful, gap, -np.inf).argmax()
    else:
        best_idx = gap.argmax()
    print(f"\nRecommended:  t_mh = {t_mh_values[best_idx]:.1e}")
    print(f"  early accept = {acc_early[best_idx]:.4f}")
    print(f"  late  accept = {acc_late[best_idx]:.4f}")
    print(f"  gap          = {gap[best_idx]:+.4f}")

    # --- plot (publication quality) ---
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(t_mh_values, acc_early, yerr=err_early,
                 label=f'Early epochs {early_lo}-{early_hi} '
                       f'($n={len(early_eps)}$)',
                 marker='o', capsize=3, color='C3')
    ax1.errorbar(t_mh_values, acc_late, yerr=err_late,
                 label=f'Late epochs {late_lo}-{late_hi} '
                       f'($n={len(late_eps)}$)',
                 marker='s', capsize=3, color='C0')

    def _sci(v):
        e = int(np.floor(np.log10(v)))
        m = v / 10**e
        return rf'10^{{{e}}}' if m == 1 else rf'{m:.0f}\!\times\!10^{{{e}}}'

    ax1.axvline(t_mh_values[best_idx], ls='--', color='gray', alpha=0.6,
                label=rf'Recommended $t_{{\mathrm{{mh}}}}={_sci(t_mh_values[best_idx])}$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$t_{\mathrm{mh}}$')
    ax1.set_ylabel('Acceptance rate')
    ax1.set_title(rf'$L={L},\;\kappa={args.k},\;c={args.c}$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    colors = ['C2' if useful[i] else 'C7' for i in range(len(t_mh_values))]
    bar_labels = []
    for t in t_mh_values:
        e = int(np.floor(np.log10(t)))
        m = t / 10**e
        if m == 1:
            bar_labels.append(rf'$10^{{{e}}}$')
        else:
            bar_labels.append(rf'${m:.0f}\!\times\!10^{{{e}}}$')
    ax2.bar(range(len(t_mh_values)), gap,
            tick_label=bar_labels, color=colors)
    ax2.set_xlabel(r'$t_{\mathrm{mh}}$')
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), fontsize=9)

    fig.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else Path(SCRIPT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    if args.ema:
        tag += "_ema"
    out_pdf = out_dir / f"calibrate_t_mh_L{L}_k{args.k}{tag}.pdf"
    fig.savefig(out_pdf)
    np.savez(str(out_pdf).replace(".pdf", ".npz"),
             t_mh_values=t_mh_values,
             acc_early=acc_early, acc_late=acc_late,
             err_early=err_early, err_late=err_late, gap=gap,
             early_epochs=np.array(early_eps), late_epochs=np.array(late_eps),
             early_per_ep=early_per_ep, late_per_ep=late_per_ep,
             c=args.c, h=h, sigma_ve=sigma_ve,
             k=args.k, L=L, run_dir=str(model_dir))
    print(f"\nSaved plot to {out_pdf}")
    print(f"Saved data to {str(out_pdf).replace('.pdf', '.npz')}")
    plt.close()


if __name__ == "__main__":
    main()
