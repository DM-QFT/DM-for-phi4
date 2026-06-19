"""
Calibrate MALA step size by sweeping h on early- vs late-epoch windows.

The earlier single-checkpoint version (epoch=49 vs epoch=9999) is replaced by
window averaging analogous to the 3D script: an early window (default
``--early_window 1-49``) and a late window (default ``--late_window
6000-10000``).  Each point in the plot is averaged over the ckpts falling
inside the requested window, with the SEM as error bar.

Usage:
    python acceptance_rate/calibrate_step_size.py --device cuda:0 \
        --L 128 --k 0.28 --run_dir runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640
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
# latter pulls in scipy at module level (cumulants_from_moments) which isn't
# installed in nenv2 on Miyabi.
from phi4_action import phi4_action as _phi4_action_nd


def find_epoch_checkpoints(model_dir):
    """Match either 'epoch=epoch=NNNN.ckpt' (single-L training) or
    'epoch=NNNN.ckpt' (newer convention)."""
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=*.ckpt")):
        if '-v' in f.stem:
            continue
        m = re.search(r'epoch=(\d+)$', f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def build_run_context(ckpt_path, device):
    """Build DiffusionModel + action_fn once per run. Network architecture,
    sigma, norm_min/max, k, lambda are constant within a run, so we only
    pay the construction cost once and then call swap_weights() per ckpt."""
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
    # DiffusionModel.on_load_checkpoint strips/adds the _orig_mod. prefix.
    model = DiffusionModel.load_from_checkpoint(
        buf, score_model=score_model, map_location=device, weights_only=False
    )
    model = model.to(device).eval()
    # ExponentialMovingAverage is a plain Python attribute, not an nn.Module,
    # so model.to() doesn't follow it. Move shadow_params explicitly so the
    # whole module is on the same device.
    if hasattr(model, 'ema'):
        model.ema.to(device)
    action_fn = functools.partial(
        _phi4_action_nd, k=k, l=l, phi_min=norm_min, phi_max=norm_max,
        spatial_dims=2,
    )
    return model, action_fn, norm_min, norm_max, sigma


def swap_weights(model, ckpt_path):
    """Replace weights of an existing DiffusionModel in-place. Runs the same
    _orig_mod prefix fixup that Lightning's load_from_checkpoint does."""
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


def sweep_one(model, action_fn, h_values, x_ref, mh_steps, t_mh):
    """Run MALA at each step size on fixed reference data."""
    results = []
    device = model.device
    num_samples = x_ref.shape[0]

    for h_val in h_values:
        x = x_ref.clone()
        h = torch.tensor(h_val, device=device, dtype=x.dtype)
        batch_t = torch.ones(num_samples, device=device) * t_mh
        total_accept = torch.zeros(num_samples, device=device)

        with torch.no_grad():
            for _ in range(mh_steps):
                score_x = model(x, batch_t)
                noise = torch.randn_like(x)
                y = x + h * score_x + torch.sqrt(2 * h) * noise

                score_y = model(y, batch_t)
                drift = h * (score_x + score_y)
                log_q = -(0.25 / h) * torch.sum(
                    (torch.sqrt(2 * h) * noise + drift) ** 2, dim=(1, 2, 3)
                ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3))
                log_pi = action_fn(x) - action_fn(y)

                accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
                accept = torch.rand(num_samples, device=device) < accept_prob
                x = torch.where(accept[:, None, None, None], y, x)
                total_accept += accept.float()

        acc = total_accept / mh_steps
        results.append((acc.mean().item(), (acc.std() / acc.shape[0] ** 0.5).item()))
        print(f"    h = {h_val:.2e}  ->  accept = {acc.mean().item():.4f} "
              f"+- {acc.std().item():.4f}")

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
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA-averaged weights for the score forward "
                             "pass. Output filename gets an '_ema' suffix to "
                             "avoid overwriting the no-EMA results.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Training data .jld2 (default: auto-detect)")
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

    # Build the model + action_fn ONCE (sigma, norm range, k, lambda, network
    # architecture are all constant within a single run). For each subsequent
    # epoch we only call swap_weights() to overwrite the weights in-place.
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
    c_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    h_values = c_values / (L ** 2)

    print(f"\nSweeping c = h * L^2  over {c_values.tolist()}")
    print(f"(h = c / {L}^2 = c / {L**2})\n")

    def _sweep_window(name, eps):
        per_ep = np.zeros((len(eps), len(c_values)))
        for i, ep in enumerate(eps):
            print(f"\n--- {name}  epoch {ep}  ({i+1}/{len(eps)}) ---")
            # Model already holds early_eps[0]'s weights from build_run_context;
            # swap_weights for every other ckpt avoids rebuilding the network.
            if ckpts[ep] != ckpts[early_eps[0]]:
                swap_weights(model, ckpts[ep])
            # If --ema, overwrite live score_model params with EMA shadow params.
            # Done after every swap because swap_weights restores the raw weights.
            if args.ema:
                model.ema.copy_to()
            res = sweep_one(model, action_fn, h_values, x_ref,
                            args.mh_steps, args.t_mh)
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

    print(f"\n{'='*50}")
    print(f"{'c':>8s}  {'h':>10s}  {'early':>7s}  {'late':>7s}  {'gap':>7s}")
    print(f"{'='*50}")
    for i, c in enumerate(c_values):
        print(f"{c:8.2f}  {h_values[i]:10.2e}  "
              f"{acc_early[i]:7.4f}  {acc_late[i]:7.4f}  {gap[i]:+7.4f}")

    useful = (acc_late > 0.3) & (acc_late < 0.85)
    if useful.any():
        best_idx = np.where(useful, gap, -np.inf).argmax()
    else:
        best_idx = gap.argmax()
    print(f"\nRecommended:  c = {c_values[best_idx]:.2f}  "
          f"(h = {h_values[best_idx]:.2e})")
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

    ax1.errorbar(c_values, acc_early, yerr=err_early,
                 label=f'Early epochs {early_lo}-{early_hi} '
                       f'($n={len(early_eps)}$)',
                 marker='o', capsize=3, color='C3')
    ax1.errorbar(c_values, acc_late, yerr=err_late,
                 label=f'Late epochs {late_lo}-{late_hi} '
                       f'($n={len(late_eps)}$)',
                 marker='s', capsize=3, color='C0')
    ax1.axvline(c_values[best_idx], ls='--', color='gray', alpha=0.6,
                label=f'Recommended $c={c_values[best_idx]:.1f}$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$c\;(h = c / L^2)$')
    ax1.set_ylabel('Acceptance rate')
    t_exp = int(np.floor(np.log10(args.t_mh)))
    t_man = args.t_mh / 10**t_exp
    t_str = rf'10^{{{t_exp}}}' if t_man == 1 else rf'{t_man:.0f}\times10^{{{t_exp}}}'
    ax1.set_title(rf'$L={L},\;\kappa={args.k},\;t_{{\mathrm{{mh}}}}={t_str}$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    ax2.bar(range(len(c_values)), gap, tick_label=[f'{c:.2g}' for c in c_values],
            color=['C2' if useful[i] else 'C7' for i in range(len(c_values))])
    ax2.set_xlabel(r'$c$')
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else Path(SCRIPT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    if args.ema:
        tag += "_ema"
    out_pdf = out_dir / f"calibrate_h_L{L}_k{args.k}{tag}.pdf"
    fig.savefig(out_pdf)
    np.savez(str(out_pdf).replace(".pdf", ".npz"),
             c_values=c_values, h_values=h_values,
             acc_early=acc_early, acc_late=acc_late,
             err_early=err_early, err_late=err_late, gap=gap,
             early_epochs=np.array(early_eps), late_epochs=np.array(late_eps),
             early_per_ep=early_per_ep, late_per_ep=late_per_ep,
             t_mh=args.t_mh, sigma_ve=sigma_ve,
             k=args.k, L=L, run_dir=str(model_dir))
    print(f"\nSaved plot to {out_pdf}")
    print(f"Saved data to {str(out_pdf).replace('.pdf', '.npz')}")
    plt.close()


if __name__ == "__main__":
    main()
