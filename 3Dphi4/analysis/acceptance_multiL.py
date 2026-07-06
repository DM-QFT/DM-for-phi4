"""
MALA acceptance rate for 3D phi^4 multi-L models, evaluated at multiple
lattice sizes (in-distribution training L's plus OOD L=64).

Mirrors `acceptance_vs_epoch.py` (single-L) but:
  * loads checkpoints written by `train_phi4_multiL.py` (`epoch=NNNN.ckpt`)
  * evaluates the same model at several L's per ckpt
  * averages over the last few log-scale ckpts (default: epochs 6280-9111)

Usage:
    python analysis/acceptance_multiL.py --device cuda:0 \\
        --run_dir runs/phi4_3d_Lmulti4-8-16-32_k0.18_l0.9_ncsnpp_sigma200 \\
        --L_eval 4 8 16 32 64 --k 0.18
"""

import sys
import re
import functools
import argparse
import io
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))   # 3Dphi4/
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # repo root

import h5py
import torch
import numpy as np

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from sample_phi4 import phi4_action_3d


def find_multiL_checkpoints(model_dir):
    """Return dict {epoch: path} for ckpts named 'epoch=NNNN.ckpt'."""
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=*.ckpt")):
        if "-v" in f.stem:
            continue
        m = re.fullmatch(r"epoch=(\d+)", f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def load_model(ckpt_path, k, l, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    sigma = hp.get("sigma", 800.0)
    norm_min = hp.get("norm_min")
    norm_max = hp.get("norm_max")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp3D(marginal_prob_std_fn)

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel3D.load_from_checkpoint(buf, score_model=score_model)
    model = model.to(device).eval()

    action_fn = functools.partial(
        phi4_action_3d, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )
    return model, action_fn, norm_min, norm_max


def load_reference(data_path, num_samples, norm_min, norm_max, device, seed=42):
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(cfgs.shape[0], size=min(num_samples, cfgs.shape[0]), replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    return x


def measure_acceptance(model, action_fn, x_ref, h, t_mh, batch_size):
    device = model.device
    N = x_ref.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x_ref.dtype)
    x = x_ref

    with torch.no_grad():
        score_x = torch.cat([
            model(x[i:i+batch_size],
                  torch.ones(min(batch_size, N - i), device=device) * t_mh)
            for i in range(0, N, batch_size)
        ])
        noise = torch.randn_like(x)
        y = x + h_t * score_x + torch.sqrt(2 * h_t) * noise

        score_y = torch.cat([
            model(y[i:i+batch_size],
                  torch.ones(min(batch_size, N - i), device=device) * t_mh)
            for i in range(0, N, batch_size)
        ])
        drift = h_t * (score_x + score_y)
        log_q = -(0.25 / h_t) * torch.sum(
            (torch.sqrt(2 * h_t) * noise + drift) ** 2, dim=(1, 2, 3, 4)
        ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3, 4))
        log_pi = action_fn(x) - action_fn(y)

        accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
        accept = (torch.rand(N, device=device) < accept_prob).float()

    return accept.mean().item(), (accept.std() / N**0.5).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--L_eval", type=int, nargs="+", required=True)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--c", type=float, default=0.5)
    p.add_argument("--t_mh", type=float, default=1e-4)
    p.add_argument("--data_prefix", type=str,
                   default="trainingdata/cfgs_wolff_fahmc")
    p.add_argument("--last_n", type=int, default=5,
                   help="Average over the last N log-scale ckpts.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = find_multiL_checkpoints(run_dir / "models")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/models")
    epochs = sorted(ckpts.keys())[-args.last_n:]
    print(f"Run: {run_dir}")
    print(f"Evaluating epochs: {epochs}")
    print(f"L_eval: {args.L_eval}")
    print(f"c={args.c}, t_mh={args.t_mh}, num_samples={args.num_samples}")

    # results[L][epoch] = (mean, std)
    results = {L: {} for L in args.L_eval}

    # Pre-load reference data per L (norm range comes from each ckpt; reload)
    for ep in epochs:
        print(f"\n--- Epoch {ep} ---")
        model, action_fn_template, norm_min, norm_max = load_model(
            ckpts[ep], args.k, args.l, args.device
        )
        for L in args.L_eval:
            data_path = f"{args.data_prefix}_k={args.k}_l={args.l}_{L}^3.jld2"
            if not Path(data_path).is_file():
                print(f"  [skip] {data_path} not found")
                continue
            x_ref = load_reference(
                data_path, args.num_samples, norm_min, norm_max,
                args.device, seed=args.seed,
            )
            action_fn = functools.partial(
                phi4_action_3d, k=args.k, l=args.l,
                phi_min=norm_min, phi_max=norm_max,
            )
            h = args.c / (L ** 3)
            bs = args.batch_size
            if L >= 32:
                bs = min(bs, 32)
            if L >= 64:
                bs = min(bs, 8)
            mean, std = measure_acceptance(
                model, action_fn, x_ref, h, args.t_mh, bs
            )
            results[L][ep] = (mean, std)
            print(f"  L={L:>3d}  accept = {mean:.4f} ± {std:.4f}  (h={h:.2e})")
            del x_ref
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    # Aggregate
    print("\n" + "=" * 60)
    print(f"{'L':>4s}  {'mean accept':>14s}  {'sd over epochs':>15s}  {'n_ep':>5s}")
    print("=" * 60)
    summary = []
    for L in args.L_eval:
        vals = [v[0] for v in results[L].values()]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append((L, m, s, len(vals)))
        print(f"{L:>4d}  {m:>14.4f}  {s:>15.4f}  {len(vals):>5d}")

    if args.out_csv is None:
        run_tag = run_dir.name
        args.out_csv = f"results/calibration/acceptance_multiL_{run_tag}.csv"
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# run_dir={run_dir}\n")
        f.write(f"# epochs={epochs}\n")
        f.write(f"# c={args.c}, t_mh={args.t_mh}, N={args.num_samples}\n")
        f.write("L,epoch,acc_mean,acc_std\n")
        for L in args.L_eval:
            for ep in sorted(results[L].keys()):
                m, s = results[L][ep]
                f.write(f"{L},{ep},{m:.6f},{s:.6f}\n")
        f.write("\n# summary (mean, sd_over_epochs, n_ep)\n")
        f.write("L,mean,sd,n_ep\n")
        for L, m, s, n in summary:
            f.write(f"{L},{m:.6f},{s:.6f},{n}\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
