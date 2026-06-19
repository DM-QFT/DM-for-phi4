"""β_conf(t) along the reverse-SDE trajectory for 3D phi^4 DM.

Mirror of ``2Dphi4/analysis/beta_conf_along_reverse.py`` with the 3D model
classes (``DiffusionModel3D``, ``NCSNpp3D``) and 3D HMC data files.

Run from `3Dphi4/`:
    python "analysis/physical observable/beta_conf_along_reverse_3d.py" \
        --L 64 --k 0.1923 --l 0.9 --output_suffix _sigma2048 \
        --ep 9929 --num_steps 2000 --schedule log \
        --num_samples 64 --record_every 8 \
        --hmc_n 256 --device cuda:0 --seed 0
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # 3Dphi4/analysis/physical observable/
PROJ = HERE.parents[1]                          # 3Dphi4/
ROOT = PROJ.parent                              # repo root
sys.path.append(str(ROOT))
sys.path.append(str(PROJ))                      # 3Dphi4/ for local modules
sys.path.append(str(ROOT / "2Dphi4" / "analysis" / "physical observable"))   # shared SDE+β̂ helpers

from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from networks_3d import NCSNpp3D
from phi4_action import phi4_beta_conf  # noqa: F401  (re-export check)

from beta_conf_along_reverse import (
    load_hmc_2d as load_hmc,            # dim-agnostic (works for 3D cfgs too)
    reverse_sde_with_beta,
    forward_noised_hmc_curve,
    plot_curves,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--output_suffix", type=str, default="_sigma2048")
    p.add_argument("--ep", type=int, default=9929)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="log")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--num_samples", type=int, default=64,
                   help="DM batch (3D needs less to saturate stats: L^3 dof per cfg)")
    p.add_argument("--record_every", type=int, default=8)
    p.add_argument("--hmc_path", type=str, default=None)
    p.add_argument("--hmc_n", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_autocast", action="store_true",
                   help="Disable bf16 autocast in EM loop (precision isolation).")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Override training run dir (e.g. for multi-L cross-L).")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path(
        f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.output_suffix}")
    ckpts = sorted(run_dir.glob(f"models/*{args.ep}*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt for ep={args.ep} in {run_dir}/models")
    ckpt_path = ckpts[-1]
    print(f"checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma_model = hparams.get("sigma", 2048.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"  sigma_model={sigma_model}  norm=[{norm_min:.4f}, {norm_max:.4f}]")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mp_std = functools.partial(marginal_prob_std, sigma=sigma_model)
    score_model = NCSNpp3D(mp_std)
    model = DiffusionModel3D.load_from_checkpoint(ckpt_path, score_model=score_model)
    # cross-L: override L on a fully-convolutional score net
    if model.L != args.L:
        print(f"  cross-L override: model.L {model.L} → {args.L}")
        model.L = args.L
    model = model.to(args.device).eval()
    assert model.spatial_dims == 3

    print(f"reverse SDE: num_steps={args.num_steps}  schedule={args.schedule}  "
          f"batch={args.num_samples}")
    dm = reverse_sde_with_beta(
        model, args.num_samples, args.num_steps, args.schedule, args.eps,
        args.k, args.l, norm_min, norm_max, record_every=args.record_every,
        use_autocast=not args.no_autocast,
    )
    print(f"  recorded {len(dm['t'])} points; "
          f"final β̂={dm['mean'][-1]:.4f} ± {dm['sem'][-1]:.4f} "
          f"at t={dm['t'][-1]:.2e}")

    if args.hmc_path is None:
        args.hmc_path = (f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}"
                         f"_{args.L}^3.jld2")
    hmc_path = Path(args.hmc_path)
    if not hmc_path.exists():
        raise FileNotFoundError(hmc_path)
    print(f"HMC: {hmc_path.name}  using {args.hmc_n} cfgs")

    hmc_cfgs = load_hmc(hmc_path, args.hmc_n)
    pmin_hmc, pmax_hmc = float(hmc_cfgs.min()), float(hmc_cfgs.max())
    print(f"  HMC range [{pmin_hmc:.3f}, {pmax_hmc:.3f}]")
    hmc_norm = 2.0 * (hmc_cfgs - norm_min) / (norm_max - norm_min) - 1.0
    hmc_norm_t = torch.from_numpy(hmc_norm).unsqueeze(1).float().to(args.device)

    hmc_fwd = forward_noised_hmc_curve(
        hmc_norm_t, dm["t"], dm["sigma_t"],
        args.k, args.l, norm_min, norm_max, seed=args.seed + 1,
        spatial_dims=3,
    )
    print(f"  HMC@smallest grid t ({dm['t'][-1]:.2e}): "
          f"β̂={hmc_fwd['mean'][-1]:.4f} ± {hmc_fwd['sem'][-1]:.4f}")

    out_dir = Path("results/beta_conf_traj")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = (Path(args.run_dir).name.replace("phi4_3d_", "")
                 if args.run_dir else f"L{args.L}_k{args.k}_l{args.l}")
    tag = (f"3d_{model_tag}_crossL{args.L}_ep{args.ep}_steps{args.num_steps}"
           f"_{args.schedule}_N{args.num_samples}"
           + ("_fp32" if args.no_autocast else ""))
    npz_path = out_dir / f"beta_conf_t_{tag}.npz"
    np.savez(npz_path,
             t=dm["t"], sigma_t=dm["sigma_t"],
             dm_mean=dm["mean"], dm_sem=dm["sem"],
             dm_p05=dm["p05"], dm_p50=dm["p50"], dm_p95=dm["p95"],
             hmc_mean=hmc_fwd["mean"], hmc_sem=hmc_fwd["sem"],
             hmc_p05=hmc_fwd["p05"], hmc_p50=hmc_fwd["p50"], hmc_p95=hmc_fwd["p95"],
             k=args.k, l=args.l, L=args.L, sigma_model=sigma_model,
             norm_min=norm_min, norm_max=norm_max, ep=args.ep)
    print(f"saved npz → {npz_path}")

    pdf_path = out_dir / f"beta_conf_t_{tag}.pdf"
    plot_curves(dm, hmc_fwd, pdf_path,
                title=f"3D  L={args.L}  κ={args.k}  λ={args.l}  ep={args.ep}")


if __name__ == "__main__":
    main()
