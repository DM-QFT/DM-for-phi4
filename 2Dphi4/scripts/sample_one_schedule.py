"""Minimal sampler: one ckpt × one schedule × one schedule-tagged output file.

Usage:
  python scripts/sample_one_schedule.py \
      --run runs/phi4_L128_k0.26_l0.022_ncsnpp_sigma300 \
      --epoch 10000 --num_steps 4000 --num_samples 512 \
      --schedule log --device cuda:2 --seed 20260422
"""
import argparse, functools, sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 2Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import numpy as np
import torch

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--num_steps", type=int, default=4000)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--schedule", type=str, default="log",
                   choices=["linear", "log", "quadratic", "cosine",
                            "power_2", "power_4", "power_6"])
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--seed", type=int, default=20260422)
    args = p.parse_args()

    ckpt_path = Path(args.run) / "models" / f"epoch={args.epoch:04d}.ckpt"
    assert ckpt_path.exists(), f"no ckpt at {ckpt_path}"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    sigma = hp.get("sigma", 150.0)
    norm_min = hp.get("norm_min") or -6.22
    norm_max = hp.get("norm_max") or 6.19
    print(f"[sampler] {ckpt_path}\n          sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]")

    sd = ck.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ck["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    import io
    buf = io.BytesIO(); torch.save(ck, buf); del ck; buf.seek(0)

    mps = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(mps)
    model = DiffusionModel.load_from_checkpoint(buf, score_model=score_model, map_location=args.device, weights_only=False)
    model = model.to(args.device).eval()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[sampler] EM  steps={args.num_steps}  N={args.num_samples}  "
          f"schedule={args.schedule}  seed={args.seed}")
    samples = model.sample(args.num_samples, args.num_steps,
                           schedule=args.schedule)
    samples_norm = samples[:, 0].cpu().numpy()
    samples_phys = (samples_norm + 1) / 2 * (norm_max - norm_min) + norm_min
    out = samples_phys.transpose(1, 2, 0).astype(np.float32)

    out_dir = Path(args.run) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"samples_em_steps{args.num_steps}_"
                          f"epoch={args.epoch:04d}_{args.schedule}.npy")
    np.save(out_path, out)
    print(f"[sampler] saved {out_path}  shape={out.shape}")


if __name__ == "__main__":
    main()
