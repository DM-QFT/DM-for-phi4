"""Generate an L32 t=0,...,100 DM chain from N(0,1) initial fields."""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


MODEL_B_DIR = Path(__file__).resolve().parent.parent
DM_DIR = MODEL_B_DIR.parent
sys.path.insert(0, str(DM_DIR))

from diffusion_lightning import marginal_prob_std

if __package__:
    from .cond_lightning import ConditionalDiffusionModel, NCSNppCond2D
else:
    from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D


RUN_DIR = MODEL_B_DIR / "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_all_data"
DEFAULT_CHECKPOINT = RUN_DIR / "models/epoch=2257.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--n_samples", type=int, default=5120)
    parser.add_argument("--batch", type=int, default=5120)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--gaussian_seed", type=int, default=20260721)
    parser.add_argument("--sampling_seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    return parser.parse_args()


def save(path, arrays, metadata):
    json_path = path.with_suffix(".json")
    if path.exists() or json_path.exists():
        raise FileExistsError(path)
    start = time.perf_counter()
    np.savez_compressed(path, **arrays)
    metadata["npz_write_seconds"] = time.perf_counter() - start
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    hparams = checkpoint_data["hyper_parameters"]
    epoch = int(checkpoint_data["epoch"]) + 1
    global_step = checkpoint_data.get("global_step")
    L = int(hparams["L"])
    sigma = float(hparams["sigma"])
    A_res = float(hparams["norm_max"])
    A_cond = float(hparams["cond_norm_max"])
    del checkpoint_data

    output_dir = args.output_dir or (
        RUN_DIR / "samples"
        / f"gaussian_chain_ep{epoch}_N{args.n_samples}_seed{args.gaussian_seed}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(args.gaussian_seed)
    phi = rng.standard_normal(
        (args.n_samples, L, L), dtype=np.float32
    )
    indices = np.arange(args.n_samples, dtype=np.int64)
    initial_path = output_dir / (
        f"gaussian_ep{epoch}_t0_N{args.n_samples}_seed{args.gaussian_seed}.npz"
    )
    save(
        initial_path,
        {
            "phi_gen": phi,
            "cond_indices": indices,
            "n_per_cond": np.int64(1),
        },
        {
            "distribution": "iid N(0,1) per lattice site",
            "gaussian_rng": "numpy Generator(PCG64).standard_normal float32",
            "gaussian_seed": args.gaussian_seed,
            "n_samples": args.n_samples,
            "L": L,
            "mean": float(phi.mean(dtype=np.float64)),
            "std": float(phi.std(dtype=np.float64)),
        },
    )

    device = torch.device(args.device)
    score_model = NCSNppCond2D(
        functools.partial(marginal_prob_std, sigma=sigma),
        in_channels=int(hparams.get("in_channels", 2)),
    )
    model = ConditionalDiffusionModel.load_from_checkpoint(
        str(checkpoint), score_model=score_model, map_location=device,
        weights_only=False,
    ).to(device)
    model.score_model = model.score_model.to(memory_format=torch.channels_last)
    model.score_model = torch.compile(
        model.score_model, mode=args.compile_mode
    )
    model.eval()

    print(
        f"checkpoint={checkpoint}\ninitial={initial_path}\n"
        f"samples={args.n_samples}, batch={args.batch}, "
        f"steps={args.num_steps}, schedule=log, compile={args.compile_mode}",
        flush=True,
    )
    parent = initial_path
    with torch.inference_mode(), model.ema.average_parameters():
        for stage, absolute_time in enumerate(range(10, 101, 10)):
            seed = args.sampling_seed + stage
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            dphi = np.empty_like(phi)
            phi_next = np.empty_like(phi)
            zero_mode_raw = np.empty(args.n_samples, dtype=np.float32)

            torch.cuda.synchronize(device)
            sample_start = time.perf_counter()
            for start in range(0, args.n_samples, args.batch):
                stop = min(start + args.batch, args.n_samples)
                condition = torch.from_numpy(
                    phi[start:stop] / np.float32(A_cond)
                )[:, None].to(device)
                generated = model.sample(
                    condition, num_steps=args.num_steps,
                    schedule="log", use_ema=False,
                )[:, 0].float().cpu().numpy()
                generated *= np.float32(A_res)
                dphi[start:stop] = generated
                phi_next[start:stop] = phi[start:stop] + generated
                zero_mode_raw[start:stop] = generated.mean(axis=(1, 2))
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - sample_start

            output = output_dir / (
                f"gaussian_chain_ep{epoch}_t{absolute_time}_N{args.n_samples}_"
                f"steps{args.num_steps}_log_seed{seed}.npz"
            )
            save(
                output,
                {
                    "phi_gen": phi_next,
                    "dphi": dphi,
                    "zero_mode_raw": zero_mode_raw,
                    "cond_indices": indices,
                    "n_per_cond": np.int64(1),
                },
                {
                    "checkpoint": str(checkpoint),
                    "epoch": epoch,
                    "global_step": global_step,
                    "parent": str(parent),
                    "absolute_time": absolute_time,
                    "horizon": float(hparams["horizon"]),
                    "n_samples": args.n_samples,
                    "L": L,
                    "kappa": hparams.get("kappa"),
                    "lam": hparams.get("lam"),
                    "sigma": sigma,
                    "A_res": A_res,
                    "A_cond": A_cond,
                    "num_steps": args.num_steps,
                    "schedule": "log",
                    "batch": args.batch,
                    "seed": seed,
                    "compile_mode": args.compile_mode,
                    "projected": True,
                    "sampling_seconds": seconds,
                },
            )
            print(
                f"t={absolute_time}: {seconds:.2f} s, "
                f"{args.n_samples / seconds:.1f} samples/s, "
                f"max |mean(dphi)|={np.abs(zero_mode_raw).max():.3e}\n"
                f"saved {output}",
                flush=True,
            )
            phi = phi_next
            parent = output


if __name__ == "__main__":
    main()
