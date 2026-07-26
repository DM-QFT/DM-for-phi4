"""Continue the epoch-2257 L32 chain from t=20 to t=100."""

from __future__ import annotations

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
if str(DM_DIR) not in sys.path:
    sys.path.insert(0, str(DM_DIR))

from diffusion_lightning import marginal_prob_std

if __package__:
    from .cond_lightning import ConditionalDiffusionModel, NCSNppCond2D
else:
    from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D


RUN_DIR = (
    MODEL_B_DIR
    / "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_all_data"
)
DEFAULT_CHECKPOINT = RUN_DIR / "models/epoch=2257.ckpt"
DEFAULT_INPUT = (
    RUN_DIR
    / "samples/chain_ep2257_t20_via_t10dm5120_steps2000_log_seed5678.npz"
)
TIMES = range(30, 101, 10)
SEEDS = range(9012, 9020)
BATCH = 5120
NUM_STEPS = 2000
SCHEDULE = "log"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=RUN_DIR / "samples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    return parser


def save_stage(path, arrays, metadata):
    json_path = path.with_suffix(".json")
    if path.exists() or json_path.exists():
        raise FileExistsError(f"output already exists: {path} or {json_path}")
    write_start = time.perf_counter()
    np.savez_compressed(path, **arrays)
    metadata["npz_write_seconds"] = time.perf_counter() - write_start
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    parent = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(parent, allow_pickle=False) as archive:
        phi = np.asarray(archive["phi_gen"], dtype=np.float32)
    index = np.arange(len(phi), dtype=np.int64)

    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    hparams = checkpoint_data["hyper_parameters"]
    epoch = int(checkpoint_data["epoch"]) + 1
    global_step = checkpoint_data.get("global_step")
    sigma = float(hparams["sigma"])
    A_res = float(hparams["norm_max"])
    A_cond = float(hparams["cond_norm_max"])
    del checkpoint_data

    device = torch.device(args.device)
    score_model = NCSNppCond2D(
        functools.partial(marginal_prob_std, sigma=sigma),
        in_channels=int(hparams.get("in_channels", 2)),
    )
    model = ConditionalDiffusionModel.load_from_checkpoint(
        str(checkpoint),
        score_model=score_model,
        map_location=device,
        weights_only=False,
    ).to(device)
    model.score_model = model.score_model.to(memory_format=torch.channels_last)
    model.score_model = torch.compile(
        model.score_model, mode=args.compile_mode
    )
    model.eval()

    print(
        f"checkpoint={checkpoint}\ninput={parent}\n"
        f"samples={len(phi)}, batch={BATCH}, compile={args.compile_mode}"
    )
    with torch.inference_mode(), model.ema.average_parameters():
        for absolute_time, seed in zip(TIMES, SEEDS):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            dphi = np.empty_like(phi)
            phi_next = np.empty_like(phi)
            zero_mode_raw = np.empty(len(phi), dtype=np.float32)

            torch.cuda.synchronize(device)
            sample_start = time.perf_counter()
            for start in range(0, len(phi), BATCH):
                stop = min(start + BATCH, len(phi))
                condition = torch.from_numpy(
                    phi[start:stop] / np.float32(A_cond)
                )[:, None].to(device)
                generated = model.sample(
                    condition,
                    num_steps=NUM_STEPS,
                    schedule=SCHEDULE,
                    use_ema=False,
                )[:, 0].float().cpu().numpy()
                generated *= np.float32(A_res)
                dphi[start:stop] = generated
                phi_next[start:stop] = phi[start:stop] + generated
                zero_mode_raw[start:stop] = generated.mean(axis=(1, 2))
            torch.cuda.synchronize(device)
            sample_seconds = time.perf_counter() - sample_start

            output = output_dir / (
                f"chain_ep{epoch}_t{absolute_time}_via_iterated_t10dm5120_"
                f"steps{NUM_STEPS}_{SCHEDULE}_seed{seed}.npz"
            )
            metadata = {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "global_step": global_step,
                "parent": str(parent),
                "absolute_time": absolute_time,
                "horizon": int(hparams["horizon"]),
                "n_samples": len(phi),
                "L": int(hparams["L"]),
                "kappa": hparams.get("kappa"),
                "lam": hparams.get("lam"),
                "sigma": sigma,
                "A_res": A_res,
                "A_cond": A_cond,
                "num_steps": NUM_STEPS,
                "schedule": SCHEDULE,
                "batch": BATCH,
                "seed": seed,
                "compile_mode": args.compile_mode,
                "projected": False,
                "sampling_seconds": sample_seconds,
            }
            save_stage(
                output,
                {
                    "phi_gen": phi_next,
                    "dphi": dphi,
                    "zero_mode_raw": zero_mode_raw,
                    "cond_indices": index,
                    "n_per_cond": np.int64(1),
                },
                metadata,
            )
            print(
                f"t={absolute_time}: {sample_seconds:.2f} s, "
                f"{len(phi) / sample_seconds:.1f} samples/s, "
                f"max |mean(dphi)|={np.abs(zero_mode_raw).max():.3e}\n"
                f"saved {output}",
                flush=True,
            )
            phi = phi_next
            parent = output.resolve()


if __name__ == "__main__":
    main()
