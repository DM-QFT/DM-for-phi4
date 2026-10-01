"""Iterate a trained conditional Model B kernel along a saved field chain."""


import argparse
import functools
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


MODEL_B_DIR = Path(__file__).resolve().parents[1]
DM_DIR = MODEL_B_DIR.parent
if str(DM_DIR) not in sys.path:
    sys.path.insert(0, str(DM_DIR))
if str(MODEL_B_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_B_DIR))

from diffusion_lightning import marginal_prob_std

from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D


RUN_DIR = MODEL_B_DIR / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip"
DEFAULT_CHECKPOINT = RUN_DIR / "models/epoch=10000.ckpt"
DEFAULT_INPUT = (
    RUN_DIR
    / "samples/hmc_chain_ep10000_N5120_seed20260901/hmc_ep10000_t0_N5120.npz"
)
# 续算由 sample/extend_chains.py 覆写；这里的默认只是从 t=0 起跑一小段
TIMES = range(10, 101, 10)
SEEDS = range(20260901, 20260911)
BATCH = 5120
NUM_STEPS = 2000
SCHEDULE = "log"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "samples/chain.h5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--n", type=int, help="use n initial fields at equal stride")
    return parser


def create_chain(path, n_steps, n, lattice, cond_indices, attrs):
    """整条链一个文件，时间是第一维；布局 [time, ensemble, x, y]。"""
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.create_dataset("phi_gen", (n_steps, n, lattice, lattice), dtype="f4",
                          chunks=(1, n, lattice, lattice))
    handle.create_dataset("zero_mode_raw", (n_steps, n), dtype="f4")
    handle.create_dataset("times", (n_steps,), dtype="f8")
    handle.create_dataset("seeds", (n_steps,), dtype="i8")
    handle.create_dataset("sampling_seconds", (n_steps,), dtype="f8")
    handle["cond_indices"] = cond_indices
    handle.attrs.update(attrs)
    handle.attrs["layout"] = "phi_gen[time, ensemble, x, y]"
    return handle


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    parent = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()

    with np.load(parent, allow_pickle=False) as archive:
        phi = np.asarray(archive["phi_gen"], dtype=np.float32)
    # 初态文件不是随机排列（HMC 前 512 条的 G_1 均值 5.6，后 512 条 7.4），子集按等间隔取
    stride = 1 if args.n is None else len(phi) // args.n
    index = np.arange(0, stride * (args.n or len(phi)), stride, dtype=np.int64)
    phi = phi[index]

    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    hparams = checkpoint_data["hyper_parameters"]
    epoch = int(checkpoint_data["epoch"]) + 1
    global_step = checkpoint_data.get("global_step")
    sigma = float(hparams["sigma"])
    A_res = float(hparams["norm_max"])
    A_cond = float(hparams["cond_norm_max"])
    horizon = float(hparams["horizon"])
    del checkpoint_data

    device = torch.device(args.device)
    score_model = NCSNppCond2D(
        functools.partial(marginal_prob_std, sigma=sigma),
        in_channels=int(hparams.get("in_channels", 2)),
        channels=list(hparams.get("channels") or (16, 32, 64, 128)),
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
    chain = create_chain(
        output, len(TIMES), len(phi), int(phi.shape[-1]), index,
        {
            "checkpoint": str(checkpoint), "epoch": epoch,
            "global_step": global_step, "parent": str(parent),
            "horizon": horizon, "n_samples": len(phi),
            "L": int(phi.shape[-1]), "checkpoint_training_L": int(hparams["L"]),
            "kappa": hparams.get("kappa"), "lam": hparams.get("lam"),
            "sigma": sigma, "A_res": A_res, "A_cond": A_cond,
            "num_steps": NUM_STEPS, "schedule": SCHEDULE,
            "batch": min(BATCH, len(phi)), "n_per_cond": 1,
            "compile_mode": args.compile_mode, "projected": True,
        },
    )
    with torch.inference_mode(), model.ema.average_parameters():
        for step, (absolute_time, seed) in enumerate(zip(TIMES, SEEDS)):
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

            chain["phi_gen"][step] = phi_next
            chain["zero_mode_raw"][step] = zero_mode_raw
            chain["times"][step] = absolute_time
            chain["seeds"][step] = seed
            chain["sampling_seconds"][step] = sample_seconds
            chain.flush()
            print(
                f"t={absolute_time}: {sample_seconds:.2f} s, "
                f"{len(phi) / sample_seconds:.1f} samples/s, "
                f"max |mean(dphi)|={np.abs(zero_mode_raw).max():.3e}",
                flush=True,
            )
            phi = phi_next
    chain.close()
    print(f"saved {output}")


if __name__ == "__main__":
    main()
