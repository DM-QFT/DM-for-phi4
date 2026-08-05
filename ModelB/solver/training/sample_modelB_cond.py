"""Generate conditional Model B samples from a diffusion checkpoint.

Examples
--------
python training/sample_modelB_cond.py --run_dir runs/my_run --ep 200
python -m training.sample_modelB_cond --checkpoint runs/my_run/models/epoch=0200.ckpt
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch


MODEL_B_DIR = Path(__file__).resolve().parents[2]
DM_DIR = MODEL_B_DIR.parent
if str(DM_DIR) not in sys.path:
    sys.path.insert(0, str(DM_DIR))

from diffusion_lightning import marginal_prob_std

if __package__:
    from .cond_lightning import ConditionalDiffusionModel, NCSNppCond2D
else:
    from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D


_EPOCH_RE = re.compile(r"epoch=(\d+)\.ckpt")


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def validate_schedule(value: str) -> str:
    if value in {"linear", "quadratic", "cosine", "log"}:
        return value
    if value.startswith("power_"):
        try:
            if math.isfinite(float(value[6:])) and float(value[6:]) > 0:
                return value
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        "schedule must be linear, quadratic, cosine, log, or power_N"
    )


def checkpoint_epoch_from_name(path: str | Path) -> int | None:
    match = _EPOCH_RE.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


def select_checkpoint(run_dir: Path, epoch: int | None) -> tuple[Path, int]:
    """Select an exact epoch, or the greatest numeric epoch when omitted."""
    by_epoch: dict[int, list[Path]] = {}
    for path in (run_dir / "models").glob("epoch=*.ckpt"):
        number = checkpoint_epoch_from_name(path)
        if number is not None:
            by_epoch.setdefault(number, []).append(path.resolve())
    if not by_epoch:
        raise FileNotFoundError(f"no epoch=NNNN.ckpt files in {run_dir / 'models'}")

    epoch = max(by_epoch) if epoch is None else epoch
    matches = by_epoch.get(epoch, [])
    if not matches:
        raise FileNotFoundError(f"checkpoint epoch {epoch} not found in {run_dir}")
    if len(matches) != 1:
        raise RuntimeError(f"multiple checkpoints found for epoch {epoch}")
    return matches[0], epoch


def parse_indices(spec: str | None, n_total: int) -> np.ndarray:
    if spec is None:
        return np.arange(n_total, dtype=np.int64)
    if ":" in spec:
        start_text, stop_text = spec.split(":", 1)
        start = int(start_text) if start_text else 0
        stop = int(stop_text) if stop_text else n_total
        indices = np.arange(start, stop, dtype=np.int64)
    else:
        indices = np.asarray([int(item) for item in spec.split(",")], dtype=np.int64)
    if (indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_total)
            or np.unique(indices).size != indices.size):
        raise ValueError(f"invalid --cond_indices for {n_total} conditions: {spec!r}")
    return indices


def read_conditions(path: Path, spec: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Read all conditions, or only the requested HDF5 rows."""
    with h5py.File(path, "r") as handle:
        dataset = handle["cfgs"]
        indices = parse_indices(spec, dataset.shape[0])
        if spec is None:
            fields = dataset[:]
        else:
            order = np.argsort(indices)
            sorted_indices = indices[order]
            if np.all(np.diff(sorted_indices) == 1):
                selected = dataset[int(sorted_indices[0]):int(sorted_indices[-1]) + 1]
            else:
                selected = dataset[sorted_indices]
            fields = selected[np.argsort(order)]
    return np.asarray(fields, dtype=np.float32), indices


def _checkpoint_and_run(args: argparse.Namespace) -> tuple[Path, Path, int]:
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run_dir = (
            checkpoint.parent.parent
            if checkpoint.parent.name == "models"
            else checkpoint.parent
        )
        epoch = checkpoint_epoch_from_name(checkpoint)
        if epoch is None:
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            epoch = int(data["epoch"]) + 1
            del data
        if args.ep is not None and args.ep != epoch:
            raise ValueError(f"--ep={args.ep} does not match checkpoint epoch {epoch}")
        return checkpoint, run_dir, epoch

    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint, epoch = select_checkpoint(run_dir, args.ep)
    return checkpoint, run_dir, epoch


def _save_exclusive(
    npz_path: Path,
    json_path: Path,
    arrays: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Save ordinary NPZ/JSON files while avoiding accidental replacement."""
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"output already exists: {npz_path} or {json_path}")
    np.savez_compressed(npz_path, **arrays)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    npz_path.chmod(0o640)
    json_path.chmod(0o640)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Checkpoint file")
    source.add_argument("--run_dir", help="Run directory containing models/")
    parser.add_argument("--ep", type=_positive_int, help="Exact epoch; latest if omitted")
    parser.add_argument(
        "--cond_indices",
        help="Optional condition range/list, for example 0:256 or 3,17,42",
    )
    parser.add_argument("--cond_path", help="Optional alternative condition HDF5")
    parser.add_argument("--n_per_cond", type=_positive_int, default=1)
    parser.add_argument("--num_steps", type=_positive_int, default=2000)
    parser.add_argument("--schedule", type=validate_schedule, default="log")
    parser.add_argument("--batch", type=_positive_int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--output", help="Output NPZ path (default: run samples/)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint, run_dir, epoch = _checkpoint_and_run(args)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint_data["hyper_parameters"]
    checkpoint_L = int(hparams["L"])
    sigma = float(hparams["sigma"])
    A_res = float(hparams["norm_max"])
    A_cond = float(hparams["cond_norm_max"])
    cond_path = Path(
        args.cond_path or hparams["cond_data_path"]
    ).expanduser().resolve()
    target_value = hparams.get("target_data_path")
    if isinstance(target_value, (list, tuple)):
        target_path = [str(Path(path).expanduser().resolve()) for path in target_value]
    else:
        target_path = (
            str(Path(target_value).expanduser().resolve()) if target_value else None
        )
    global_step = checkpoint_data.get("global_step")
    del checkpoint_data

    phi0, cond_indices = read_conditions(cond_path, args.cond_indices)
    if phi0.ndim != 3 or phi0.shape[1] != phi0.shape[2]:
        raise ValueError(f"conditions must have shape (N,L,L), got {phi0.shape}")
    L = int(phi0.shape[1])
    n_cond = len(cond_indices)
    n_samples = n_cond * args.n_per_cond

    if args.output:
        npz_path = Path(args.output).expanduser().resolve()
        if npz_path.suffix != ".npz":
            npz_path = npz_path.with_suffix(".npz")
    else:
        selection = "all" if args.cond_indices is None else "subset"
        name = (
            f"cond_samples_ep{epoch:04d}_steps{args.num_steps}_{args.schedule}"
            f"_seed{args.seed}_{selection}_c{n_cond}_n{args.n_per_cond}.npz"
        )
        npz_path = run_dir / "samples" / name
    json_path = npz_path.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"output already exists: {npz_path} or {json_path}")

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
    model.score_model = torch.compile(model.score_model, mode=args.compile_mode)
    model.eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dphi = np.empty((n_samples, L, L), dtype=np.float32)
    phi_gen = np.empty_like(dphi)
    zero_mode_raw = np.empty(n_samples, dtype=np.float32)

    print(
        f"checkpoint={checkpoint}\nconditions={n_cond}, n_per_cond={args.n_per_cond}, "
        f"samples={n_samples}, compile={args.compile_mode}\noutput={npz_path}"
    )
    with torch.inference_mode():
        for start in range(0, n_samples, args.batch):
            stop = min(start + args.batch, n_samples)
            rows = np.arange(start, stop) // args.n_per_cond
            phi0_batch = phi0[rows]
            cond = torch.from_numpy(phi0_batch / np.float32(A_cond))
            cond = cond[:, None].to(device)
            generated = model.sample(
                cond, num_steps=args.num_steps, schedule=args.schedule
            )[:, 0].float().cpu().numpy()
            generated *= np.float32(A_res)
            zero_mode = generated.mean(axis=(1, 2))
            zero_mode_raw[start:stop] = zero_mode
            dphi[start:stop] = generated
            phi_gen[start:stop] = phi0_batch + generated

    metadata = {
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "global_step": global_step,
        "L": L,
        "checkpoint_training_L": checkpoint_L,
        "sigma": sigma,
        "A_res": A_res,
        "A_cond": A_cond,
        "kappa": hparams.get("kappa"),
        "lam": hparams.get("lam"),
        "horizon": hparams.get("horizon"),
        "source_path": str(cond_path),
        "cond_path": str(cond_path),
        "target_path": target_path,
        "n_cond": n_cond,
        "n_per_cond": args.n_per_cond,
        "num_steps": args.num_steps,
        "schedule": args.schedule,
        "seed": args.seed,
        "batch": args.batch,
        "compile_mode": args.compile_mode,
        "projected": True,
    }
    _save_exclusive(
        npz_path,
        json_path,
        {
            "phi_gen": phi_gen,
            "dphi": dphi,
            "cond_indices": cond_indices,
            "n_per_cond": np.int64(args.n_per_cond),
            "zero_mode_raw": zero_mode_raw,
        },
        metadata,
    )
    print(
        f"saved {phi_gen.shape}; raw max |mean(dphi)|="
        f"{np.abs(zero_mode_raw).max():.3e}"
    )


if __name__ == "__main__":
    main()
