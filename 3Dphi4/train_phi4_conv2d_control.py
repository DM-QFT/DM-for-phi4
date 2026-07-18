"""Train the Conv2d-only 3D control for the 4D convolution diagnosis.

The score is predicted directly.  This entry point intentionally has no
Gaussian baseline, residual/full-score wrapper, or projected zero-mode loss.
Its network topology is the dimensional-reduction counterpart of the current
4D network.  Channel widths, normalization, and the final convolution bias are
explicit so the Conv2d-fold implementation can also reproduce the effective
configuration of historical native-Conv3d runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve()
ROOT3D = HERE.parent
REPO = ROOT3D.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT3D))

from diffusion_lightning import DiffusionModel, marginal_prob_std  # noqa: E402
from networks_3d_conv2d import (  # noqa: E402
    NCSNpp3DConv2d,
    convert_conv2d_wrappers_to_native,
)


torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True


class LogScaleCheckpoint(Callback):
    def __init__(self, directory: Path, max_epochs: int, count: int = 100):
        super().__init__()
        self.directory = directory
        epochs = np.unique(np.geomspace(1, max_epochs, num=count).astype(int))
        self.save_epochs = set(epochs.tolist())

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if epoch in self.save_epochs:
            trainer.save_checkpoint(self.directory / f"epoch={epoch:04d}.ckpt")


class PrintLoss(Callback):
    def __init__(self, every: int = 10):
        self.every = every
        self.start = time.time()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if epoch == 1 or epoch % self.every == 0:
            value = trainer.callback_metrics.get("train_loss_epoch")
            print(
                f"epoch {epoch:5d}  loss {float(value):12.3f}  "
                f"elapsed {time.time() - self.start:9.1f}s",
                flush=True,
            )


def _scalar(handle, *names: str, default=None):
    for name in names:
        if name in handle:
            return handle[name][()]
    return default


def parse_channels(value: str) -> tuple[int, int, int, int]:
    try:
        channels = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"channels must be four comma-separated integers, got {value!r}"
        ) from error
    if len(channels) != 4 or any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError(
            f"channels must contain four positive widths, got {value!r}"
        )
    return channels


def load_3d_jld2(path: Path) -> tuple[np.ndarray, float, float, int]:
    """Load JLD2/HDF5 and return batch-first ``(N,L,L,L)`` configurations."""
    with h5py.File(path, "r") as handle:
        if "cfgs" not in handle or not isinstance(handle["cfgs"], h5py.Dataset):
            raise ValueError(f"{path} must contain one dense cfgs dataset")
        cfgs = np.asarray(handle["cfgs"], dtype=np.float32)
        kappa = float(_scalar(handle, "kappa", "κ", default=np.nan))
        lam = float(_scalar(handle, "lambda", "λ", default=np.nan))
        saved_L = int(_scalar(handle, "N", default=-1))

    if cfgs.ndim != 4:
        raise ValueError(f"expected a 4D cfg array, got {cfgs.shape}")
    # HDF5 readers see a Julia (L,L,L,Ncfg) array as (Ncfg,L,L,L).
    if cfgs.shape[1] == cfgs.shape[2] == cfgs.shape[3]:
        batch_first = cfgs
        L = int(cfgs.shape[1])
    elif cfgs.shape[0] == cfgs.shape[1] == cfgs.shape[2]:
        batch_first = np.moveaxis(cfgs, -1, 0)
        L = int(cfgs.shape[0])
    else:
        raise ValueError(f"cannot identify sample axis in cfgs shape {cfgs.shape}")
    if saved_L > 0 and saved_L != L:
        raise ValueError(f"metadata N={saved_L} disagrees with cfg shape {cfgs.shape}")
    if not np.isfinite(batch_first).all():
        raise ValueError("training configurations contain non-finite values")
    if not np.isfinite(kappa) or not np.isfinite(lam):
        raise ValueError("missing finite kappa/lambda metadata")
    return np.ascontiguousarray(batch_first), kappa, lam, L


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        type=Path,
        default=ROOT3D
        / "trainingdata/cfgs_wolff_fahmc_k=0.1923_l=0.9_8^3.jld2",
    )
    parser.add_argument("--sigma", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--ema_start", type=int, default=0)
    parser.add_argument("--num_ckpts", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        choices=("conv2d_only", "native_conv3d"),
        default="conv2d_only",
        help="requested Conv2d-only model or its exactly weight-matched control",
    )
    parser.add_argument(
        "--channels",
        type=parse_channels,
        default=(4, 8, 16, 32),
        metavar="C1,C2,C3,C4",
    )
    parser.add_argument(
        "--normalization",
        choices=("symmetric_maxabs", "minmax"),
        default="symmetric_maxabs",
    )
    parser.add_argument(
        "--final_bias",
        action="store_true",
        help="enable the final convolution bias (historical NCSNpp3D setting)",
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead"),
        default="default",
    )
    parser.add_argument("--output_suffix", default="")
    args = parser.parse_args()

    if args.sigma <= 1.0:
        parser.error("--sigma must be larger than one")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch_size and --epochs must be positive")
    args.data_path = args.data_path.resolve()
    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)

    pl.seed_everything(0)
    cfgs, kappa, lam, L = load_3d_jld2(args.data_path)
    if args.normalization == "symmetric_maxabs":
        norm_scale = float(np.abs(cfgs).max())
        if not np.isfinite(norm_scale) or norm_scale <= 0.0:
            raise ValueError(f"invalid symmetric normalization scale {norm_scale}")
        norm_min, norm_max = -norm_scale, norm_scale
        normalized = cfgs / norm_scale
        normalization_description = (
            f"symmetric norm=[{norm_min:.8g}, {norm_max:.8g}]"
        )
    else:
        norm_min, norm_max = float(cfgs.min()), float(cfgs.max())
        if not norm_max > norm_min:
            raise ValueError(f"invalid minmax range [{norm_min}, {norm_max}]")
        norm_scale = None
        normalized = 2.0 * (cfgs - norm_min) / (norm_max - norm_min) - 1.0
        normalization_description = f"minmax norm=[{norm_min:.8g}, {norm_max:.8g}]"
    x = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(1)
    expected_shape = (len(cfgs), 1, L, L, L)
    if tuple(x.shape) != expected_shape or not x.is_contiguous():
        raise RuntimeError(
            f"bad input layout {tuple(x.shape)}, expected {expected_shape}, "
            f"contiguous={x.is_contiguous()}"
        )
    print(
        f"data: {tuple(x.shape)}, kappa={kappa:g}, lambda={lam:g}, "
        f"{normalization_description}",
        flush=True,
    )

    device = torch.device(args.device)
    dataset = TensorDataset(
        x.to(device), torch.zeros(len(x), device=device, dtype=torch.float32)
    )
    loader_generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    std_fn = lambda t: marginal_prob_std(t, sigma=args.sigma)
    score_model = NCSNpp3DConv2d(
        std_fn,
        channels=args.channels,
        final_bias=args.final_bias,
    )
    if args.backend == "native_conv3d":
        score_model = convert_conv2d_wrappers_to_native(score_model)
    parameter_count = sum(p.numel() for p in score_model.parameters())
    if parameter_count <= 0:
        raise RuntimeError("score model has no parameters")
    if not args.no_compile:
        compile_kwargs = (
            {} if args.compile_mode == "default" else {"mode": args.compile_mode}
        )
        score_model = torch.compile(score_model, **compile_kwargs)

    # Direct full score only: no FullScore wrapper/baseline and no k0 auxiliary.
    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=L,
        ema_start_epoch=args.ema_start,
        norm_min=norm_min,
        norm_max=norm_max,
        spatial_dims=3,
        zero_mode_loss_weight=0.0,
    )

    channel_tag = "-".join(str(channel) for channel in args.channels)
    norm_tag = "symnorm" if args.normalization == "symmetric_maxabs" else "minmax"
    bias_tag = "bias" if args.final_bias else "nobias"
    output_name = (
        f"phi4_3d_L{L}_k{kappa:g}_l{lam:g}_ncsnpp_"
        f"{args.backend}_sigma{args.sigma:g}_ch{channel_tag}_"
        f"{norm_tag}_{bias_tag}_directscore_k0w0{args.output_suffix}"
    )
    output_dir = ROOT3D / "runs" / output_name
    launcher_files = {"train.log", "train.pid"}
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    conflicts = [p for p in existing if p.name not in launcher_files]
    if conflicts:
        raise FileExistsError(
            f"refusing to mix runs in {output_dir}; found "
            f"{[p.name for p in conflicts]}"
        )
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        **vars(args),
        "data_path": str(args.data_path),
        "L": L,
        "k": kappa,
        "l": lam,
        "n_samples": len(cfgs),
        "norm_min": norm_min,
        "norm_max": norm_max,
        "norm_scale": norm_scale,
        "normalization": args.normalization,
        "score_parameterization": "direct_full_score",
        "gaussian_baseline": False,
        "zero_mode_loss_weight": 0.0,
        "convolution_backend": args.backend,
        "channels": list(args.channels),
        "final_convolution_bias": args.final_bias,
        "param_count": parameter_count,
    }
    with (output_dir / "training_config.yaml").open("w") as handle:
        yaml.safe_dump(configuration, handle, sort_keys=False)

    checkpoint = LogScaleCheckpoint(models_dir, args.epochs, args.num_ckpts)
    device_id = device.index if device.index is not None else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint, PrintLoss()],
        logger=CSVLogger(output_dir, name="logs"),
        default_root_dir=output_dir,
        precision="bf16-mixed",
        enable_progress_bar=False,
        log_every_n_steps=50,
    )
    print(
        f"model: {args.backend} direct score, parameters={parameter_count:,}; "
        f"output={output_dir}",
        flush=True,
    )
    # Native conversion constructs temporary parameters before copying the
    # mapped weights.  Reset training RNG so both backends see identical t/z.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    trainer.fit(model, loader)
    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
