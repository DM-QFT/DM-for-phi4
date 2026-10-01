"""Train the conditional Model B diffusion model on every available pair."""

import argparse
import functools
import json
import math
import os
import re
import sys
import time

MODEL_B_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DM_DIR = os.path.dirname(MODEL_B_DIR)
sys.path.insert(0, DM_DIR)
sys.path.insert(0, MODEL_B_DIR)

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch_ema import ExponentialMovingAverage

from diffusion_lightning import marginal_prob_std
from cond_lightning import (
    ConditionalDiffusionModel, ModelBPairDataModule, NCSNppCond2D,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def canonical_path(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def default_run_dir(args):
    L_tag = (
        str(args.L) if len(args.Ls) == 1
        else "multi" + "-".join(map(str, args.Ls))
    )
    return canonical_path(os.path.join(
        MODEL_B_DIR, "runs",
        f"modelBcond_L{L_tag}_k{args.kappa}_l{args.lam}"
        f"_t{args.horizon:g}_ncsnpp{args.output_suffix}",
    ))


def validate_args(args, parser):
    if any(L <= 0 or L % 8 for L in args.Ls):
        parser.error("every training L must be a positive multiple of 8")
    for name in ("kappa", "sigma", "lr"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.sigma <= 1:
        parser.error("--sigma must be greater than 1")
    if not np.isfinite(args.lam) or args.lam < 0:
        parser.error("--lam must be finite and non-negative")
    if (not np.isfinite(args.zero_mode_loss_weight)
            or args.zero_mode_loss_weight < 0):
        parser.error("--zero_mode_loss_weight must be finite and non-negative")
    for name in ("horizon", "batch_size", "epochs", "num_ckpts"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0 <= args.ema_start < args.epochs:
        parser.error("--ema_start must satisfy 0 <= ema_start < epochs")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        parser.error("--device must be cuda or cuda:N")
    if (len(args.output_suffix) > 64
            or not re.fullmatch(r"[A-Za-z0-9_.-]*", args.output_suffix)):
        parser.error("invalid --output_suffix")
    if args.cuda_graph and (args.L_list or args.ckpt_path):
        parser.error("--cuda_graph only supports single-L training from scratch")


def checkpoint_run_dir(checkpoint_path):
    checkpoint_path = canonical_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    models_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(models_dir) != "models":
        raise ValueError("resume checkpoint must be inside a models directory")
    return checkpoint_path, os.path.dirname(models_dir)


def validate_resume(checkpoint_path, args, data_module):
    """Check only the normalization/model facts needed for a safe resume."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError(f"cannot load checkpoint: {exc}") from exc
    hparams = checkpoint.get("hyper_parameters", {})
    if hparams.get("L") != args.L:
        raise ValueError(
            f"checkpoint L={hparams.get('L')} does not match L={args.L}"
        )
    expected_scales = {
        "norm_min": -data_module.A_res,
        "norm_max": data_module.A_res,
        "cond_norm_min": -data_module.A_cond,
        "cond_norm_max": data_module.A_cond,
    }
    for key, expected in expected_scales.items():
        try:
            matches = np.isclose(
                float(hparams[key]), expected, rtol=1e-12, atol=1e-12
            )
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"checkpoint {key}={hparams.get(key)!r} does not match "
                f"the full-data scale {expected:.17g}"
            )
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError("checkpoint has no integer epoch")
    completed_epochs = epoch + 1
    if args.epochs <= completed_epochs:
        raise ValueError(
            f"--epochs must exceed the {completed_epochs} completed epochs"
        )
    return completed_epochs


def write_run_metadata(output_dir, args, data_module):
    metadata = {
        "training_data_scope": "all_data",
        "normalization_scope": "all_data",
        "validation": "none",
        "A_res": data_module.A_res,
        "A_cond": data_module.A_cond,
        "num_pairs": data_module.num_pairs,
        "num_pairs_by_L": data_module.num_pairs_by_L,
        "L": args.L,
        "training_Ls": args.Ls,
        "kappa": args.kappa,
        "lam": args.lam,
        "horizon": args.horizon,
        "sigma": args.sigma,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "ema_decay": args.ema_decay,
        "init_ckpt": args.init_ckpt,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "cond_path": args.cond_path,
        "target_path": args.target_path,
    }
    with open(os.path.join(output_dir, "meta.json"), "x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def fit_cuda_graph(model, data_module, args, output_dir):
    """整步（取批、加噪、损失、backward、Adam、EMA）捕获成一张 CUDA graph 重放。"""
    dev = data_module.device
    model.to(dev)
    model.ema.to(dev)
    model.train()
    x_all = data_module.data_on_gpu
    N, B = len(x_all), args.batch_size
    steps_per_epoch = N // B
    total_steps = steps_per_epoch * args.epochs
    params = list(model.score_model.parameters())  # 与 model.ema.shadow_params 同序
    shadow = model.ema.shadow_params
    lr_t = torch.tensor(args.lr, device=dev)
    opt = torch.optim.Adam(params, lr=lr_t, fused=True, capturable=True)
    lr_min, ema_start = model.lr_min, args.ema_start * steps_per_epoch
    decay_t = torch.tensor(model.ema.decay, device=dev)
    # lr 与 EMA 预热要在图内算，所以计数器必须是设备张量
    n_t = torch.zeros((), device=dev)
    m_t = torch.zeros((), device=dev)
    k_t = torch.zeros((), dtype=torch.long, device=dev)
    perm = torch.randperm(N, device=dev)
    base = torch.arange(B, device=dev)
    loss_acc = torch.zeros((), device=dev)
    # loss_fn 的 Z2 翻转和加噪都是设备随机数，每次重放都新抽

    def step():
        x = x_all[perm[base + k_t * B]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.loss_fn(x)
        loss.backward()
        lr_t.copy_(lr_min + 0.5 * (args.lr - lr_min)
                   * (1 + torch.cos(math.pi * n_t / total_steps)))
        opt.step()
        # torch_ema 的预热 decay=min(decay,(1+m)/(10+m))；ema_start 之前 w=0 不更新
        on = (n_t >= ema_start).float()
        m_t.add_(on)
        w = (1 - torch.minimum(decay_t, (1 + m_t) / (10 + m_t))) * on
        torch._foreach_mul_(shadow, 1 - w)
        torch._foreach_add_(shadow, torch._foreach_mul(params, w))
        n_t.add_(1)
        k_t.add_(1)
        loss_acc.add_(loss.detach())

    # whole-network capture：side stream 预热 3 步（含编译和 cudnn autotune），捕获前 grad 置 None
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        step()
    # 预热的 3 步真实更新保留（8M 步里可忽略），计数器归零
    for t in (n_t, m_t, k_t, loss_acc):
        t.zero_()

    models_dir = os.path.join(output_dir, "models")
    save_epochs = Checkpoints(models_dir, args.epochs, args.num_ckpts).save_epochs
    print(f"CUDA graph: {steps_per_epoch} steps/epoch, {total_steps} steps", flush=True)
    start = time.perf_counter()
    with open(os.path.join(output_dir, "loss.tsv"), "w") as f:
        f.write("epoch\tlr\tloss\n")
        for epoch in range(1, args.epochs + 1):
            perm.copy_(torch.randperm(N, device=dev))
            k_t.zero_()
            for _ in range(steps_per_epoch):
                graph.replay()
            f.write(f"{epoch}\t{lr_t.item():.4e}\t{loss_acc.item() / steps_per_epoch:.6f}\n")
            f.flush()
            loss_acc.zero_()
            if epoch in save_epochs:
                # 与 Lightning 同键，sample 端 load_from_checkpoint / on_load_checkpoint 不变
                model.ema.num_updates = int(m_t.item())
                torch.save({
                    "state_dict": model.state_dict(),
                    "hyper_parameters": dict(model.hparams),
                    "ema_state_dict": model.ema.state_dict(),
                    "epoch": epoch - 1, "global_step": epoch * steps_per_epoch,
                    "pytorch-lightning_version": pl.__version__,
                }, os.path.join(models_dir, f"epoch={epoch:04d}.ckpt"))
            if epoch % max(args.epochs // 20, 1) == 0:
                print(f"epoch {epoch}/{args.epochs}: {time.perf_counter() - start:.0f} s",
                      flush=True)


class Checkpoints(Callback):
    """Save only the requested log-spaced checkpoint history."""

    def __init__(self, dirpath, max_epochs, num_checkpoints):
        self.dirpath = dirpath
        epochs = np.geomspace(1, max_epochs, num=num_checkpoints).astype(int)
        self.save_epochs = set(np.unique(epochs).tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch in self.save_epochs:
            path = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            if os.path.exists(path):
                raise FileExistsError(f"refusing to overwrite {path}")
            trainer.save_checkpoint(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--L_list", help="comma-separated mixed training sizes")
    parser.add_argument("--kappa", type=float, default=0.28)
    parser.add_argument("--lam", type=float, default=0.022)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=250.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, help="cosine-anneal to this; default constant")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--init_ckpt", help="start from this checkpoint's EMA weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--ema_start", type=int, default=0)
    parser.add_argument("--zero_mode_loss_weight", type=float, default=0.0)
    parser.add_argument("--no_z2_aug", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cond_path")
    parser.add_argument("--target_path")
    parser.add_argument("--cond_paths", nargs="+")
    parser.add_argument("--target_paths", nargs="+")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--num_ckpts", type=int, default=100)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument(
        "--compile_mode", default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--cuda_graph", action="store_true",
                        help="replay the whole training step as one CUDA graph")
    parser.add_argument("--channels", default="16,32,64,128",
                        help="NCSNpp widths per level, comma-separated")
    args = parser.parse_args()
    if args.L_list:
        try:
            args.Ls = sorted({int(value) for value in args.L_list.split(",")})
        except ValueError:
            parser.error("--L_list must be comma-separated integers")
        if len(args.Ls) < 2:
            parser.error("--L_list must contain at least two sizes")
        args.L = max(args.Ls)
        if args.cond_path or args.target_path:
            parser.error("use --cond_paths and --target_paths with --L_list")
        if (not args.cond_paths or not args.target_paths
                or len(args.cond_paths) != len(args.Ls)
                or len(args.target_paths) != len(args.Ls)):
            parser.error("multi-L training needs one condition and target per L")
    else:
        args.Ls = [args.L]
        if args.cond_paths or args.target_paths:
            parser.error("--cond_paths and --target_paths require --L_list")
    validate_args(args, parser)
    return args


def main():
    args = parse_args()
    if len(args.Ls) == 1:
        if args.cond_path is None:
            args.cond_path = os.path.join(
                DM_DIR, "2Dphi4", "trainingdata",
                f"cfgs_wolff_fahmc_k={args.kappa}_l={args.lam}_{args.L}^2.jld2",
            )
        if args.target_path is None:
            args.target_path = os.path.join(
                MODEL_B_DIR, "runs",
                f"L{args.L}_Ne10240_dt0p002_hmc_t1_t10_t100",
                f"modelB_L{args.L}_Ne10240_hmc_kappa_{args.kappa}"
                f"_t{args.horizon:g}_dt0.002.jld2",
            )
        args.cond_path = canonical_path(args.cond_path)
        args.target_path = canonical_path(args.target_path)
    else:
        args.cond_path = [canonical_path(path) for path in args.cond_paths]
        args.target_path = [canonical_path(path) for path in args.target_paths]

    if args.ckpt_path:
        args.ckpt_path, output_dir = checkpoint_run_dir(args.ckpt_path)
    else:
        output_dir = default_run_dir(args)
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"output already exists: {output_dir}; use a new --output_suffix"
            )

    # 续训沿用 checkpoint 的归一化常数，新数据集的极值不同会让权重与输入尺度错位
    fixed_norm = None
    if args.init_ckpt:
        hp = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
        fixed_norm = (float(hp["norm_max"]), float(hp["cond_norm_max"]))

    data_module = ModelBPairDataModule(
        args.cond_path, args.target_path,
        batch_size=args.batch_size,
        device=args.device,
        expected_L=args.Ls,
        fixed_norm=fixed_norm,
    )
    data_module.setup()

    if args.ckpt_path:
        resume_epoch = validate_resume(args.ckpt_path, args, data_module)
        print(f"Resuming after epoch {resume_epoch}; all {data_module.num_pairs} "
              "pairs are used from the next epoch")
    else:
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=False)
        write_run_metadata(output_dir, args, data_module)

    marginal_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    channels = [int(c) for c in args.channels.split(",")]
    score_model = NCSNppCond2D(marginal_fn, in_channels=2, channels=channels)
    score_model = score_model.to(memory_format=torch.channels_last)
    # 整步自己捕获时不能再用 reduce-overhead（inductor 内部的 CUDA graph 不能嵌套）
    score_model = torch.compile(
        score_model, mode="default" if args.cuda_graph else args.compile_mode,
        dynamic=False,
    )
    model = ConditionalDiffusionModel(
        score_model, sigma=args.sigma, lr=args.lr, L=args.L,
        ema_start_epoch=args.ema_start,
        norm_min=-data_module.A_res, norm_max=data_module.A_res,
        cond_norm_min=-data_module.A_cond, cond_norm_max=data_module.A_cond,
        zero_mode_loss_weight=args.zero_mode_loss_weight,
        kappa=args.kappa, lam=args.lam, horizon=args.horizon,
        cond_data_path=args.cond_path, target_data_path=args.target_path,
        run_dir=output_dir, in_channels=2, z2_aug=not args.no_z2_aug,
        ema_decay=args.ema_decay, lr_min=args.lr_min, channels=channels,
    )
    if args.init_ckpt:
        # 续训：参数 ← 旧 run 的 EMA 权重，EMA 重建，优化器和 epoch 计数重新开始
        checkpoint = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        shadow = checkpoint["ema_state_dict"]["shadow_params"]
        params = list(model.score_model.parameters())
        assert len(params) == len(shadow), (len(params), len(shadow))
        with torch.no_grad():
            for p, s in zip(params, shadow):
                p.copy_(s)
        model.ema = ExponentialMovingAverage(params, decay=args.ema_decay)
        model._ema_parameters = tuple(params)

    models_dir = os.path.join(output_dir, "models")
    checkpoints = Checkpoints(
        models_dir, args.epochs, args.num_ckpts
    )
    print(f"Output directory: {output_dir}/")
    print(f"Log-scale checkpoints: {len(checkpoints.save_epochs)}")
    if args.cuda_graph:
        fit_cuda_graph(model, data_module, args, output_dir)
    else:
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        trainer = pl.Trainer(
            max_epochs=args.epochs, accelerator="gpu", devices=[device_id],
            callbacks=[checkpoints],
            default_root_dir=output_dir, precision="bf16-mixed",
            log_every_n_steps=10, enable_progress_bar=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("Training complete!")


if __name__ == "__main__":
    main()
