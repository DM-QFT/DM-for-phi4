"""
反对称分数 (非守恒分数) 随训练 epoch 的变化。

对一个 run 的多个 log-scale checkpoint, 在固定的几个 t 上计算
    rho(t) = ||J - J^T||_F / (2 ||J||_F)   (非守恒分数, 见 curl_fraction.py),
画出 rho 随 epoch 的演化曲线 —— 看非守恒残留是否随训练收缩。

关键设计:
  - HMC 数据只加载一次, 所有 epoch 共用同一批组态。
  - 每个 epoch 前重置 RNG, 使加噪 z 和 Hutchinson 探针 v 在所有 epoch 间完全相同
    (配对比较, 消除 epoch 间因随机数不同带来的虚假抖动)。

复用 curl_fraction.py 的 antisym_fraction / load_model / load_hmc, 不修改该文件。

用法:
    python curl_fraction_vs_epoch.py \
        --run_dir runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640 \
        --data_path trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_128^2.jld2 \
        --t_list 0.01,0.1,1.0 --max_ckpts 25 \
        --num_samples 256 --batch_size 256 --n_hutch 8 \
        --network ncsnpp --L 128 --device cuda:0 --seed 0
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))   # repo root (DM): diffusion_lightning, networks, phi4_action
sys.path.insert(0, str(_HERE.parent))       # 同目录: curl_fraction

import os
import re
import glob
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curl_fraction import antisym_fraction, load_model, load_hmc


def discover_ckpts(run_dir, epochs_arg, max_ckpts):
    """返回按 epoch 升序的 [(epoch:int, path), ...]。"""
    items = []
    for p in glob.glob(os.path.join(run_dir, "models", "epoch=*.ckpt")):
        m = re.search(r"epoch=(\d+)\.ckpt$", p)
        if m:
            items.append((int(m.group(1)), p))
    items.sort()
    if not items:
        raise FileNotFoundError(f"no epoch=*.ckpt under {run_dir}/models")

    if epochs_arg:                                   # 显式指定 epoch 列表
        want = {int(x) for x in epochs_arg.split(",")}
        items = [it for it in items if it[0] in want]
    elif max_ckpts and len(items) > max_ckpts:       # 几何抽样, 兼顾早晚期
        idx = np.unique(np.round(
            np.geomspace(1, len(items), max_ckpts)).astype(int) - 1)
        items = [items[i] for i in idx]
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--t_list", type=str, default="0.01,0.1,1.0")
    p.add_argument("--epochs", type=str, default=None,
                   help="逗号分隔的 epoch 列表; 不给则用全部 (受 --max_ckpts 限制)")
    p.add_argument("--max_ckpts", type=int, default=25,
                   help="未指定 --epochs 时, 几何抽样最多这么多 checkpoint (0=全部)")
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_hutch", type=int, default=8)
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    t_list = [float(s) for s in args.t_list.split(",")]
    ckpts = discover_ckpts(args.run_dir, args.epochs,
                           args.max_ckpts if args.max_ckpts > 0 else None)
    print(f"{len(ckpts)} checkpoints: epochs {[e for e, _ in ckpts]}")

    torch.manual_seed(args.seed)
    x0 = None
    rows = []   # (epoch, t, rho_mean, rho_sem, rho2_mean, n)

    for ei, (ep, ckpt) in enumerate(ckpts):
        a = argparse.Namespace(checkpoint=ckpt, network=args.network,
                               L=args.L, device=args.device)
        model, hp = load_model(a)
        if x0 is None:                               # 只加载一次 HMC 数据
            x0 = load_hmc(args.data_path, hp["norm_min"], hp["norm_max"],
                          args.num_samples, args.device, args.seed)

        # 每个 epoch 用相同噪声+探针 -> 配对比较
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        with model.ema.average_parameters():
            line = []
            for t_val in t_list:
                rho2 = []
                for i in range(0, x0.shape[0], args.batch_size):
                    rho2.append(antisym_fraction(
                        model, x0[i:i + args.batch_size], t_val, args.n_hutch, gen))
                rho2 = torch.cat(rho2)
                rho = rho2.clamp_min(0).sqrt()
                n = rho.numel()
                sem = rho.std().item() / np.sqrt(n) if n > 1 else 0.0
                rows.append((ep, t_val, rho.mean().item(), sem, rho2.mean().item(), n))
                line.append(f"t={t_val:g}:{rho.mean().item():.4f}")
        print(f"[{ei+1}/{len(ckpts)}] epoch={ep:5d}  " + "  ".join(line))

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # 输出
    if args.output is None:
        args.output = os.path.join(args.run_dir, "data", "curl_fraction_vs_epoch")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    arr = np.array(rows, dtype=np.float64)
    np.savetxt(args.output + ".csv", arr,
               header="epoch t rho_mean rho_sem rho2_mean n", comments="", fmt="%.6e")
    print(f"\nSaved -> {args.output}.csv")

    # 画 rho vs epoch, 每个 t 一条线
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t_val in t_list:
        sub = arr[np.isclose(arr[:, 1], t_val)]
        sub = sub[np.argsort(sub[:, 0])]
        ax.errorbar(sub[:, 0], sub[:, 2], yerr=sub[:, 3],
                    marker="o", ms=4, capsize=2, label=f"t={t_val:g}")
    ax.set_xscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel(r"$\rho = \|J-J^\top\|_F / (2\|J\|_F)$  (非守恒分数)")
    ax.set_title(os.path.basename(os.path.normpath(args.run_dir)))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output + ".png", dpi=150)
    print(f"Saved -> {args.output}.png")


if __name__ == "__main__":
    main()
