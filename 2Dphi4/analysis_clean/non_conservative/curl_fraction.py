"""
Jacobian 反对称分数 (非守恒分数) 扫描 —— 衡量训练好的 score net 还剩多少非守恒成分。

对真score s = ∇log p_t, Jacobian J_ij = ∂s_i/∂x_j 必对称 (守恒/梯度场)。
网络直接回归向量场, 不保证对称。反对称部分 A = (J - J^T)/2 度量非守恒。
本脚本计算无量纲分数

    rho^2 = ||J - J^T||_F^2 / (4 ||J||_F^2) = ||A||_F^2 / ||J||_F^2  ∈ [0,1],

rho=0 完全守恒。它是论文 (arXiv:2510.01328) Fig.3 curl 在全部 d(d-1)/2 个分量上的
聚合 (高维版)。用 Hutchinson + 自动微分估计 (不显式构造 d×d 的 J):
    J^T v   一次反向 (VJP);  J v   double-backprop (纯反向, 对 LeakyReLU 也稳).

评估点: 对每个 t, 取 HMC 样本 x0 (归一化到 [-1,1]) 前向加噪到 p_t:
    x_t = x0 + std(t)·z,   std = marginal_prob_std(t),
即在真实数据流形扰动出的 p_t 上评估, 而非反向生成的样本。

用法:
    python curl_fraction.py --checkpoint runs/.../models/xx.ckpt \
        --data_path ../trainingdata/xxx.jld2 \
        --t_list 0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0 \
        --num_samples 512 --n_hutch 16 --device cuda:0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root (DM): diffusion_lightning, networks, phi4_action

import os
import re
import functools
import argparse
from pathlib import Path

import h5py
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks import ScoreNet, ScoreNetUNetPeriodic, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


@torch.enable_grad()
def antisym_fraction(model, x0, t_val, n_hutch, generator=None):
    """返回每个样本的 rho^2 = ||J - J^T||_F^2 / (4||J||_F^2) ∈ [0,1].

    x0: (B,1,L,L) 干净的归一化 HMC 组态;  t_val: 标量噪声层级.
    每个样本被加噪到 p_t 上 (x_t = x0 + std(t)·z) 后评估 score 的 Jacobian 反对称性.
    """
    device = x0.device
    B = x0.shape[0]
    t = torch.full((B,), float(t_val), device=device)

    # HMC 样本 + 相应噪声 -> p_t 上的点
    std = model.marginal_prob_std_fn(t)
    z = torch.randn(x0.shape, device=device, generator=generator)
    x = (x0 + model._bcast(std) * z).requires_grad_(True)

    red = tuple(range(1, x.dim()))                 # 对场自由度求和, 保留 batch
    num = x.new_zeros(B)                           # 累加 ||(J - J^T) v||^2
    den = x.new_zeros(B)                           # 累加 ||J v||^2
    for _ in range(n_hutch):
        v = torch.randn(x.shape, device=device, generator=generator)  # 高斯探针 Cov=I
        s = model(x, t)                            # score, 形状同 x
        # J^T v : 一次反向 (VJP)
        JTv = torch.autograd.grad(s, x, grad_outputs=v, retain_graph=True)[0]
        # J v : double-backprop —— g = J^T w 关于 w 线性, 对 w 求导得 J v
        w = torch.zeros_like(s, requires_grad=True)
        g = torch.autograd.grad(s, x, grad_outputs=w, create_graph=True)[0]
        Jv = torch.autograd.grad(g, w, grad_outputs=v)[0]
        diff = Jv - JTv                            # = (J - J^T) v
        num = num + (diff ** 2).sum(dim=red)
        den = den + (Jv ** 2).sum(dim=red)
    return (num / (4.0 * den)).detach()            # per-sample rho^2


def load_model(args):
    """照搬 sample_phi4.py 的加载流程, 返回 (model, hparams)。"""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    L = hparams.get("L", args.L)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"L={L}, sigma={sigma}, norm=[{norm_min:.4f}, {norm_max:.4f}]")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn)
    elif args.network == "scorenet":
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)
    if getattr(args, "random_init", False):
        # 未训练的随机权重 —— rho 的"随机天花板"参照 (~0.707)。
        # 仍用 checkpoint 的 sigma/L/norm 保证评估点一致, 只是不加载权重。
        model = DiffusionModel(score_model=score_model, sigma=sigma, L=L)
        print("** random_init: 使用未训练随机权重 (忽略 checkpoint 权重) **")
    else:
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint, score_model=score_model,
            map_location=args.device, weights_only=False)
    model = model.to(args.device).eval()
    return model, dict(sigma=sigma, L=L, norm_min=norm_min, norm_max=norm_max)


def load_hmc(data_path, norm_min, norm_max, num_samples, device, seed):
    """读 HMC 组态 (cfgs: (N,L,L)), 用训练归一化映到 [-1,1], 返回 (n,1,L,L) tensor。"""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    if cfgs.ndim == 4 and cfgs.shape[1] == 1:      # (N,1,L,L) -> (N,L,L)
        cfgs = cfgs[:, 0]
    assert cfgs.ndim == 3, f"expected (N,L,L) HMC cfgs, got {cfgs.shape}"
    # 与训练一致: 用 checkpoint 的 norm_min/max (不是重新从这批数据估)
    cfgs = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2

    rng = np.random.default_rng(seed)
    n = min(num_samples, cfgs.shape[0])
    idx = rng.choice(cfgs.shape[0], size=n, replace=False)
    x0 = torch.from_numpy(cfgs[idx]).unsqueeze(1).float().to(device)
    print(f"HMC samples: {tuple(x0.shape)} from {data_path}")
    return x0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True, help="HMC 训练数据 (h5/jld2, key 'cfgs')")
    p.add_argument("--t_list", type=str,
                   default="0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--num_samples", type=int, default=512, help="参与统计的 HMC 组态数")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_hutch", type=int, default=16, help="每个样本的 Hutchinson 探针数")
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random_init", action="store_true",
                   help="用未训练的随机权重 (rho 的随机天花板参照 ~0.707), 不加载 checkpoint 权重")
    p.add_argument("--output", type=str, default=None,
                   help="输出前缀 (默认 checkpoint 同目录下 curl_fraction_<ep>)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    model, hp = load_model(args)
    x0 = load_hmc(args.data_path, hp["norm_min"], hp["norm_max"],
                  args.num_samples, args.device, args.seed)
    t_list = [float(s) for s in args.t_list.split(",")]

    rows = []   # (t, rho_mean, rho_sem, rho2_mean, rho2_std, n)
    print(f"\n{'t':>8}  {'rho_mean':>10}  {'rho_sem':>10}  {'rho2_mean':>11}  {'n':>6}")
    print("-" * 56)
    with model.ema.average_parameters():           # 用采样时的 EMA 权重
        for t_val in t_list:
            rho2_all = []
            for i in range(0, x0.shape[0], args.batch_size):
                xb = x0[i:i + args.batch_size]
                rho2_all.append(antisym_fraction(model, xb, t_val, args.n_hutch, gen))
            rho2 = torch.cat(rho2_all)
            rho = rho2.clamp_min(0).sqrt()
            n = rho.numel()
            rho_mean = rho.mean().item()
            rho_sem = (rho.std().item() / np.sqrt(n)) if n > 1 else 0.0
            rows.append((t_val, rho_mean, rho_sem, rho2.mean().item(), rho2.std().item(), n))
            print(f"{t_val:8.4g}  {rho_mean:10.4f}  {rho_sem:10.4f}  {rho2.mean().item():11.4e}  {n:6d}")

    # 输出
    if args.output is None:
        ep = re.search(r"(\d+)[^/]*\.ckpt$", str(args.checkpoint))
        ep = ep.group(1) if ep else "ckpt"
        tag = "randinit" if args.random_init else f"ep{ep}"
        args.output = str(Path(args.checkpoint).parent.parent / "data" / f"curl_fraction_{tag}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    arr = np.array(rows, dtype=np.float64)
    header = "t rho_mean rho_sem rho2_mean rho2_std n"
    np.savetxt(args.output + ".csv", arr, header=header, comments="", fmt="%.6e")
    print(f"\nSaved -> {args.output}.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("t (noise level)")
    ax.set_ylabel(r"$\rho = \|J-J^\top\|_F / (2\|J\|_F)$  (非守恒分数)")
    ax.set_title(f"L={hp['L']}  sigma={hp['sigma']}  {args.network}"
                 + ("  (random init)" if args.random_init else ""))
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output + ".png", dpi=150)
    print(f"Saved -> {args.output}.png")


if __name__ == "__main__":
    main()
