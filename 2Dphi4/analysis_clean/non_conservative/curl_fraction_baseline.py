"""
Baseline: 用真实漂移 -∂S/∂φ 替代 score net, 计算反对称分数 rho。

玻尔兹曼分布 p(φ) ∝ exp(-S(φ)) 的真实 score 是
    s(x) = ∇log p = -∂S/∂x        (归一化 [-1,1] 空间, 见 phi4_action.phi4_grad_S),
它按构造是标量场 -S 的梯度 => Jacobian = -Hessian(S) 对称 => 守恒, rho ≡ 0。

因此本程序给出估计器的 "噪声地板": 在一个可证明守恒的场上跑同一套 Hutchinson 估计,
得到的 rho 只来自有限探针数 + 浮点误差。把它和 curl_fraction.py 跑出的 score-net 的
rho(t) 叠在一起, 高出这个地板的部分才是 score net 真正的非守恒残留。

评估点与 score net 一致: HMC 样本前向加噪到 p_t (x_t = x0 + std(t)·z), 再在该点求
-∂S/∂x 的 Jacobian 反对称性。(注: -∂S/∂x 是 t=0 的真score; t>0 时它仍是守恒场,
 rho 应保持 ≈0, 与 t 无关 —— 这正是地板该有的样子。)

复用 curl_fraction.py 的 antisym_fraction / load_hmc, 不修改该文件。

用法:
    python curl_fraction_baseline.py \
        --checkpoint runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640/models/epoch=10000.ckpt \
        --data_path trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_128^2.jld2 \
        --t_list 0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0 \
        --num_samples 512 --batch_size 512 --n_hutch 16 \
        --device cuda:0 --seed 0
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))   # repo root (DM): diffusion_lightning, networks, phi4_action
sys.path.insert(0, str(_HERE.parent))       # 同目录: curl_fraction

import os
import re
import functools
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_lightning import marginal_prob_std
from phi4_action import phi4_grad_S
from curl_fraction import antisym_fraction, load_hmc


class ForceField:
    """冒充 antisym_fraction 需要的最小 score 接口, 返回真实 score s = -∂S/∂x。

    需要的属性/方法: marginal_prob_std_fn (加噪), _bcast (广播), __call__(x, t)。
    s(x) 与 t 无关 (t=0 真score); 是守恒场 => rho≈0。
    """

    def __init__(self, k, l, norm_min, norm_max, sigma, spatial_dims=2):
        self.k = k
        self.l = l
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.spatial_dims = spatial_dims
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    def _bcast(self, t):
        return t.view((-1,) + (1,) * (self.spatial_dims + 1))

    def __call__(self, x, t):
        return -phi4_grad_S(x, self.k, self.l, self.norm_min, self.norm_max,
                            self.spatial_dims)


def read_hparams(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    return dict(
        sigma=h.get("sigma", 150.0),
        L=h.get("L", 128),
        norm_min=h.get("norm_min") or -6.22,
        norm_max=h.get("norm_max") or 6.19,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="该 run 的任一 ckpt, 用于读取 sigma/norm_min/norm_max (与 net 对齐)")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--k", type=float, default=None, help="默认从 data_path 解析")
    p.add_argument("--l", type=float, default=None, help="默认从 data_path 解析")
    p.add_argument("--t_list", type=str,
                   default="0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_hutch", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    hp = read_hparams(args.checkpoint)
    # k, l: 优先命令行, 否则从 data_path 解析 (e.g. ..._k=0.28_l=0.022_...)
    k = args.k if args.k is not None else float(re.search(r"k=?([\d.]+)", args.data_path).group(1))
    l = args.l if args.l is not None else float(re.search(r"l=?([\d.]+)", args.data_path).group(1))
    print(f"k={k}, l={l}, sigma={hp['sigma']}, norm=[{hp['norm_min']:.4f}, {hp['norm_max']:.4f}]")

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    force = ForceField(k, l, hp["norm_min"], hp["norm_max"], hp["sigma"])
    x0 = load_hmc(args.data_path, hp["norm_min"], hp["norm_max"],
                  args.num_samples, args.device, args.seed)
    t_list = [float(s) for s in args.t_list.split(",")]

    rows = []
    print(f"\n{'t':>8}  {'rho_mean':>10}  {'rho_sem':>10}  {'rho2_mean':>11}  {'n':>6}")
    print("-" * 56)
    for t_val in t_list:
        rho2_all = []
        for i in range(0, x0.shape[0], args.batch_size):
            xb = x0[i:i + args.batch_size]
            rho2_all.append(antisym_fraction(force, xb, t_val, args.n_hutch, gen))
        rho2 = torch.cat(rho2_all)
        rho = rho2.clamp_min(0).sqrt()
        n = rho.numel()
        sem = rho.std().item() / np.sqrt(n) if n > 1 else 0.0
        rows.append((t_val, rho.mean().item(), sem, rho2.mean().item(), rho2.std().item(), n))
        print(f"{t_val:8.4g}  {rho.mean().item():10.4e}  {sem:10.4e}  {rho2.mean().item():11.4e}  {n:6d}")

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(args.checkpoint)),
            "data", "curl_fraction_baseline_dSphi")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    arr = np.array(rows, dtype=np.float64)
    np.savetxt(args.output + ".csv", arr,
               header="t rho_mean rho_sem rho2_mean rho2_std n", comments="", fmt="%.6e")
    print(f"\nSaved -> {args.output}.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t (noise level)")
    ax.set_ylabel(r"$\rho$  (baseline: $-\partial S/\partial\phi$, 守恒场地板)")
    ax.set_title(f"L={hp['L']}  k={k}  l={l}  (conservative floor)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output + ".png", dpi=150)
    print(f"Saved -> {args.output}.png")


if __name__ == "__main__":
    main()
