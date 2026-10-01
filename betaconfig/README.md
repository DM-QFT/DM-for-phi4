# betaconfig

沿反向扩散轨迹测量构型的有效逆温度 beta_conf，并验证它作为准平衡判据是否成立。

## 入口

| 做什么 | 命令（从 `betaconfig/` 跑） |
|---|---|
| 单个 L 的 beta_conf 轨迹 | `python analysis/beta_conf_along_reverse.py --L 64 --k 0.2705 --l 0.022` |
| beta_conf 随 epoch 演化 | `python analysis/beta_conf_vs_epoch_2d.py --L 64 --k 0.2705 --l 0.022` |
| 多 L 正/反向对比 | `python analysis/beta_conf_fwd_rev_multiL.py --device cuda` |
| 多 L 汇总图 | `python analysis/plot_beta_conf_fwd_rev_multiL.py` |
| 论文图 | `python analysis/draft_figures.py` |

验证脚本在 `tests/`，按 a2/b1/b2/b3/c1/c2/c3/c4 分组，结论见 `tests/REPORT.md`。
`tests/*.jl` 需要 `tools/julia/WolffFAHMC_ND.jl`。

## 参数

kappa=0.28、lambda=0.022，L=8..256，ep10000、steps2000、log schedule。

3D 版 c2（`tests/c2_sensitivity_3d.py`）用 `3Dphi4/runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp` 的 ep=10000，
L=32 在训练范围内，L=64 是跨 L，步数扫到 4000，compile(default) 加 EMA。四节点作业
`tests/pbs_c2_3d.sh` 由用户提交。HMC 参考值 L=64 为 β̂=1.00007(11)。

## 目录

遵循 `docs/REPO_LAYOUT.md`。

```
analysis/   分析与绘图代码
tests/      验证代码（common.py 提供 ROOT/OUT/FIGS 与共用估计量）
results/
├── beta_conf_traj_multiL/   多 L 轨迹的 npz/pdf/png
└── verification/            tests/ 的产物，figs/ 与 crosscheck_wolff/ 在内
cjp.mplstyles                论文图样式
```

论文图输出到 `paper/betaconfig/assets/figs/2dphi4/`，预览 png 到
`results/verification/figs/`。

## 2026-08-05 整理对照

| 原位置 | 现位置 |
|---|---|
| 顶层 5 个 `*.py` | `analysis/` |
| `verification/*.py`, `*.jl` | `tests/` |
| `verification/` 的产物、`figs/`、`crosscheck_wolff/` | `results/verification/` |
| `verification/REPORT.md` | `tests/REPORT.md` |
| `draft/` | `paper/betaconfig/`（仓库级 Tier 1） |

改路径的地方：脚本下沉一层，`ROOT = HERE.parent` 改为 `HERE.parents[1]`；
项目内路径改用 `HERE.parent`；`tests/` 的产物写入由 `HERE` 改为
`common.OUT`。`beta_conf_along_reverse.py` 与 `beta_conf_vs_epoch_2d.py`
原先用 `Path("results/beta_conf_traj")` 这种相对 CWD 的写法，一并改为项目相对。
