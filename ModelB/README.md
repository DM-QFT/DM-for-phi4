# Model B

Julia 求解器和数据生成入口统一放在 `data_generation/`：

- `data_generation/ModelB.jl`：通用 SciML SDE 实现。
- `data_generation/ModelB_fixed.jl`：fixed-SOSRA 融合实现。
- `data_generation/ModelB_fixed_sciml.jl`：fixed-SOSRA 的 SciML wrapper。

## 训练流程

生成配对训练 target：

```bash
julia data_generation/run_hmc_10240.jl
julia data_generation/run_hmc_10240.jl 32
```

生成 L32 相关 Gaussian 加随机常数初态及其 `t=10` target：

```bash
julia data_generation/run_l32_correlated_gaussian_t10.jl
```

`run_hmc_10240.jl` 读取 `../2Dphi4/trainingdata/` 中的 HMC 初态，并把 Model B 端点写入
对应的 `runs/L{L}_Ne10240_dt0p002_hmc_t1_t10_t100/`。

其它 Model B 数据生成与续演化入口也统一放在这里：

```bash
julia data_generation/run.jl
```

训练 conditional diffusion model：

```bash
zsh -ic 'nenv2 && python training/train_modelB_cond.py --epochs 10000 --output_suffix _all_data'
zsh -ic 'nenv2 && python training/train_modelB_cond.py --L 32 --epochs 10000 --output_suffix _all_data'
```

训练上述相关 Gaussian 数据不需要修改训练程序：

```bash
zsh -ic 'nenv2 && python training/train_modelB_cond.py --L 32 --cond_path runs/L32_Ne51200_dt0p002_corrgauss_uniformM_t10/modelB_L32_Ne51200_corrgauss_t0_seed20260802.jld2 --target_path runs/L32_Ne51200_dt0p002_corrgauss_uniformM_t10/modelB_L32_Ne51200_corrgauss_kappa_0.28_t10_dt0.002_seed20260803.jld2 --output_suffix _corrgauss_uniformM_noskip'
```

训练直接使用所给文件中的全部数据（HMC 文件为 10240 对，上述相关 Gaussian
文件为 51200 对），并用全部数据计算 `A_res` 和 `A_cond`；没有
holdout、Lightning validation loop、`val_loss` 或 best checkpoint。数据始终
驻留在指定 GPU，batch 默认为 64。程序只保留对数间隔的具名 checkpoint；
默认请求 100 个点，在 10000 个整数 epoch 上去重后实际为 84 个。
默认使用 `torch.compile(mode="reduce-overhead")`，用 `--compile_mode` 改。

归一化来自全部数据，因此从任一 checkpoint 都能恢复模型、Adam 和 EMA：

```bash
zsh -ic 'nenv2 && python training/train_modelB_cond.py --L 32 --epochs 20000 --ckpt_path runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip/models/epoch=10000.ckpt'
```

`--ckpt_path` 从 checkpoint 路径反推 run 目录并续写同一个 run。

采样默认使用条件文件中的全部配置；测试小批量时用 `--cond_indices`：

```bash
zsh -ic 'nenv2 && python training/sample_modelB_cond.py --run_dir runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip --ep 10000 --cond_indices 0:256'
```

反向生成默认对 score network 使用 `torch.compile(mode="reduce-overhead")`；
`--compile_mode` 的值直接传给 `torch.compile`，采样器不另做检查。
生成结果保留网络直接输出的零模，不再做终点均值投影；NPZ 中的
`zero_mode_raw` 用于检查网络自行学到的守恒精度。

同名新 run 默认拒绝覆盖；续训需显式传入 `--ckpt_path`。生成后的独立物理
统计和 ground-truth 检查仍在 `validation/`，它们不占用训练数据。

## 目录

- `data_generation/`：训练 target、参考轨迹和续演化的入口脚本。
- `solver/`：Model B 求解器本体（`ModelB_fixed.jl` 融合 SOSRA、
  `ModelB_fixed_sciml.jl` 其 SciML 包装、`ModelB.jl` 通用参考实现）。
- `training/`：conditional model、训练和采样。
- `validation/`：独立 Model B ground truth 与条件模型验证。
- `analysis/`：对角模式关联分析。`t200_data.py` 只算数据并写
  tsv/npz，`t200_plots.py` 只读这些文件出图，两者不互相 import。
- `dynamic_exponent/`：固定 \(M=0\) 初态、最低模实时测量和有限尺寸
  \(z\) 拟合。
- `checks/`：求解器测试。
- `runs/`：数据、checkpoint、样本和分析结果；已有路径保持不变。

`runs/` 的两点现状：`_noskip` run 的 83 个中间 checkpoint 已归档为
`models/logscale_ckpts.tar`（`tar -xf` 还原），只有 `epoch=10000.ckpt` 直接可用；
L128 长时间演化（t>2000）的原始 jld2 已删除，只保留 propagator、two-time
和时间关联的分析产物，重算需先用 `data_generation/run.jl` 重新生成轨迹。
