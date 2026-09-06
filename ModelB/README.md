# Model B

二维守恒序参量动力学（Model B）与其条件扩散模型代理。κ=0.34034、λ=1 是
临界点。临界模型在 L=8/16/32 训练并外推到 L=64/128。粗化模型还使用
L=64 合成训练对。

求解器本体在 `solver/`，数据生成入口在 `data_generation/`。

- `solver/ModelB_fixed.jl`：fixed-SOSRA 融合实现，实际使用的就是它。
- `solver/ModelB_fixed_sciml.jl`：其 SciML wrapper。
- `solver/ModelB.jl`：通用 SciML 参考实现，供 `tests/` 对拍。

## 数据生成

```bash
julia --project=. data_generation/run_l32_l1_k0p340340_correlated_gaussian_t10.jl  # 训练对
julia --project=. data_generation/make_canonical.jl                                # canonical 初态
julia --project=. data_generation/run_modelb_references.jl                         # 各链的 Model B 参照
julia --project=. data_generation/run_multiL_extrapolation_references.jl           # multiL 的 Model B 参照
julia --project=. data_generation/run_coarsening_tanh_gaussian_pairs.jl            # 粗化训练对
julia --project=. data_generation/run_coarsening_references.jl                     # 粗化参照
```

`scan_pseudocritical.jl` 与 `save_configs.jl` 服务于 `analysis/dynamic_exponent/`。

## 训练

```bash
python train/train_modelB_cond.py --L_list 8,16,32 --kappa 0.34034 --lam 1.0 \
    --horizon 10 --sigma 250 --epochs 10000 --output_suffix _hmc_noskip
```

训练用给定文件的全部数据，并用全部数据计算 `A_res` 和 `A_cond`；没有
holdout、Lightning validation loop、`val_loss` 或 best checkpoint。只保留对数
间隔的具名 checkpoint。同名新 run 默认拒绝覆盖，续训用 `--ckpt_path`
（它从 checkpoint 路径反推 run 目录并续写同一个 run）。

超参数记在 `runs/<run名>/meta.json` 和 checkpoint 的 `hyper_parameters` 里。

## 采样

```bash
python sample/run_chains.py            # 全部链：dt=10 到 t=1000，dt=0.5 到 t=50
python sample/run_chains.py L64        # 只跑名字含 L64 的
```

链一律从 t=0 直跑到目标时间，不做续算、不拼接段落；每条链写成一个
`chain.h5`（时间是最外维）。逐条链的 checkpoint、初态、seed 在
`run_chains.py` 顶部的表里。

反向生成在**零和残差子空间**内进行：`cond_lightning.py::sample()` 对初始
残差噪声、每步 score、每步噪声和最终残差都投影掉空间均值，因此逐轨迹的
总磁化守恒到 ~1e-8（Model B 求解器本身是 ~1e-6）。

## 分析

```bash
python analysis/chain_data.py          # 单个 run 的 DM vs Model B，写 tsv/npz
python analysis/plot_validation_L32.py # L=32 验证图
python analysis/plot_multiL_extrapolation.py   # mixed-L 的长时外推与标度坍缩
python analysis/compare_multil_reverse.py      # 单 vs 混 kernel 的单步精度
python analysis/dm_vs_sosra/run_all.py         # 动力学标度分析
python analysis/dm_vs_sosra/plots_prl.py       # 论文图
python analysis/plot_coarsening_references.py  # Model B 粗化参照图
python analysis/plot_coarsening_dm.py          # 扩散模型粗化图
```

`chain_data.py` 默认跑到 t=2000，用 `MODELB_CHAIN_TMAX` 改；
`MODELB_CHAIN_RUN` 指定 run。

## 目录

遵循 `../docs/REPO_LAYOUT.md`。

- `solver/`：Model B 求解器本体，只有三个实现文件（`ModelB_fixed.jl` 融合
  SOSRA、`ModelB_fixed_sciml.jl` 其 SciML 包装、`ModelB.jl` 通用参考实现）。
  它们是被 include 的库，不是入口。
- `data_generation/`：训练 target、参考轨迹、续演化和 canonical 初态的入口脚本。
- `cond_lightning.py`：conditional model 本体，项目根的库模块。
- `train/`、`sample/`：训练与采样入口。
- `edm2/`：EDM2 条件网络的训练与生成入口。
- `jobs/`：PBS 作业脚本。
- `analysis/`：关联函数、验证、尺寸外推和粗化分析。
  `analysis/dynamic_exponent/` 使用 canonical 初态分析 L=32 动量色散，不固定
  \(M=0\)，也不做不同尺寸的 \(z\) 拟合。
- `tests/`：只有求解器正确性测试（`test_fixed_sciml.jl`、`test_generic_modelb.jl`）。
- `trainingdata/`：训练输入。
- `runs/<run名>/`：训练检查点、样本、TensorBoard 记录和运行元数据。
- `results/<主题>/`：全部分析产物。主题为 `dm_vs_modelb`、`multiL`、
  `modelb_reference`、`dynamic_exponent` 和 `coarsening`。跨多个 run 的主题下
  再按 run 名分。
- `results/dm_vs_modelb/<run名>/` 只保存 NPZ 和 TSV 数值产物。主题级图放在
  `results/dm_vs_modelb/`，论文采用的图再复制到 `draft/figures/`。
- 跨格点尺寸的派生结果统一放在 `results/multiL/analysis/`，不混入来源 run
  的 L32 验证目录。
- `logs/`：生成与分析的运行日志。

已完成 run 的中间 checkpoint 归档在 `models/logscale_ckpts.tar`。可用
`tar -xf` 还原。最终的 `epoch=10000.ckpt` 保持直接可用。
