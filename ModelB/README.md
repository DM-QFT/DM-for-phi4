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

Fig. 3 的百万对 HMC 续训使用两个独立作业，分别对应 Δt=0.5 和 10。
每个作业与 Fig. 2 一样，自行生成训练数据，再从旧检查点续训。
两个作业可以并行，使用各自的种子、数据文件和日志。
每个核共 1024000 对，其中 L=8 和 16 各 341248 对，
L=32 有 341504 对。HMC 热化、抽稀和 SOSRA 步长沿用旧数据的设置。
两个核分别加载原 Fig. 3 混合尺寸模型的 `epoch=10000.ckpt` EMA 权重和归一化常数。
NCSN++ 通道保持 `[16,32,64,128]`，参数总数为 2564305。
沿用 Fig. 2 当前版本的权重续训方式，重新初始化 Adam 和 EMA 计数，学习率 0.001，
batch 256，额外训练 2000 epoch。每个核更新 800 万步。
开启 `torch.compile` 的 `reduce-overhead`。GH200 上 600 步短训的后半段约
6.0 ms/步，估计每个核约 13.4 小时，尚未计入编译和检查点写入。

在 ModelB 目录分别提交。每个作业使用一张 GPU，墙钟上限为 24 小时。

```bash
qsub jobs/train_modelB_multiL_hmc_1m_b256_t0p5.pbs
qsub jobs/train_modelB_multiL_hmc_1m_b256_t10.pbs
```

新模型写入 `runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t{0.5,10}_ncsnpp_hmc_1m_b256_noskip/`。
最终权重为各目录中的 `models/epoch=2000.ckpt`。
作业日志为 `runs/modelB_multiL_hmc_1m_b256_t0p5.pbs.log` 和
`runs/modelB_multiL_hmc_1m_b256_t10.pbs.log`。
旧模型和论文图继续保留。

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

## 稿件仓库 `draft/`

`draft/` 是嵌套在 DM 仓库里的独立 Overleaf clone，remote 名 `overleaf`，分支 main，
凭据在 `~/.git-credentials`。稿件正文的写作规则在 `CLAUDE.md`。

- 所有操作写 `git -C /work/gw97/w97000/project/DM/ModelB/draft ...`。禁止 `commit -am`
  和 `add -A`。2026-08-23 一条裸 `git commit -am` 跑在父仓库里，提交了 75 个无关文件。
- 推送流程是 fetch overleaf、`merge --ff-only overleaf/main`、commit、`push overleaf main`。
  新图和正文一起 commit。
- `ml4ps2026/` 和 `reference_audit.md` 故意不进 Overleaf。`draft/arxiv/` 是 arXiv 打包快照，
  对应 `modelb_arxiv.tar.gz`，改稿只改根目录的 `modelb.tex` 再重新打包。
  `overleaf_modelb/` 是旧副本，不要推。
- Miyabi 没有 latexmk，用 pdflatex 加 bibtex 编译。
- ML4PS 2026 的 4 页短文 `draft/ml4ps2026/ml4ps2026.tex` 已投，2026-10-10 出通知，
  非 archival，双盲。arXiv 全文等通知后再挂，不单独挂 4 页版。

## 论文图

样式表 `../plots/cjp_prl_nature.mplstyle`（字号 10.5/9.5）加 `../plots/modelb_palette.py`。
大字号是 2026-09-10 定的，不钉回旧图的 8/9。

| 图 | 脚本 | 备注 |
|---|---|---|
| Fig.1 schematic | `analysis/plot_schematic_framework.py` | 2026-09-06 从旧 PDF 反演重建。面板是 L128 tanhgauss 链轨迹 0 的真实数据加示意噪声。usetex 需要 `~/texmf` 里的 type1cm |
| Fig.2 validation_L32、S2 rho_L32 | `analysis/plot_validation_L32.py` | boxa3 核。Fig.2 全部 add_axes 按英寸手排，字号混合（(a) 与色标 10.5，(b)(c) 9，行标签 8）。S2 纯样式表 |
| Fig.3 | `analysis/dm_vs_sosra/plots_prl.py` | 主轴按样式表，legend 8，inset 7/8，比值 inset 只画圈不连线 |
| Fig.4、S7 | `analysis/plot_coarsening_dm.py` | 同 Fig.3 |
| Fig.5 | `analysis/plot_experiment.py` | 标注全在轴框内，是这条规则的标准样例 |
| S5 | `analysis/plot_L32_chain_paper.py` | 严格按样式表，无覆盖 |
| S6 | `analysis/plot_one_step_paper.py` | 严格按样式表，无覆盖 |

- `plot_validation_L32.py` 和 `plot_coarsening_dm.py` 一次重写两张图，直接写进 `draft/figures/`。
  只改一张时另一张用 `git -C draft checkout --` 还原。
- 重跑前用 fitz 的 `get_text("dict")` 比旧 PDF 的 span 字号。图在稿里的尺寸要在编译后的
  `modelb.pdf` 里量轴框。按原尺寸插入时写显式 `width=<figsize>in`。

## 经验与决定

代码里看不出来的事实和已经定下的决定。

**求解器。** 生产实现是 `solver/ModelB_fixed.jl`（融合 SOSRA，dt=0.002）。带宽已经打满，
dt 是唯一的杠杆，dt=0.02 时 action 偏差只有 1%。永远显式传 dt，SDE 的 auto-initdt 在这个
刚性系统里会坏。DRI1 类求解器结构性无效。kernel 用 Int32 索引，L=32 单次最多 524287 个构型。
`physical_reference` 的 jld2 经 h5py 读出后相对 npz 是空间转置的，配对前 `np.swapaxes(a,1,2)`。
Float32 有两个陷阱，都只在临界点、慢模、长时间、红外量、有独立精确基准时才看得见。

- 谱积分器里 `exp(-λdt)*uk` 对慢模的舍入误差是 2.5%，t=2000 时 S(k) 偏 19%。
  必须写 `uk + expm1(-λdt)*uk`。
- R2C 半谱存态时 kx=0 与 kx=L/2 两列的反对称分量要每步投影回共轭对称，否则 t≈600 炸掉。

**条件核。** 目标是残差 Δφ 不是场。条件 φ0 是干净通道，不加噪。σ 由残差空间的
std(1)≈2·D_max 定。Z2 配对增广同时翻两个通道。生成端的零和投影强制 ΔM=0，每条链保持
自己的 M₀，不是投到 M=0。`load_from_checkpoint` 前按 `hparams["channels"]` 用
`in_channels=2` 重建网络。

**compile。** 训练和 `sample/run_chains.py` 用 `reduce-overhead`。L=128 采样要用 `default`
（`sample/extend_quench_L128.py`），`reduce-overhead` 的 cudagraph 会破坏 NCSNpp 的 nearest
上采样。`results/multiL/t10_L128.h5` 和 `t0p5_L128.h5` 的 attrs 记的是 `reduce-overhead`，
是否受影响没有核对过。

**L32 临界核的训练集谱系（合成相关高斯条件）。**

- zlog（Z∼log-U[0.05,10]）平衡态 G_1 比值 0.94，Δt=0.5 版迭代到 t=20 只剩 0.855。
  短视界核每步的软模偏差被 τ_1/Δt 放大（Δt=10 时 19 倍，0.5 时 380 倍），短视界核不能
  默认可以迭代。
- 覆盖密度是主因。zhmc 窄盒无漂移，zbox 宽盒更糟。
- 盒 A（Z[0.45,1.15]，m²[0,0.8]，M±0.8，定为不用联合重采样）的 boxa2 消除了 6% 亏损，
  但 n≥2 的模低 2 到 3%，τ_1 高 10%。宽网络 loss 更低反而过冲，DSM loss 低不等于长时统计好。
- boxa3（1024000 对，从 boxa2 续训 2000 epoch）是目前最好的核，逐模 1% 以内，τ_1 偏 −0.4%。
  Fig.2 和 S2 用它。
- 再做训练集时按 `results/train_dist/clouds.npz` 的经验云联合重采样 (Z, m², M)。
  HMC 里 M 强双峰（±0.65），m² 与 |M| 强相关。

**测量。** 链上 G_1 比值的实现噪声约 ±1.5%（512 条轨道），3% 以内的核对比不可判，要 2048 条
轨道或两组种子。同种子不同反向步数是配对的，2000 步够用。`hmc_ep10000_t0_N1024.npz`
前后两半的 G_1 差 30%，取子集必须等间隔（`--n` 已经这样做）。

**结果与决定。**

- dm_vs_sosra 管线（multi-L HMC 核）L32 给出 z_DM=3.792(19)，z_SOSRA=3.812(20)，
  双方各 2048 条。非指数尾是真值的性质，L64 的迭代链未平衡。z 只用 canonical 初态的 L=32
  色散，早期的 τ(L) 有限尺寸拟合已退役。
- 粗化用 κ=0.85、L=128，可信窗口 L(t)<0.2 L_box。训练条件是硬化关联高斯 tanhgauss。
  OU 谱失败，UV 斑点淹没大畴。只在 t0 训练的核迭代必发散。multiL 模型外推到 L=128 的
  L(t) 比值是 1.0085(48)。
- 胶体线的参照只用论文的 Langevin，不用高斯基线。均匀 t 采样加 10 倍 epoch。eq 出射时间的
  6% 亏缺稿子里不解释。条件里加原始位置必败，坐标会泄露轨迹身份。
- 训练成本实测 L32 zlog 约 12 h，multi-L hmc 约 40 h，coarsening 约 21 h。

**整步 CUDA graph 训练（`--cuda_graph`）。** 小网络被 Lightning 循环主导时有用，MLP 快 5 倍，
NCSNppCond2D 快 1.6 倍。此时 compile 必须 `default`（inductor 的 CUDA graph 不能嵌套），
`Adam(fused=True, capturable=True, lr=tensor)`，EMA 保留 torch_ema 的预热。主机端的随机增广
会被冻成固定的群元素，`torch.tensor(float, device)` 不能进 step()。采样端计算主导，没有收益。
