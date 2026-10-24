# 粒子 U-Net 条件扩散，A ↔ 2A 反应扩散玩具模型

通向 UrQMD 的第一步。学习固定物理间隔 `DT_PAIR = 10` 的条件坐标分布
`p(X10 | X0, N10 = M)`，自由度数 `N0 → M` 由真实模拟器决定。方案在
`particle_unet_dm_v0.md`。

代码自包含，不 import 上层目录。画图样式 `cjp_prl_nature.mplstyle` 和
`modelb_palette.py` 从 `../plots/` 复制而来。训练和生成只在 GPU 上跑，
入口脚本开头 `torch.set_default_device("cuda")`，训练集整个常驻 GPU，每步只切片拼接，
采样把整个测试集一次送进网络，不分 batch。

网络主体 `core` 用 `torch.compile(mode="default", dynamic=True)`。几何构建（cell list、
分配权重）形状随数据变，留在 eager。`pad_geometry` 把节点、粒子边和粒子到网格的分配对
补齐到 `BUCKET` 的倍数（哑节点只收自环消息和零权重分配），网格内部的边数由 L 决定本来固定。
编译的是方法不是 module，checkpoint 的键不带 `_orig_mod` 前缀。
2026-10-12 在本机 torch 2.12 上用 6 事例 batch 探针，eager 65 ms/步，`default` 加动态形状 35 ms，
`reduce-overhead` 加动态形状在反向传播里段错误，`reduce-overhead` 静态形状（桶放大到只有一种形状）
稳态 100 ms，所以训练不用 `reduce-overhead`。

## 物理

`config.py` 是唯一的参数来源。长度单位 `r = 1`（反应半径）。

| 量 | 值 | 来源 |
|---|---|---|
| `D` | 0.025 | 扩散长度 `sqrt(4 D DT_PAIR) = 1 r`，一个平均间距 |
| `LAM` | 0.05 | 每粒子分支率，`DT_PAIR` 内平均分支 0.5 次，长波记忆 `e^{-LAM DT_PAIR} = 0.61` |
| `GAM` | `2 LAM / pi` | 每个 `r` 内无序对的合并率，使密度 1 的 Poisson 点过程成为严格平稳态 |
| `DT_SIM` | 0.05 | Euler 方案事件率误差约 `(GAM pi/2 + LAM) DT_SIM`，低于 0.5% |
| `SIGMA_MIN, SIGMA_MAX` | 1e-2, 40 | 训练与采样共用。σ 低于 1e-2 时目标纯是噪声，`SIGMA_MAX` 取最大测试盒 |

**关键事实。** 出生核（半径 r 圆盘均匀）与合并核（半径 r top-hat）形状相同，
在 `GAM = 2 LAM / (pi r^2 n*)` 下 Poisson(n*) 对跳跃部分满足逐点 detailed balance，
Brownian 部分也保持 Poisson，所以均匀密度 1 的 Poisson 点过程是严格平稳态，
没有任何等时空间关联。后果有两条。一是 `g(ρ)`、`S(k)`、box-count 方差对模型完全不敏感，
它们只能当作粒子坍缩的检测，评估必须用条件量，即密度传播子
`G(k) = <ρ_k(X0)* ρ_k(X10)> / <|ρ_k(X0)|^2>` 和交叉对关联 `g01(ρ)`。
均匀 X0 的线性响应参考为 `G(k) = exp(-(LAM + D k^2) DT_PAIR)`、
`g01(ρ) = 1 + exp(-LAM DT_PAIR) Gauss(ρ; 2 D DT_PAIR)`。
二是均匀初态几乎不含条件信息，所以训练和测试的初态轮流取均匀 Poisson、
正弦密度调制（振幅 `U(0, 0.7)`，整数波矢 `|n| ≤ 3`）和 Gaussian 团簇/空洞
（中心密度 3–5 倍或 0.2–0.5 倍），数据文件里 `init` 记录类型。
每条轨迹连续记录 `K = 4` 对，`k` 记录是第几对（k=0 的 X0 就是结构化初态本身）。
训练集和测试集按轨迹划分。`L=16, 32` 另有 16 个结构化初态各 64 个独立未来（`*_multi.h5`）。

训练尺寸为 `L = 8, 10, ..., 32` 共 13 个，每个尺寸的轨迹数按 `131072 / L^2` 取，
使每个尺寸的粒子总数相同，测试集为其四分之一。`L = 48, 64` 只做测试，各 64 条轨迹。
`L = 8` 的团簇/空洞初态会把盒子几乎清空，存在 `N0 = 0` 或 `N10 = 0` 的事例，
训练里 `M` 取 `max(M, 1)` 让它们贡献零。

数据标签。`reaction` 为 A ↔ 2A，`brownian` 为同样初态但关闭反应（X10 = X0 加独立
Gaussian 位移），`single` 为每盒一个自由粒子（wrapped Gaussian 精确解）。

## 入口

```bash
python solver/check_dt.py                       # 模拟步长收敛，看每粒子事件数、G(k)、g01，写 results/solver/
python solver/make_pairs.py reaction            # 训练对，写 trainingdata/。另有 brownian、single
python tests/test_geometry.py                   # 周期 cell list 对照 scipy cKDTree
python tests/test_model.py                      # 形状、置换、batch 一致、平移、D4 对称、局域性、网格相位敏感度
python train/train_particle.py reaction unet <run>      # 混合尺寸 L=8..32 训练。gnn 为单尺度基线
python sample/sample_particle.py <run> <step> <L> <split> <K>   # 已知 M 的 Heun 反向扩散，写 runs/<run>/samples/
bash sample/eval_sizes.sh <run> <step> [K]              # L=8,16,32,48,64 采样加比较，再做 L=16、64 的 loss_bins
python analysis/brownian/single_particle.py <run> <step> <K>    # 单粒子位移对比精确 score 的同一采样器和 wrapped Gaussian
python analysis/brownian/single_particle_score.py <run> <step>  # 单粒子 ε̂ 对比精确公式
python analysis/conditional/loss_bins.py <run> <step> <L> <split>              # 按 log σ 分 bin 的测试 loss，真 X0 与打乱 X0
python analysis/conditional/compare_statistics.py <run> <step> <L> <split> <K> # g、g01、S(k)、G(k) 生成 vs 模拟
```

## 目录

遵循 `../docs/REPO_LAYOUT.md`。库模块在项目根，入口脚本按类别分目录，产物不入库。

- `config.py` 物理、噪声、网络几何常数。`geometry.py` 跨事例向量化的周期 cell list、
  辅助网格、单位分解权重。`model.py` 网络与 batch 拼接。`simulate.py` 模拟器、初态、h5 读写。
  `observables.py` 点集统计量。
- `solver/` 数据生成和步长检查。`train/`、`sample/` 训练和采样入口。
- `tests/` 正确性验证。`analysis/<主题>/` 分析脚本，产物镜像到 `results/<主题>/`。
- `trainingdata/` 训练对。`runs/<run>/{models,samples,logs,meta.json}`。`logs/` 作业日志。

数据文件格式。`x0`、`x10` 为所有事例坐标的拼接 `float32[*, 2]`，`off0`、`off10` 为
事例偏移，`L`、`traj`、`k`、`init` 每事例一个，物理参数在 attrs。`read_pairs` 还原为列表。

## 网络

`ParticleUNet`。节点为 `X0 ∪ Z` 并带类型标记，每个节点保留标量通道和二维矢量通道，
消息传递为 PaiNN 式（标量门控、相对位移 d 本身而非单位矢量、矢量通道线性组合），
求和聚合，每个节点内 LayerNorm，计数通道单独保留。三层级为粒子（半径 1）、
间距 1 的周期辅助网格（半径 2）、间距 2 的网格（半径 4）。起始小配置为通道
32/8、64/16、128/32，每级 1 块，感受野半径 20，约 2.9e5 参数。粒子与网格之间用
余弦截断核做单位分解权重，下采样和上采样共用同一组权重，下采样的和除以固定期望计数
2 和 4。残差块最后一层零初始化。时间嵌入用 `log σ` 的固定几何频率。输出为 `Z` 节点
矢量通道的线性组合，所有作用在矢量上的线性映射无偏置。网格平移在 `[0,2)^2` 抽，
训练每事例一次，采样每步一次。

对称性。平移严格等变，网格相位为零时盒子的 D4 群严格等变，粒子置换等变，
`tests/test_model.py` 在 `L=64`、每级 2 块（感受野 30）验证了局域性。

## 采样

VE 反向 SDE 在方差变量 `v = σ^2` 下的 stochastic Heun，预测步在 `σ_hi`，校正步在 `σ_lo`，
两步共用同一 Gaussian 增量。Euler–Maruyama 的一阶偏差约 `11/K`（单粒子方差 K=128 时 +9%），
Heun 在 K=128 已低于 0.2%。单粒子检查里用精确 wrapped score 跑同一采样器作参考，
把积分器误差和网络误差分开。每次生成显式指定 `M`，用真实未来的 `M` 只是已知粒子数诊断。

## 状态（2026-10-12，小规模 v0）

每组训练 1000 步、batch 12 个事例（单粒子 300 个）、lr 1e-3、EMA 0.995，约 1 分钟。
采样 K=128 的 Heun，整个测试集一次送入，L=32 的 256 个事例每步 0.19 s。
run 名为 `brownian_unet_v1`、`single_unet_v2`、`reaction_unet_v1`，产物在 `results/brownian/` 和 `results/conditional/`。

- 几何和网络测试全部通过。参数与评估方案按 2026-10-12 的三方审查（物理、网络、扩散数学）修订，见上文关键事实。
- 单粒子。L=16 生成位移方差 0.53，精确 score 跑同一采样器 0.49，解析 0.50，KS p=0.05。
  σ=3 时网络 ε̂ 与精确公式整条重合（含周期回绕），σ=0.3、1 只在训练从不出现的 |z−x0|>4 处偏低。
  L=24、32 有少量事例没找到 x0（方差 5.9、47），大盒子事例少且每步只有 100 个。
- 纯扩散 L=16。g01(0) 1.36 对 1.38，G(k_min) 0.93 对 0.96，G(k) 整条随 k 的衰减与模拟一致，S(k) 的长波峰（来自结构化初态）也复现。
- 反应 L=16。g01(0) 1.19 对 1.21，G(k_min) 0.43 对 0.46，G(k) 曲线与模拟和线性响应一致。
- 尺寸外推（冻结参数）。L=32、48 上 g01(ρ) 和 k ≥ 0.5 的 G(k) 仍与模拟一致，
  但 k < 0.5 的长波记忆丢失，反应 L=48 的 G(k_min) 为 0.11 对 0.58，纯扩散 L=48 为 0.32 对 1.0。
  `loss_bins` 显示网络只在 σ ∈ [0.2, 4] 用到了 X0，σ > 4 的 loss 与打乱 X0 无差别。
  波长 λ 的密度模在 σ ≲ λ/(2π) 之前都要靠网络写进样本，λ > 12 r 对应 σ ∈ [4, 16]，正是没学到的区间。
  L=16 的最低模 k=0.39 在 σ ∈ [1, 4] 决定，所以 L=16 没问题。这是训练量和大 σ 处信号弱（ε 单位下
  loss 降幅约 (kσA)^2）的问题，不是几何或等变性的问题。可选的下一步是加训练步数、对 σ 加权或改预测量。
- 反应 g(ρ) 在 ρ < 0.2 处生成样本比模拟低约 5%（母子对的短程峰），对应 σ < 0.2 处也没学到。

## 状态（2026-10-12，13 尺寸混合训练，外推到 64）

训练集 `L = 8..32`，每步每尺寸 2 个事例共 26 个，编译后 1 块/级 50 ms/步，2 块/级 90 ms/步。
run 名 `reaction_unet_v2`（1 块，1000 步）、`reaction_unet_v3`（1 块，5000 步）、
`reaction_unet_v4_2blocks`（2 块，5000 步）。评估用 `sample/eval_sizes.sh`。

| L | G(k_min) 模拟 | v2 1 块 1000 步 | v3 1 块 5000 步 | v4 2 块 5000 步 | g01(0) 模拟 | v4 |
|---|---|---|---|---|---|---|
| 8 | 0.370 | 0.377 | 0.377 | 0.380 | 1.143 | 1.124 |
| 16 | 0.428 | 0.354 | 0.395 | 0.408 | 1.199 | 1.185 |
| 32 | 0.522 | 0.190 | 0.229 | 0.281 | 1.226 | 1.176 |
| 48 | 0.541 | 0.109 | 0.131 | 0.172 | 1.198 | 1.190 |
| 64 | 0.573 | 0.049 | 0.063 | 0.094 | 1.205 | 1.183 |

- 局域量 g01(ρ) 和 k ≥ 0.5 的 G(k) 在所有尺寸含外推的 48、64 都与模拟一致，粒子坍缩比例也一致。
- 最长波的记忆随 L 增大丢失。训练步数 1000 → 5000 和深度 1 块 → 2 块都只带来小幅改善。
  loss_bins 在 L=16 和 64、三个 run 都显示 σ > 4 处真 X0 与打乱 X0 的 loss 无差别。
  L=8 的最低模 k=0.79 在 σ≈1 决定所以没事，L=64 的 k=0.1 要在 σ≈10 决定，而 σ ≳ 4 的 score 网络还没学到。
  原因估计有两层。一是 ε 参数化下大 σ 处的可学信号很弱，振幅 A、波矢 k 的密度模给出的 ε̂ 约
  `σ k A exp(-k²σ²/2)`，loss 降幅只有百分之一量级，淹没在方差为 2 的 DSM 目标噪声里，梯度信噪比低。
  二是粗层只有 1 到 2 跳（半径 4），做不了 σ≈10 尺度的密度平滑。可选的下一步是对大 σ 加权（或改预测
  `σ ε̂` 这类 x0 形式的目标）、加训练步数、给粗层加跳数或加第四层级。
- 2 块/级的第一次训练在第 3230 步前向溢出成 inf（矢量通道残差流无归一化，高密度团簇事例上指数增长），
  加了 `vnorm`（节点内按通道均方范数的等变缩放，只进消息和门控）后 5000 步无 NaN。
- L=64 采样 256 个事例（210 万粒子节点、1260 万条边）需要静态形状编译，动态形状下 inductor 用 32 位
  偏移，消息张量 2.4 GB 越界报 `CUDA driver error: invalid argument`。静态编译约 100 s，之后每个 Heun 步 1.1 s。
