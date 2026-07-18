# L=128 Model-B 固定 SOSRA 的 H100/GH200 完整优化结论

## 结论

在本次固定目标

```text
L=128, Nensemble=1024, Float32
SOSRA(), adaptive=false, dt=0.002, t=2
```

上，最快且已经通过 HMC 平稳性验证的是实验目录中的专用 fixed-SOSRA：

- 持久显式 cuRAND Philox；
- 一次填充 W.x/W.y/Z.x/Z.y 四个连续分量；
- 一个 kernel 同时构造守恒 dW/dZ；
- 三个 kernel 分别融合 SOSRA 的三个 drift stage 和状态更新；
- launch：stage `1024 threads x 264 blocks`，noise `512 x 528`。

持续功率测试中，复用已分配 workspace 的 `t=2` 热积分中位时间为
**0.6734 s**，当前 production API 的同步基线为 **5.1577 s**，即
**7.66x** 加速。把 marker 内的状态重置、重新设 seed、有限性检查和输出也计入，
两者分别为 **0.6764 s/solve** 和 **5.1905 s/solve**，对应 **7.67x**；因此
结论不依赖内层计时边界。GPU 能耗从约 **2151 J/solve** 降到
**461 J/solve**（-78.6%）。它的滚动功率中位/p95/峰值为
**698.7/699.6/699.95 W**，已经达到该节点实际 enforced 的 700 W 限制。

`ModelB.jl` 从开始到结束均未修改，最终 SHA256 为：

```text
5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3
```

## 完整性能阶梯

正式功率轮均先在 marker 外完成同尺寸、同长度预热并保留 CUDA memory
pool；marker 开销仅 0.10--0.12 s/约 20 s 窗口。表中是热积分中位时间：
production 的计时包含每次 `SDEProblem`/noise/workspace/state-copy 构造，专用版
计时复用 workspace，且把 `copyto!`/重新设 seed 放在内层 timer 外。上段给出的
marker 总墙钟/solve 是包含这些外围操作的交叉检查；一次性的专用 workspace
分配不在正式窗口内，推荐的重复轨迹用法本来就应复用它。

| 路径 | t=2 | 加速 | 平均/中位 GPU 功率 | GPU 能耗/solve |
|---|---:|---:|---:|---:|
| production sync RSwM3 | 5.1577 s | 1.00x | 414.5/425.2 W | 2151 J |
| production drift, RSwM1 async | 4.3224 s | 1.19x | 424.5/435.8 W | 1845 J |
| direct drift + fused noise + RSwM1 | 3.2853 s | 1.57x | 413.9/424.3 W | 1369 J |
| persistent packed Philox + generic SciML | 2.4266 s | 2.13x | 534.9/549.4 W | 1308 J |
| 4-way Philox ensemble shards | 2.3007 s | 2.24x | 未单独采样 | 未单独采样 |
| specialized five-operation fixed SOSRA | **0.6734 s** | **7.66x** | **681.3/698.7 W** | **461 J** |

专用版每秒完成约 `2.49e10` lattice-site steps；基线约 `3.25e9`。

## 为什么原来只有约 425 W

基线不是“大部分 GPU 都闲着”：正式数据中 GRACT 93.4%、SM active
81.8%，SM/显存时钟一直是最大 1980/2619 MHz。问题是 instruction mix：

| 路径 | SM active | Occupancy | DRAM active | FP32 active | INT active |
|---|---:|---:|---:|---:|---:|
| production sync | 81.8% | 52.3% | 23.5% | 2.1% | 77.6% |
| packed Philox generic | 93.0% | 72.7% | 39.6% | 4.3% | 84.5% |
| specialized fixed SOSRA | **94.3%** | **74.4%** | **68.6%** | **15.2%** | 47.9% |

原实现主要受整数地址/分支、随机数、全数组 broadcast 和同步限制，低 FP32、
低 DRAM 活动不会产生 H100 的最高功耗。专用版把工作压成连续且更密集的
memory+FP32 kernel，才真正触到 power cap。其 cap counter 在 20.29 s 窗口中
增加 19.21 s，SM 时钟因功率限制从 1980 降到约 1921 MHz；这不是温度限制。

`nvidia-smi` 显示 requested/default/max 为 900 W，但 `nvidia-smi -q` 与 DCGM
的实际 current/enforced limit 都是 **700 W**；GH200 module limit 为 1000 W。

## 数值与物理验证

1. callback 去同步、RSwM1、direct drift、fused divergence/scale：相同初态与
   GPU seed 连续十步后，完整 `(128,128,1024)` 终场与 production **逐元素完全
   相同**，max/RMS difference 都是 0。
2. 专用 SOSRA：给定完全相同 dW/dZ，与三次独立 direct drift 加朴素 SOSRA
   公式的 11 项单步验证全部通过，最终场差为 0。
3. Philox 正态检验使用 33,554,432 个样本：均值在 `4.4e-4` 内、方差
   `[0.9992,1.0008]`、分量相关绝对值 `<2.9e-4`；守恒散度方差约 4，符合理论。
4. L128 HMC ground truth：512 个未去均值 HMC 构型、`t=5`、两次独立重复。
   专用版 action 相对漂移 `+0.000306%`（paired `z=0.004`），phi2/phi4/
   最近邻漂移的 `|z|<0.60`，全部与零相容；最大磁化密度漂移 `5.97e-8`。
   这一稳定性结论只覆盖 `kappa=0.2705`、`lambda=0.022`、`gamma=T=1`、
   `dt=0.002`，不能直接外推到其他参数或步长。
5. production 自带 `checks/test_sosra.jl` 最终仍为 **10/10 passed**；完整
   终端输出保存在 `production_test_raw.log`。

专用版因常数 diffusion 的代数合并改变了少量 Float32 运算顺序。与通用 SciML
逐场对比，一步 max/RMS difference 为 `4.77e-7/8.39e-8`，十步为
`1.43e-6/2.38e-7`；HMC 检验直接确认这些舍入差没有可见物理偏差。

## 各种尝试的取舍

- **L128 bit-mask direct indexing**：drift `0.3067 -> 0.0939 ms`，约 3.27x；
  是正确方向。
- **shared-memory tile**：端到端慢 1.05%；最佳 `32x8` tile 的 drift microbench
  慢 10.8%，`16x16` tile 慢 38%，halo 和 barrier 抵消收益。
- **噪声 divergence+scale 融合**：`0.2987 -> 0.0755 ms`，约 3.96x。
- **packing 本身**：default RNG 下完整求解 `3.275 -> 3.303 s`，没有收益；
  真正收益来自持久显式 cuRAND Philox。
- **dual stream/prefetch**：default RNG 有约 8--9% 收益；Philox 下均比简单
  单 stream 慢，因此不采用。
- **ensemble shards**：generic Philox 用 2/4 shards 约 `2.301 s`，只再快
  5.3%；8 shards 开始变慢。
- **maxregs**：production 限 64 仅约 0.5% 改善；强压到 32 反而慢 3.5x。
  direct kernel 天然只有 32 registers，无需强制。
- **cuFFT spectral drift**：功率约 550 W，但单 drift `0.7549 ms`，约比 direct
  慢 8x；这是“高功率但低效率”的明确反例。
- **FP32/TF32 GEMM 控制**：约 689 W，确实触到同一 700 W cap，证明监控和
  硬件没有问题。
- **CUDA Graph**：没有用于正式结果。短 graph 反复 replay 可能重复捕获时的
  cuRAND offset；一次捕获完整 1000 步收益上限又很小。
- **Philox dynamic ordering**：raw fill 快约 7%，专用 full solver 中位数
  `0.6579 s`，比同轮 default ordering 的 `0.6738 s` 快 2.36%；但跨硬件随机
  序列可复现性较弱，且没有完整统计/HMC/功率正式轮，因此只保留为探索项，
  不作为默认。

## 推荐使用方式

当前不改生产文件，直接从实验目录调用：

```julia
using CUDA
include("h100_exploration_20260718/fused_fixed_sosra.jl")
using .FusedFixedSOSRA128

u = copy(u0)  # CuArray{Float32,3}, shape (128,128,Nensemble)
workspace = allocate_workspace(
    u; rng_kind=:library_philox, seed=20260718
)
params = ModelBParameters(
    gamma=1.0f0,
    temperature=1.0f0,
    kappa=0.2705f0,
    lambda=0.022f0,
)

solve_fixed_sosra!(
    u, workspace, params;
    dt=0.002f0,
    t_end=2.0f0,
    threads=1024,
    blocks=264,
    noise_threads=512,
    noise_blocks=528,
)
CUDA.synchronize()
```

重复轨迹时复用 workspace，并在每条轨迹前显式调用
`seed_packed_rng!(workspace, seed)`；需要相同初态时再 `copyto!(u,u0)`。

适用边界必须严格遵守：该原型只支持 `L=128`、`Float32`、三维
`(x,y,ensemble)` 状态、常数加性 diffusion、固定步长且无 rejection。它不提供
adaptive bridge、任意 L、Float64 或 SciML 的完整 solution/saveat 接口。若以后
恢复 adaptive，必须回到完整 NoiseProcess/SOSRA 语义并重新验证。

## 文件入口

- `fused_fixed_sosra.jl`：推荐实验实现。
- `fused_fixed_sosra_results.md`：公式、资源和单步/full benchmark。
- `power_results_report.md` / `power_summary.csv`：正式功率、能耗、pipeline。
- `hmc_validation.md` / `fused_vs_hmc.tsv`：HMC 平稳性。
- `RNG_REPORT.md`：生成器、stream、prefetch 结果。
- `kernel_results.md`：direct/shared/fused/FFT 内核结果。
- `ensemble_shards_results.md`：1/2/4/8 stream 结果。
- `maxregs_results.md`：register cap 搜索。
- `trajectory_validation.md`：production 同轨迹逐元素验证。
- `production_test_raw.log`：最终 production 10/10 回归输出。

另有两个仅影响测试流程的冷启动现象：新 Julia 进程中曾观察到 production
896-thread drift 对全新未初始化输出的第一次 launch 异常，以及 Julia 1.12
围绕第一次长 GPU loop 的 host timer subnormal 返回。所有正式结果都做了完整
预热、显式同步并采用 marker/DCGM 或 CUDA event 交叉验证；生产测试 10/10
通过。正式合并任何代码前，仍建议保留一次同尺寸 preflight 和有限性检查。
