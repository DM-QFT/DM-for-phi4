# Model B 性能排查结果

测试环境：`mg0828`，NVIDIA GH200 120GB，Julia 1.12.6，CUDA.jl 6.2.0。
测试日期为 2026-07-26。`solver/` 和 `references/` 均未修改。

## 结论

当前 `run_randn.jl` 的慢主要来自计算量本身，而不是 SciML wrapper、Julia
循环、`L=256` 的索引或 GPU 未饱和：

1. `t=100, dt=0.002` 是 50,000 个时间步。
2. 每个 fixed SOSRA 步生成四个 `L×L×Nensemble` 高斯 link 场，并计算
   三次 13 点 drift stencil。
3. 当前数组有 8,388,608 个元素，因此一次 `t=100` 演化共生成约
   `1.68e12` 个高斯数并计算约 `1.26e12` 个 site drift。
4. 热启动后实测为 `0.299 ms/step`，所以 50,000 步预计 `14.96 s`。
5. 第一次调用还有约 `4.27 s` 的 CUDA kernel/SciML 方法编译与初始化。
   实际 `run_randn.jl` 报告的首次 solve 时间是 `19.51 s`，与
   `14.96 + 4.27 s` 一致。

整个新 Julia 进程的实测 wall time 是 `35.16 s`。其中还包括加载 CUDA 和
SciML、建立 CUDA context，以及 solve 之后首次编译 `all(isfinite, ...)` 和
`mean(...; dims=...)` 的 GPU reduction。这些冷启动成本不属于热 solver
throughput，但短脚本中会很显眼。

实际调用链是 `run_randn.jl -> ModelB_fixed_sciml.jl ->
ModelB_fixed.jl`。生产入口没有调用通用的 `solver/ModelB.jl`，所以通用版本
callback 中的逐次 `CUDA.@sync` 不是这次运行的慢因。

`run_randn.jl` 的 `save_start=true` 只额外保留一个 32 MiB 初态；默认
algorithm 也只在 solve 开头分配一次约 256 MiB workspace。它们会进入冷启动
成本，但都不在 50,000 步内重复。

## 热吞吐与形状

每项取 1,000 步热启动中位数：

| L | ensembles | 元素数 | ms/step | ns/(element·step) |
|---:|---:|---:|---:|---:|
| 64 | 2048 | 8,388,608 | 0.2956 | 0.035 |
| 128 | 512 | 8,388,608 | 0.2983 | 0.036 |
| 256 | 128 | 8,388,608 | 0.2991 | 0.036 |

相同元素数下三种形状只相差约 1.2%，所以当前 `L=256` 的 periodic indexing
不是瓶颈。

`L=256` 的规模扫描为：

| ensembles | 元素数 | ms/step | ns/(element·step) |
|---:|---:|---:|---:|
| 8 | 524,288 | 0.0274 | 0.052 |
| 16 | 1,048,576 | 0.0425 | 0.041 |
| 32 | 2,097,152 | 0.0835 | 0.040 |
| 64 | 4,194,304 | 0.1584 | 0.038 |
| 128 | 8,388,608 | 0.2992 | 0.036 |

小数组会受固定 launch 成本影响；当前 `Nensemble=128` 已在高吞吐区。

## 每步 GPU 时间

CUDA profiler 对 `L=256, Nensemble=128` 的 20 个热步骤给出：

| 部分 | 每步时间 | 每步 GPU 时间占比 |
|---|---:|---:|
| stage 1：drift、link divergence、四个 stage 输出 | 102.48 μs | 34.8% |
| stage 2：第二次 drift 与 stage 更新 | 75.61 μs | 25.7% |
| Philox：四个 link 场 | 68.31 μs | 23.2% |
| stage 3：第三次 drift 与最终更新 | 48.22 μs | 16.4% |
| 合计 | 294.62 μs | 100% |

热 solver 是 `299.06 μs/step`。kernel 合计占 98.5%，剩余 launch、Julia
循环和 SciML 边界约 1.5%。因此 CUDA Graph、减少 wrapper 调用或改写外层
Julia 循环不会带来明显收益。

独立的 direct fixed 路径为 `299.064 μs/step`，SciML wrapper 路径为
`299.076 μs/step`；差异不可测。

同一基准使用 `julia -O3` 得到 `299.88 μs/step`，与默认优化级别的差异在
运行波动内；缺少 `-O3` 也不是慢因。

Nsight Compute 的硬件计数器为：

| kernel | DRAM throughput | compute throughput | achieved occupancy | registers/thread |
|---|---:|---:|---:|---:|
| stage 1 | 68.5% | 52.2% | 72.1% | 40 |
| stage 2 | 70.5% | 56.0% | 89.0% | 32 |
| stage 3 | 34.0% | 63.6% | 75.9% | 32 |

stage 1/2 主要受 stencil 全局内存流量限制，stage 3 更偏计算受限。当前
264-block cap 正好给 132 个 SM 两个 block，occupancy 并不低。

## SOSRA 本身的成本

独立目录中的 Euler--Maruyama workload 使用同一 spatial drift，但每步只做
一次 drift 和两个 link-noise 场。它只用于计算量对照，不是生产求解器：

| 方法 | ms/step | 最大 mean-field 漂移 |
|---|---:|---:|
| fixed SOSRA | 0.299064 | `1.64e-8` |
| benchmark Euler | 0.095743 | `1.26e-8` |

相同步长下 SOSRA 的工作量实测是该单 stage 路径的 `3.124×`。这个差异与
三次 drift、两套 Wiener increment 的理论工作量一致。能否换低 stage 方法
取决于所需的 weak/strong order、稳定性和目标 observable 的步长误差，不能
只依据速度决定。

## 与 `references/` 的区别

### `chattopadhyay-modelB`

该实现不是同一个 SDE integrator。它在三维 `Float64` lattice 上做
Kawasaki 型 Metropolis pair exchange：

- 每个 proposal 对两个相邻 site 做 `+q/-q`，再作 Metropolis 接受；
- 一个 sweep 是四种 coloring 乘三个方向，共 12 个 GPU kernel；
- microscopic step 标成 `Δt=0.04`，是当前 `0.002` 的 20 倍；
- 参数、维数和 time normalization 也不同。

它与当前 overdamped Langevin SOSRA 属于同一 Model-B 动力学普适类，但
Monte Carlo sweep 不是当前 SDE 的一个大步长离散。因此不能比较相同 `t`
标签所需 wall time；应比较达到同一统计误差的物理 observable 和有限步长
外推。

### `phi4lattice`

它的 2D conservative force stencil 与当前 drift 的
`-Δ²φ + m²Δφ + gΔ(φ³)` 结构相同，但程序同时演化 `φ` 和 conjugate
momentum `π`。每个 leapfrog substep 对 `π` 做一次 conservative force
update，并在 Langevin 模式中对 `π` damping、生成 vector noise，再更新
`φ`。其命令行默认把一个单位输出间隔分成 160 步，即内部步长
`0.00625`，也是当前 `0.002` 的 3.125 倍。这不是当前直接对 `φ` 积分的
一阶 overdamped SOSRA。

所以参考 C++ kernel 的“一步一次 stencil”不能直接替代当前 SOSRA 的
“一步三次 stencil”；上面的单-stage workload 已量化了这项算法差异的理想
速度量级。

## 当前入口的预计成本

- `run_randn.jl`：50,000 步，热计算约 `14.96 s`，首次 solve 约
  `19.2–19.5 s`。
- `run.jl`：每组参数有 2,000,000 步，同样 8,388,608 个元素，热计算约
  `596.5 s = 9.94 min`。两种初态乘三个 `κ` 共六组，纯 solver 约
  `59.7 min`。
- `run.jl` 每组还保留初态加 100 个 32 MiB snapshot，约 3.16 GiB，并在
  solve 后传回 CPU、写入 JLD2。这不解释 solver 内的 10 分钟，但会增加
  整个脚本的 wall time 和内存/I/O。

## 后续优化的有效优先级

1. 先对实际训练 target/observable 做 `dt=0.002, 0.005, 0.01, 0.02`
   的步长误差与稳定性检查。运行时间严格近似正比于 `1/dt`，这是最大杠杆；
   参考程序的 `0.04` 不能作为当前 SDE 的正确性依据。
   `dynamic_exponent/` 已有较大步长对临界指数的相容性证据，但它不能自动
   证明 `κ=0.28` 的逐配置训练 target 也满足所需误差。
2. 如果只要求平衡分布或 weak observables，独立验证低 stage integrator。
   单 stage 的硬件工作量上限约快 `3.1×`，但最终收益取决于它允许的 `dt`。
3. 若必须保留 SOSRA，优化重点只能是三次 stencil 的数据复用和 Philox/link
   noise 的约 23% 成本。外层 Julia/SciML 优化的收益上限只有约 1.5%。
4. 对许多短任务，把多个 case 放进同一 Julia 进程，避免每次重复 CUDA
   context、kernel 和 reduction JIT。长达数十万步的任务中这不是主因。

## 实空间有限时间分布补充测试

未使用 Fourier acceleration 的 Euler、stochastic Heun 与 SOSRA 多步长
对照见 [`realspace_distribution/results.md`](realspace_distribution/results.md)。
在现有 `L=32, t≤10` 的 HMC/Gaussian 初态上，实空间 Heun `dt=0.002`
与细 SOSRA 参考的逐时统计差异处于采样噪声量级，同时约快 `1.70×`；
Euler `dt=0.002` 已有显著 action-density 偏差，不适合作为逐时分布 target。

## Random-number fusion 补充测试

counter-based Philox 与 `32×8` shared-memory tile 的独立测试见
[`fused_rng/results.md`](fused_rng/results.md)。它不改变 SOSRA、`dt` 或空间
离散。进一步定位发现，实际 `L=32,Nensemble=5120` 的四个 link 场不是
2 的幂；CUDA.jl 6.2.0 每步把 20,971,520 个数补到 33,554,432，再做一次
D2D copy。直接按合法偶数长度调用 cuRAND 已可由 `0.2667 ms/step` 降到
`0.1928 ms/step`（1.384×）；公开 `CUDA.RNG` 为 `0.2004 ms/step`
（1.331×）。tiled fusion 为 `0.1859 ms/step`（1.435×），只比直接 cuRAND
再快约 3.7%。三种新随机路径的逐时统计均处于细参考的采样噪声范围。
