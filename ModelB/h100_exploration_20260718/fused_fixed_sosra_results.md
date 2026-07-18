# L=128 固定步长 SOSRA 融合原型结果

## 边界与复现

- 生产文件 `../ModelB.jl` 未修改；测试后 SHA256 仍为
  `5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3`。
- 原型只覆盖本次目标：`Float32`、`L=128`、状态形状
  `(128,128,Nensemble)`、常数加性扩散、`adaptive=false`。
- 正式测试：`Nensemble=1024`、`dt=0.002f0`、`t_end=2.0f0`，即 1000 步。
- 代码：`fused_fixed_sosra.jl`。
- 单步验证：
  `julia --project=. h100_exploration_20260718/validate_fused_fixed_sosra.jl`
- 性能测试：
  `julia --project=. h100_exploration_20260718/benchmark_fused_fixed_sosra.jl full 3 1024 264 512 528 library_philox`

## 实现

每一步只保留五个主要 GPU 操作：

1. 一个持久的显式 cuRAND `PHILOX4_32_10` generator，一次填充连续的
   `(128,128,Nensemble,4)` buffer，四个 slab 分别是 W.x、W.y、Z.x、Z.y；
2. 一个 kernel 同时形成守恒的 `dW` 和 `dZ`，并融合 `sqrt(dt)`；
3. `stage1` 同时算 `k1=F(u)` 和 `H01`；
4. `stage2` 同时算 `k2=F(H01)` 和 `H02`；
5. `stage3` 计算 `k3=F(H02)` 后直接原位更新 `u`。

这消除了通用 SOSRA 路径中的三个常数 diffusion fill、五个 stage
broadcast、RSwM bookkeeping、四次独立 RNG 调用和一个 divergence kernel。
显式 generator 在 workspace 构造时创建，绝不在时间步循环内重复创建。

## 数值验证

使用完全相同的 packed links、dW 和 dZ，将融合实现与“三次独立 direct
stencil + 朴素 SOSRA broadcast 公式”比较：

| 项目 | 最大绝对误差 |
|---|---:|
| packed dW 对 CPU 周期散度 | 0 |
| packed dZ 对 CPU 周期散度 | 0 |
| k1 | 1.3e-44（Float32 subnormal，等效为 0） |
| H01 | 0 |
| k2 | 0 |
| H02 | 0 |
| 最终 u | 0 |

验证共 11 项，全部通过。两个 ensemble 的单步守恒噪声空间和最大残差分别为
`8.34e-7`（dW）和 `8.34e-7`（dZ）。随机 buffer 的样本均值、方差为
`-0.00214387`、`0.999427`；增量方差为 dW `0.00809256`、dZ
`0.00801366`，理论值 `4*dt=0.008`。

packed 一次填充改变了相对生产路径的随机数消费顺序，因此不能要求两条完整
轨迹逐位相同；单步公式在给定相同增量时逐位相同，噪声分布和守恒结构相同。

## kernel 调优

四个自定义 kernel 均为 32 registers/thread，`maxthreads=1024`。空闲 H100
上的同步 microbenchmark：

| 操作 | 配置 | 中位时间 |
|---|---:|---:|
| packed 四分量 Philox randn | 单次 cuRAND 调用 | 0.144865 ms |
| packed 四分量 implicit default randn | 单次 cuRAND 调用 | 1.036534 ms |
| 同时构造 dW/dZ | 512 threads, 528 blocks | 0.133841 ms |
| 三个融合 stage 合计 | 1024 threads, 264 blocks | 0.430547 ms |

Philox 在这个大连续 buffer 上比 implicit default RNG fill 快约 `7.16x`。
短程全步进搜索也选择了 `1024x264` 作为三个 stage 的统一 launch，选择
`512x528` 作为双增量 launch。共享内存 stencil 没被采用：此前 direct
bitmask stencil 已经比 shared tile 更快，本原型继续沿用 direct addressing。

## 正式 L=128, Nensemble=1024, t=2 结果

计时边界前后各做一次 device synchronize，正式结果使用 `time_ns()`；未采用
Julia 1.12 在本脚本首次 `@elapsed` 调用中出现过的一次无效 subnormal 返回值。

| repetition | 秒 | us/步 | finite | maxabs(u) | max abs ensemble sum |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.668668375 | 668.668 | true | 3.8174901 | 5.569458e-4 |
| 2 | 0.668653238 | 668.653 | true | 3.8192463 | 6.637573e-4 |
| 3 | 0.673126896 | 673.127 | true | 3.7685847 | 5.493164e-4 |

中位时间为 **0.668668375 s**。相对于已经测得的：

- production sync SOSRA `5.164691 s`：约 **7.72x** 加速，时间减少 87.1%；
- direct drift + fused noise + RSwM1 async `3.271561 s`：约 **4.89x** 加速，
  时间减少 79.6%；
- 同样 direct/fused/RSwM1 且显式 Philox 的通用 solver `2.433 s`：约
  **3.64x** 加速。

三个完整运行都有限。最大 ensemble 空间和残差 `6.64e-4`，除以 16384 个
格点后对应平均场残差约 `4.05e-8`。

workspace 的逻辑数组容量约 640 MiB（六个 64 MiB 状态大小数组，加一个
256 MiB 四分量 links buffer）；连同 u 为约 704 MiB。CUDA memory pool、JIT
代码和库 workspace 会使 `nvidia-smi` 显示的进程占用更大。

## 不应混用的场景

该原型没有 adaptive rejection、NoiseProcess bridge、任意 L、Float64 或状态
相关 diffusion。它是此次固定步长生产负载的专用性能上界，不是通用 SOSRA
替代品。若以后恢复 adaptive 步长，必须回到完整 NoiseProcess/SOSRA 语义并
重新验证。

CUDA Graph 没有纳入正式结果。对一个时间步的小 graph 反复 replay 可能重复
捕获时的 cuRAND offsets，物理上不可接受；只有一次性捕获完整 1000 步并只
replay 一次才是安全候选，而当前每步已经约 0.669 ms、launch overhead 不是
主要项，收益上限很小。
