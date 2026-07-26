# Random-number data path 测试结果

测试日期：2026-07-26。环境为 Miyabi-G GH200、Julia 1.12.6、
CUDA.jl 6.2.0。没有使用 FFT/Fourier acceleration，`solver/` 与
`references/` 均未修改。

## 结论

可以借鉴“随机数靠近更新 kernel”，但对当前实际形状，首先应该修复的不是
fusion，而是 CUDA.jl `LibraryRNG.randn!` 的 padding 路径。

`L=32,Nensemble=5120` 每步需要
`4×32²×5120 = 20,971,520` 个 link normals。CUDA.jl 6.2.0 会把非 2
次幂的数组临时补到 `33,554,432`，每步多生成 60% 的数，再把前
20,971,520 个数 D2D copy 回 80 MiB 的目标数组。cuRAND 本身对
Philox pseudorandom normal generation 只要求 `n` 为偶数。
对应实现位于当前环境
`~/.julia/packages/CUDA/kig1B/lib/curand/src/random.jl:89` 的
`inplace_pow2`；cuRAND 的偶数要求见
[NVIDIA Host API 文档](https://docs.nvidia.com/cuda/curand/group__HOST.html)。

完整 SOSRA step 的实测结果是：

| 方法 | ms/step | 相对当前 | 是否保留旧 seed 轨迹 | link workspace |
|---|---:|---:|---|---:|
| 当前 CUDA.jl `LibraryRNG.randn!` | 0.26670 | 1.000× | 是 | 80 MiB + 128 MiB 临时 |
| 预分配 2 次幂 buffer | 0.21774 | 1.225× | 是 | 128 MiB |
| 公开 `CUDA.RNG` | 0.20039 | 1.331× | 否 | 80 MiB |
| 直接偶数长度 cuRAND | 0.19276 | 1.384× | 否 | 80 MiB |
| `32×8` tiled stateless Philox | 0.18591 | 1.435× | 否 | 0 |

因此，直接偶数长度 cuRAND 已取得 tiled fusion 约 92% 的 wall-time
reduction，而代码仍保留现有的 link buffer 和三个 SOSRA stage kernel。
若只使用公开 API，`CUDA.RNG` 仍有 1.33×。只有追求最后约 3.7% 时，才值得
承担 tiled kernel 的复杂度和 CUDA.jl 内部 primitive 依赖。

若采用 fusion，有效设计是：

1. 用无状态 Philox4x32-10 定义
   `noise = Philox(global_link_index, physical_step, seed)`；
2. 一个 `32×8` CUDA block 正好让一个 warp 对齐格点的一整行；
3. 每条 block 内 link 只生成一次并写入 shared memory；
4. 第一个 warp 生成左侧和下侧 halo；
5. stage 1 直接消费 shared-memory link，不建立全局 `links` 数组。

对实际的 `L=32, Nensemble=5120`，该原型把 5000 步
`t=10,dt=0.002` 从约 `1.338 s` 降到 `0.932–0.952 s`。公开
`CUDA.RNG` 和直接 cuRAND 分别为约 `1.008 s` 和 `0.969 s`。

这个收益有明显的形状依赖。对于已有高吞吐的 8,388,608-element case，
四个 link 场正好有 `33,554,432` 个数，本来就是 2 的幂，因此非 fusion
方案没有收益：

| L | ensembles | 当前 | 公开 RNG | 直接 cuRAND | tiled |
|---:|---:|---:|---:|---:|---:|
| 32 | 5120 | 0.26670 | 0.20039 | 0.19276 | 0.18591 |
| 32 | 8192 | 0.29757 | 0.31175 | 0.29760 | 0.28983 |
| 128 | 512 | 0.29838 | 0.31289 | 0.29794 | 0.29033 |
| 256 | 128 | 0.29973 | 0.31413 | 0.30040 | 0.29278 |

单位为 ms/step。padding 修复适合当前 `run_l32_references.jl` 的形状，
不是所有 lattice 规模上的统一优化；tiled 在已经对齐的规模上也只有约
2–3%。

## padding 成本拆分

仅测实际 20,971,520-element link fill：

| 方法 | 实际生成数 | ms/fill |
|---|---:|---:|
| 当前 CUDA.jl wrapper + copy | 33,554,432 | 0.12110 |
| 预分配 2 次幂、不 copy | 33,554,432 | 0.06839 |
| 公开 `CUDA.RNG` | 20,971,520 | 0.05337 |
| 直接 `curandGenerateNormal` | 20,971,520 | 0.04419 |

这也解释了为什么较小的 `L=32,Nensemble=5120` 反而看起来与
8,388,608-site case 几乎一样慢。

## 为什么不能直接在 kernel 中调用随机数

Model-B noise 是 link field 的离散散度。相邻 site 必须对公共 link 使用逐位
相同的随机数，才能得到

`η(x)=ξx(x)-ξx(x-e₁)+ξy(x)-ξy(x-e₂)`

以及逐配置的 `Σx η(x)=0`。普通 kernel-local `rand()` 的结果依赖 thread
调度，邻居无法可靠复现同一条 link；有状态 RNG 还需要保存、更新和同步
state。

counter-based Philox 把随机数变成 link 索引和 step 的纯函数，因此无需保存
per-thread state，也不依赖 block 调度顺序。新的 seed 在原型内部可逐位复现，
但不会复现旧 cuRAND 数据集的具体随机轨迹。

## 不同写法的实测

下表使用已经对齐的 `L=256,Nensemble=128`，即 8,388,608 个 site：

| 方法 | ms/step | 相对当前 | stage-1 registers | 全局 link workspace |
|---|---:|---:|---:|---:|
| 当前 cuRAND + links | 0.3005 | 1.000 | 40 | 128 MiB |
| 单独 CUDA.jl GPUArrays RNG | 0.3154 | 0.953× | 40 | 128 MiB |
| 每个 site 直接重算 Philox | 0.3173 | 0.947× | 78 | 0 |
| `32×8` tiled Philox | 0.2938 | 1.023× | 57 | 0 |
| 双 buffer、双 stream overlap | 0.2992 | 1.004× | 40 | 256 MiB |

所以在已经对齐的形状上：

- 仅把 cuRAND 换成 CUDA.jl 的独立填充 kernel 没有收益；
- 直接 fusion 会为当前、左侧和下侧 origin 执行三次 Philox，寄存器从 40
  增到 78，反而约慢 5%；
- tile 复用把寄存器降到 57，并避免全局 link 写入与回读；
- RNG 与 stage 2/3 使用双 stream overlap 没有可测收益，因为这些 kernel
  会竞争相同的 GPU 资源，而且还要多保留一份 links。

对 `L=32,Nensemble=5120`，tiled 版本还省去约 80 MiB link workspace。

## 数值与统计验证

`test_fused_philox.jl` 检查了：

- 四个输出分量的均值、方差、偏度和 excess kurtosis；
- 分量之间的相关系数；
- 离散 noise 的理论协方差
  `Var[η(x)]=4, Cov[η(x),η(x±eμ)]=-1`，非近邻为零；
- 每个 ensemble 的 noise 空间和；
- 1000 步后的守恒均值；
- 相同 seed 的逐位复现；
- 公开 `CUDA.RNG` 的守恒与逐位复现；
- 预分配 padded buffer 与当前 wrapper 连续调用的逐位相同输出；
- 直接偶数长度 cuRAND 在第一次调用时与当前 wrapper 前缀逐位相同，并在
  后续按实际消费长度推进；
- tiled 与非 tiled stateless 实现的逐位相同轨迹；
- 双 stream cuRAND 与当前串行 cuRAND 的逐位相同轨迹。

随后对实际的 HMC/Gaussian 初态各做了两份独立 tiled-noise 演化，并在
`t=0:0.1:10` 比较有限时间分布：

| 初态 | 方法 | 平均六维联合 W1 | 采样噪声底 | action-density 平均 W1 | 采样噪声底 |
|---|---|---:|---:|---:|---:|
| HMC | 当前 cuRAND | 0.01641 | 0.01660 | 0.02324 | 0.02351 |
| HMC | 公开 `CUDA.RNG` | 0.01654 | 0.01660 | 0.02428 | 0.02351 |
| HMC | 直接 cuRAND | 0.01632 | 0.01660 | 0.02365 | 0.02351 |
| HMC | tiled A / B | 0.01664 / 0.01639 | 0.01660 | 0.02327 / 0.02457 | 0.02351 |
| Gaussian | 当前 cuRAND | 0.02346 | 0.02430 | 0.02715 | 0.02563 |
| Gaussian | 公开 `CUDA.RNG` | 0.02324 | 0.02430 | 0.02688 | 0.02563 |
| Gaussian | 直接 cuRAND | 0.02236 | 0.02430 | 0.02590 | 0.02563 |
| Gaussian | tiled A / B | 0.02375 / 0.02386 | 0.02430 | 0.02593 / 0.02682 | 0.02563 |

没有看到相对于两份细步长 `SOSRA, dt=0.0005` 参考的系统偏移。逐配置
`M` 最大漂移不超过 `7.2e-7`。

## 移入 solver 前的注意点

建议按需求选择：

1. 必须逐位复现旧数据：预分配 padded buffer，速度约 1.225×；
2. 只接受公开 CUDA.jl API：换为 `CUDA.RNG(seed)`，速度约 1.331×；
3. 接受一个很薄的版本相关调用：直接以偶数长度调用
   `curandGenerateNormal`，速度约 1.384×，是当前最合适的性能/复杂度折中；
4. 确实需要最后约 3.7% 和 80 MiB link memory：再考虑 tiled fusion。

直接 cuRAND 原型仍调用了 CUDA.jl 内部的 stream update helper；tiled 原型
还调用了内部 stateless Philox/Box--Muller primitives。这两条都不应该不经
版本固定和测试就直接复制到 solver。公开 `CUDA.RNG` 没有这个 API 风险。

除 padded 方案外，新 RNG 都会改变同一 seed 对应的具体轨迹，但不会改变
noise law、SOSRA、`dt` 或空间离散。完整数据位于
`output/rng_fill_timings.tsv`、`output/shape_timings.tsv`、
`output/variant_timings.tsv`、`output/l32_distribution_summary.tsv` 和
`output/l32_rng_stats.jld2`。
