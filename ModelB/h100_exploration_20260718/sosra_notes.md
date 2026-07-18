# 当前 ModelB 的 SOSRA 源码审计与融合边界

## 审计范围

- 生产文件只读：`ModelB.jl`，审计时 SHA256 为
  `5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3`。
- 本机包版本：Julia 1.12.6，StochasticDiffEq 7.1.1，
  StochasticDiffEqHighOrder 2.1.1，StochasticDiffEqCore 2.0.2，
  DiffEqNoiseProcess 5.33.1，SciMLBase 3.35.1。
- SOSRA 的实际 mutable/in-place 步进代码位于：
  `/work/gw97/w97000/.julia/packages/StochasticDiffEqHighOrder/Xt4IG/src/perform_step/sra.jl:238-309`。
- SOSRA 系数位于：
  `/work/gw97/w97000/.julia/packages/StochasticDiffEqHighOrder/Xt4IG/src/caches/sra_caches.jl:171-196`。
- NoiseProcess 固定步长下仍使用的 RSwM3 路径位于：
  `/work/gw97/w97000/.julia/packages/DiffEqNoiseProcess/qjOvV/src/noise_interfaces/noise_process_interface.jl:83-236`。
- 本文只做源码推导和实现设计，没有运行 GPU。

## 1. 当前方程与噪声的准确含义

记 `ModelB.jl` 的 stencil drift 为 `F(u)`：

\[
F(u)=\gamma^{-1}\left[2\kappa(-\Delta^2u)+m_0^2\Delta u
                  +4\lambda\Delta(u^3)\right].
\]

代码中的离散 Laplacian 约定为

\[
(\Delta u)_{x,y}=u_{x-1,y}+u_{x+1,y}+u_{x,y-1}+u_{x,y+1}-4u_{x,y}.
\]

`modelb_noise_dist!` 每调用一次，先生成两个独立标准正态 link 场，
再形成其离散散度

\[
(D\xi)_{x,y}=\xi^x_{x,y}-\xi^x_{x-1,y}
             +\xi^y_{x,y}-\xi^y_{x,y-1},
\]

最后返回 \(\sqrt{|h|}D\xi\)。因此其空间协方差为

\[
\operatorname{Cov}(D\xi)=-\Delta,
\]

即中心为 4、四个最近邻为 -1。周期格点上的空间和为零，所以这是守恒噪声。
扩散回调返回常数

\[
\sigma=\sqrt{2T/\gamma}.
\]

当前 SDE 可写为

\[
du=F(u)\,dt+\sigma\,d\mathcal W,
\qquad \operatorname{Cov}(d\mathcal W)=(-\Delta)dt.
\]

SciML 在这里走 `is_diagonal_noise(prob)==true` 分支，是因为没有提供
`noise_rate_prototype`，且 `g` 是与状态同形的逐元素乘法。这里的
“diagonal”只描述 `g .* dW` 的乘法布局；`dW` 本身由自定义 NoiseProcess
生成，仍具有上述空间相关性，并不意味着各格点独立。

## 2. SOSRA 的当前一步公式

令当前状态为 \(u_n\)，步长为 \(h\)。NoiseProcess 对 `W.dW` 和
`W.dZ` 分别调用一次 `modelb_noise_dist!`，故得到两个相互独立、同分布的
守恒增量

\[
\Delta W_n=\sqrt h D\xi_n,
\qquad
\Delta Z_n=\sqrt h D\zeta_n.
\]

SOSRA 源码首先形成

\[
\chi_n=\frac12\left(\Delta W_n+\frac{\Delta Z_n}{\sqrt3}\right).
\]

`dZ` 不是可删除的冗余噪声。它是强二阶 additive-noise 方法构造
辅助随机积分所需的第二个独立过程。令 \(k_i\) 表示未乘步长的 drift，
源代码的一步为

\[
\begin{aligned}
g_1 &= g(u_n,t+c_{11}h),          &k_1 &= F(u_n,t),\\
H_{01} &= u_n+h a_{21}k_1+b_{21}\chi_n g_1,\\
g_2 &= g(H_{01},t+c_{12}h),       &k_2 &= F(H_{01},t+c_{02}h),\\
H_{02} &=u_n+h(a_{31}k_1+a_{32}k_2)
          +\chi_n(b_{31}g_1+b_{32}g_2),\\
g_3 &= g(H_{02},t+c_{13}h),       &k_3 &= F(H_{02},t+c_{03}h),\\
E_2 &=\chi_n(\beta_{21}g_1+\beta_{22}g_2+\beta_{23}g_3),\\
u_{n+1}&=u_n+h(\alpha_1k_1+\alpha_2k_2+\alpha_3k_3)+E_2\\
&\quad+\Delta W_n(\beta_{11}g_1+\beta_{12}g_2+\beta_{13}g_3).
\end{aligned}
\]

本机 SOSRA 的 Float32 系数为：

| 系数 | 值 |
|---|---:|
| \(\alpha_1,\alpha_2,\alpha_3\) | 0.2889875, 0.68598807, 0.025024459 |
| \(a_{21},a_{31},a_{32}\) | 0.6923962, -3.1609142, 4.1609144 |
| \(b_{21},b_{31},b_{32}\) | 1.3371633, 1.442371, 1.8632741 |
| \(c_{02},c_{03}\) | 0.6923962, 1.0 |
| \(c_{11},c_{12},c_{13}\) | 0.0, 0.04124817, 1.0 |
| \(\beta_{11},\beta_{12},\beta_{13}\) | -16.792534, 17.514996, 0.27753845 |
| \(\beta_{21},\beta_{22},\beta_{23}\) | 0.4237536, 0.60103816, -1.0247917 |

在当前 ModelB 中，`diffusion!` 完全不依赖 `u` 和 `t`，所以
\(g_1=g_2=g_3=\sigma\)。公式严格按方法的实数系数可化为

\[
\begin{aligned}
q_n&=\sigma\chi_n,\\
k_1&=F(u_n),\\
H_{01}&=u_n+h a_{21}k_1+b_{21}q_n,\\
k_2&=F(H_{01}),\\
H_{02}&=u_n+h(a_{31}k_1+a_{32}k_2)+(b_{31}+b_{32})q_n,\\
k_3&=F(H_{02}),\\
u_{n+1}&=u_n+h(\alpha_1k_1+\alpha_2k_2+\alpha_3k_3)+\sigma\Delta W_n,
\end{aligned}
\]

因为 \(\sum_i\beta_{2i}=0\)、\(\sum_i\beta_{1i}=1\)。这也说明：

- `dZ` 只通过 \(q_n\) 改变两个内部 stage，但仍然不可删除；
- 最终显式随机项只含 \(\Delta W_n\)；
- 三次 `diffusion!` 的全数组 `fill!` 对这个特定模型是可消除的。

浮点注意：上述是方法层面的等价化，不保证当前 Float32 GPU 路径逐位相同。
以普通 Float32 左结合相加，`beta2` 的和是 0.0，但 `beta1` 的和是
1.0000001；而源码还经过 `@muladd`。若测试要求逐位重现，融合 kernel
必须保留源码中三个 beta 乘加的次序，不能直接替换成 `sigma*dW`。
若要求相同数值方法和统计分布，简式是正确的。

## 3. 每步调用次数和全数组 passes

假设 `adaptive=false`、没有 rejection/bridge、最终时间是 `dt` 的整数倍。
稳态每个已接受 SOSRA 步：

| 类别 | 次数 | 当前具体工作 |
|---|---:|---|
| drift `f` | 3 | 三次 radius-2 stencil kernel，计算 `k1,k2,k3` |
| diffusion `g` | 3 | 三次把 `g1,g2,g3` 填成常数 sigma |
| noise `dist` | 2 | 分别生成 `dW,dZ` |
| noise `bridge` | 0 | 固定步长、无拒绝时不调用 |
| 正态 RNG fill | 4 | 每个 `dist` 各填 `xi_x,xi_y` |
| divergence kernel | 2 | 每个 `dist` 一次 |
| noise scale broadcast | 2 | `sqrt(dt)*noise_work` |
| SOSRA broadcast | 5 | `chi2,H01,H02,E2,u` |
| RSwM3 zero fill | 2 | 清零 `W.dW,W.dZ` |
| RSwM3 add broadcast | 2 | `dW+=dWtilde,dZ+=dZtilde` |

因此当前稳态每步至少有：

- 3 个 drift stencil passes；
- 2 个 divergence passes；
- 4 个 RNG passes；
- 14 个其余全数组逐元素 passes
  （5 SOSRA + 3 diffusion + 2 scale + 2 zero + 2 add）；
- 合计约 23 个主要 GPU 全数组操作/launches，尚不含保存和框架 bookkeeping。

这与每 1000 步约 2.4 万个 kernel 的量级一致。当前回调还显式执行：

- 每个 drift 一次 `CUDA.@sync`：3 次/步；
- 每个 `dist` 在 divergence 后和 scale 后各一次：4 次/步；
- 合计 7 次显式 device synchronization/步。

NoiseProcess 在 solver 初始化时会预生成第一步噪声；每个 accepted step 的
`accept_noise!` 又以 `setup=true` 预取下一步。因此标准 solve 的 N 个时间步
通常会做约 `2(N+1)` 次 `dist`，包含终点后一个未被数值步使用的 `dW,dZ`
预取。它只造成约 `1/N` 的噪声开销，但在短 benchmark 中需要计入。

### 默认 RSwM3 为什么在固定步长仍有额外 passes

`modelb_noise_process` 没有指定 `rswm`，所以采用默认 `RSwM3`。即使
`adaptive=false` 且 stack 为空，`setup_next_step!` 仍会：

1. 清零 `dW,dZ`；
2. 把两次 `dist` 写到 `dWtilde,dZtilde`；
3. 再把 tilde 分别加回 `dW,dZ`。

对严格固定步长实验，可在独立实现中用
`RSWM(adaptivealg=:RSwM1)`。stack 为空时它直接通过
`calculate_step!` 写 `dW,dZ`，消除 2 次 zero fill 和 2 次 add broadcast；
噪声分布和调用顺序不变。不要把这个配置带到 adaptive benchmark 后仍假定
结论不变，应另做 rejection/bridge 验证。

## 4. 建议按风险排序测试的融合方案

### A. 低风险、仍使用标准 SOSRA 步进器

1. **固定步长 NoiseProcess 改用 RSwM1。**
   预期直接移除 4 个全数组 passes/步。
2. **只在实验文件去掉回调内的 `CUDA.@sync`。**
   RNG、divergence、broadcast、SOSRA stage 若全在同一个 CUDA task/stream，
   stream 顺序已经保证依赖；只在计时边界调用一次 `CUDA.synchronize()`。
   必须同时用 compute-sanitizer/逐步对照确认没有跨 stream 使用。
3. **把 `sqrt(dt)` 融入 conserved-noise kernel。**
   令 divergence kernel 直接写 `rand_vec=sqrt(dt)*Dxi`，删除
   `noise_work -> rand_vec` 的 scale pass；每个时间步可再移除 2 个 passes，
   `noise_work` 也不再需要。

这一层不改 StochasticDiffEq 的 `perform_step!`，但三次常数 diffusion fill 和
五次 SOSRA broadcast 仍存在。

### B. 特化 fixed-step additive-constant SOSRA（推荐的主要实验）

不要覆盖包内 `perform_step!(::ThreeStageSRACache)`，避免 type piracy。
在测试目录定义 `ExperimentalFixedSOSRA` 和自己的 cache/step，或写一个很短的
专用 fixed-step loop。将每步组织成三个有全局 stage 边界的 kernel：

1. `stage1!`：在同一个 kernel 中计算 `k1=F(u)`，并写
   `H01=u+h*a21*k1+b21*q`；不物化 `chi2/g1`。
2. `stage2!`：计算 `k2=F(H01)`，并写
   `H02=u+h*(a31*k1+a32*k2)+(b31+b32)*q`；不物化 `g2`。
3. `stage3_update!`：计算 `k3=F(H02)` 后立即写最终 `u_next`；
   不物化 `k3,g3,E2`，也不另启最终 update broadcast。

这样标准 SOSRA 的 5 个 broadcast 和 3 个 diffusion fills 都可消失；
仍保留算法所必需的三次 drift stencil 以及 stage 间的两个全局边界。

建议的最小数组集合是：

- ping-pong 状态 `u,u_next`；
- stage 状态 `H01,H02`；
- 后续仍需的 `k1,k2`；
- `dW,dZ`；
- RNG link workspace。

相比当前 `ThreeStageSRACache` 的 `g1,g2,g3,k1,k2,k3,E1,E2,chi2,...`，
不仅 launch 更少，常驻显存和全局内存流量也显著下降。

### C. 噪声生成的进一步实验

1. 一次填充形状为 `(L,L,Nensemble,4)` 的连续正态 buffer，四个分量分别作为
   `W.x,W.y,Z.x,Z.y`，再用一个 divergence kernel 同时写 `dW,dZ`。
   这可能把 4 次 RNG launch 降为 1 次、2 次 divergence 降为 1 次，
   但会改变随机数消费/轨迹；只保证分布等价，必须重新做统计验证。
2. 用 counter-based Philox 在 divergence kernel 内按
   `(seed,step,ensemble,x,y,direction,process)` 生成 link 正态数，省去 link
   数组和 CURAND passes。相邻格点必须用同一个 edge key 获得逐位相同的
   link 值，才能保持 telescoping conservation。每条 edge 可能被重复生成，
   正态变换成本可能使它更慢，因此只能以 benchmark 决定。
3. shared-memory tile 先生成含 halo 的 link 噪声，再做 divergence，可避免
   重复 RNG，但同步、halo 和寄存器压力都更高。

### D. launch/流水线方向

- fixed-step、固定尺寸和固定指针适合测试 CUDA Graph；若状态 ping-pong，捕获
  两步图比一步图更自然。必须确认 graph replay 时 RNG state 会前进，绝不能
  每步重放同一批随机数。
- 可测试 cooperative persistent kernel，在三个 stage 间做 grid sync；但
  cooperative launch 会限制同时驻留 block 数。若不能比三个普通 kernel 更快，
  不应仅因功率更高而保留。
- 每个 fused stage 再测试 2D shared-memory stencil tile（例如 x 连续的
  `32 x 8`、`32 x 16`，radius-2 halo）。这属于 drift 本体优化，与 SOSRA
  广播融合正交。

## 5. 不可以直接做的“融合”

- **不能删除 `dZ`，也不能令 `dZ=dW`。** 这改变 SOSRA 辅助随机积分的联合分布。
- **不能在普通 kernel 中无 grid barrier 地连续算 k1、k2、k3。**
  `F(H01)` 和 `F(H02)` 都读取邻居，前一 stage 必须全格点完成。可以用三个
  kernel、cooperative grid sync，或付出很大的扩展 stencil 重算代价；仅用
  `__syncthreads()` 不正确，因为它不能同步不同 blocks。
- **不能把 conserved noise 当作格点独立噪声。** `g` 的 diagonal trait 不会
  消除 `dW` 的空间协方差 `-Delta`。
- **不建议用 FP16/TF32/Tensor Core 只为提高功率。** radius-2 nonlinear
  stencil 不是天然矩阵乘，且临界点附近的稳定性和统计偏差风险更高。
- **不要以接近 700 W 作为接受标准。** 真正排序指标应是每个物理时间单位的
  wall time、正确性和 joule-to-solution。融合后总工作减少，平均功率可能下降，
  即使求解明显变快。

## 6. SRA1 的特化公式（仅供独立算法对照）

本机 `SRA1Cache` 的 fixed-step 源码每步调用 2 次 drift、2 次 diffusion。
当前 `g=sigma` 时可化为

\[
\begin{aligned}
K_1 &= hF(u_n),\\
Y &= u_n+\frac34K_1+\frac32\sigma\chi_n,\\
K_2 &= hF(Y),\\
u_{n+1}&=u_n+\frac13K_1+\frac23K_2+\sigma\Delta W_n.
\end{aligned}
\]

它可以特化成两个 fused drift-stage kernels，因此单步理论工作量低于 SOSRA。
但它不是与 SOSRA 等价的实现，稳定域不同；不能因功率或单步时间更好就替换
当前算法。若纳入测试，必须在 `dt=0.002` 及更大/更小 dt 上分别检验爆解率、
强/弱误差和临界点 observable。

## 7. 正确性验收顺序

1. **同一 `dW,dZ` 的单步对照**：绕过 RNG，将标准 SOSRA 与 fused step 使用
   完全相同增量；报告 max-abs、RMS、ULP 分布和逐 ensemble 空间均值漂移。
2. **多步 matched-noise 对照**：保持 RNG 消费顺序不变时，比较完整轨迹。
   若融合 RNG 改变消费顺序，则不能再用逐轨迹差作为主要判据。
3. **守恒检查**：每个 ensemble 检查
   `mean(u[:,:,e]) - mean(u0[:,:,e])`，分别测试 drift-only、noise-only、full step。
4. **统计检查**：改变 RNG 实现时，用足够 ensemble 比较结构因子、二点函数、
   action/observable 分布及 HMC ground truth，而不是只比较一个 seed。
5. **稳定性检查**：至少覆盖目标 `L=128,Nensemble=1024,dt=0.002`，并扫描
   临界步长附近，确认 fused 计算次序没有放大溢出/NaN。
6. **性能检查**：预热后交错运行 baseline/variant；计时边界显式同步；同时记录
   kernel 数、GPU active、SM/DRAM/整数/FP32 pipeline、平均/峰值功率、总能耗。

优先级建议：`RSwM1` -> 去逐 kernel sync -> noise scale fusion -> 三个
additive-constant fused SOSRA stage -> combined RNG/divergence -> shared-memory
stencil/CUDA Graph。这个顺序能先拿到容易验证的收益，再进入会改变浮点或 RNG
轨迹的方案。
