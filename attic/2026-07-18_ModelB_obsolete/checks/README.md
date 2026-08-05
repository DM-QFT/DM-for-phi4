# ModelB.jl 验证与临界点定位（2026-07-17）

> **2026-07-18，L=128 EM/SOSRA 完整实测：** 在与 `DM/2Dphi4` HMC
> 严格匹配的临界参数 `kappa=0.2705, lambda=0.022, T=1` 下，定步长
> **SOSRA 占据误差--耗时 Pareto 前沿**。同一步长 EM 每步约快 1.65 倍，
> 但同等墙钟下 SOSRA 的场强误差小 4.5--22 倍，平衡 action 偏差也小一个
> 数量级以上。建议均衡档 `SOSRA(dt=0.01)`、高精度档 `dt=0.005`。
> 详见 [`hmc_l128_em_sosra_report.md`](hmc_l128_em_sosra_report.md)。

> **2026-07-18 HMC-matched update:** `DM/2Dphi4` 的 HMC 数据对应
> `T=1, kappa=0.2705, lambda=0.022` 的格点 action，与本文下方旧
> `T≈4.45` Hamiltonian 不是同一个基准。严格匹配 HMC 的新测试再次选出
> fixed-step `SOSRA()`；完整结论、误差--耗时表与复现命令见
> [`hmc_solver_report.md`](hmc_solver_report.md)。请不要用旧 `T≈4.45` 结果直接解释
> `2Dphi4` HMC 偏差。

> **求解器更新：** 新的共同噪声临界区基准表明，在正确提供守恒辅助过程 `Z`
> 且使用 `adaptive=false` 时，**fixed-step `SOSRA()` 是已测方法中的综合最优选择**；
> 推荐 `dt=0.01`（均衡）或 `dt=0.005`（高精度）。未修改的原始 `ModelB.jl`
> 仍只应使用 fixed-step EM。详见 [`critical_solver_report.md`](critical_solver_report.md)。

## 结论

`ModelB.jl` 的物理和离散化**正确**：drift 严格等于格点哈密顿量
`H = Σ_x [ (1/2)Σ_μ(φ_{x+μ}-φ_x)² - φ²/2 + φ⁴/24 ]`
的守恒梯度流 `(1/γ)∇²₅ₚₜ(δH/δφ)`（13 点 biharmonic 恰为 (∇²₅ₚₜ)²），
守恒噪声（后向差分散度）协方差 = −(2T/γ)∇²₅ₚₜ，与耗散算符精确匹配（离散 FDT）。
连续时间极限下平衡分布严格为 exp(−H/T)（固定总磁化 M 截面）。

**临界温度：T_c ≈ 4.45 ± 0.05**（Binder 交叉 L=32/64/128：4.457 / 4.426 / 4.434；
交叉点 U4 ≈ 0.61，符合 2D Ising 普适值 0.6107）。

## 数值验证

- 守恒性：所有运行 Σφ 漂移 < 1.2e-6（Float32 舍入水平）
- 稳定性：EM dt=0.005 稳定（线性稳定阈约 dt < 0.03）
- 平衡态交叉验证（vs 独立 Metropolis+Wolff MC，同一 H）：
  - T=6, L=32, t≤2000：S(k) 全曲线吻合，S(kmin) 差 3.6%
  - T=8, L=64, t≤1000：⟨φ²⟩ 差 1.1%
- EM 的 O(dt) 弱偏差（T=6, L=32 的 ⟨φ²⟩）：dt=0.01 → −5.9%，0.005 → −2.9%，
  0.0025 → −2.7%（后者受统计/热化限制）。要 1% 精度需 dt ≲ 0.002。

## 使用注意（非 bug，但要小心）

1. `CURAND.default_rng()` 在当前 CUDA v6.2.0 已移除（旧版 API）→ 原样运行第 91 行会
   UndefVarError；改用 `CUDA.default_rng()`（或 `CUDA.CURAND.library_rng()`），物理等价。
3. 噪声核的 dx 硬编码为 1.0f0；且注释中"换成 dx"的提示不完整——2D 中正确标度是
   divη·√dt/dx²（核内已除一次 dx，还差 dx^{d/2}）。dx=1 时无影响。
4. **只能用 EM 类定步长 solver**：自定义 NoiseProcess 的 bridge 是 no-op，
   自适应/高阶 solver（含默认参数 DRI1NM）会给出错误结果；abstol/reltol 对 EM 无效。
5. `randn` 初值总磁化 ~N(0,1/L)≠0 且被守恒动力学冻结；建议 `u0 .-= mean(u0)`。
6. `maxregs=4` 导致寄存器溢出，仅影响性能。
7. 噪声函数忽略传入的 rng（无法用 seed 复现）。
8. Model B 动力学 z≈3.75：T_c 附近大格子无法在可行时间内平衡
   （τ(k_min)~1/[k̂²(k̂²+m²)]，L=64、T=6 时 ~10⁴）。定 T_c 用平衡 MC 高效得多；
   Model B 适合动力学/粗化研究。

## 求解器实测（T=6, L=32, t≤1500；MC 基准 ⟨φ²⟩=2.8094）

前提：`modelB_runner.jl` 已实现精确守恒 bridge（同散度核 + 标准 bridge 公式）与可选 Z 过程。

| solver | Σφ 漂移 | ⟨φ²⟩ | 偏差 | 结论 |
|---|---|---|---|---|
| EM dt=0.005 | 1e-6 | 2.729 | −2.9% | 基准，可靠 |
| EM dt=0.0025 | 1e-6 | 2.734 | −2.7%(受统计限) | ✓ |
| **DRI1NM** | **2.3 (O(1)!)** | 4.086 | +45% | **不可用**：内部自抽三点分布辅助随机变量，绕过守恒噪声过程 → 守恒/FDT 结构性破坏，与容差无关 |
| SOSRA tol=1e-2 | — | 中止(dt→dtmin) | — | 刚性+粗糙初值卡死误差控制器 |
| SOSRA tol=0.5, dt₀=1e-3 | 1.7e-6 | 2.766 | −1.5% | 可用（加性噪声弱二阶，只线性用 dW/dZ）；同计算成本下偏差约为 EM 一半 |
| LambaEM tol=0.05 | 2.5e-6 | 2.691 | −4.2% | 可用但无精度优势（仍弱一阶） |

理论判据：只有对噪声增量**线性**的格式（EM/LambaEM/SRA 家族）对这种空间关联的加性噪声保持有效；
凡内部使用分量独立性假设或自抽辅助随机变量的弱二阶方法（DRI1/DRI1NM/RI 系）不合法。
大幅提速的正道是谱空间 ETD/指数积分器（线性部分含噪声严格积分，消除刚性限制）。

**SDE 自动初始步长（省略 dt）在本系统上不可用**：`sde_determine_initdt` 对 randn 初值
（∇⁴ 作用后 ‖f‖ 巨大）给出 ≤dtmin 的垃圾 dt₀，无论 dtmin=1.2e-4/1e-6/1e-9 均在 t=0 中止，
force_dtmin=true 则直接数值爆炸——**必须手动给 dt₀**（仅作首步猜测，之后控制器完全自适应）。
实测 SOSRA(tol=0.5, dt₀=1e-3) 在 L=256、T=1 淬火：6553 步跑完 t=100（平均 dt≈0.015，
3× EM 步长，零拒绝步），守恒漂移 1.5e-8。ModelB.jl 已升级：真守恒 bridge + Z 过程 +
默认 solver 改 EM + 底部示例改 SOSRA。

## L=128 求解器 benchmark（2026-07-17；T=8, t=600；MC 基准 ⟨φ²⟩=3.0636；单次运行统计噪声 ~±0.4%）

| solver | dt/平均dt | 墙钟(t=600) | ⟨φ²⟩ 偏差 | 判定 |
|---|---|---|---|---|
| **EM 0.001** | 0.001 | 126.6s | −0.30% | 精度档 |
| **EM 0.0025** | 0.0025 | 59.4s | −0.74% | 平衡测量推荐 |
| **EM 0.005** | 0.005 | 36.5s | −1.18% | **生产默认推荐** |
| **EM 0.01** | 0.01 | 26.9s | −3.2% | 快速探索档 |
| EM 0.02 | — | — | Unstable | 稳定边界 0.01–0.02 |
| EulerHeun 0.005/0.01 | | 42.9/30.1s | −2.6%/−4.9% | 劣于同 dt 的 EM |
| LambaEM tol=0.05 | →0.0102 | 39.1s | −3.1% | ≈EM@0.01 + 开销 |
| SOSRA tol=0.5/0.2 | →0.0064/0.0025 | 62.9/135.9s | −2.8%/−1.4% | 两轴均被 EM 支配 |
| SRA1 tol=0.5 | →0.011 | 42.7s | −5.1% | 差 |
| SRA3 tol=0.5 | — | — | DtLessThanMin | 起不来 |

**结论：L=128 刚性区（a_max·dt≈0.4–0.7 预渐近），定步长 EM 完整占据 Pareto 前沿**（fig_solver_pareto_L128.png）。
弱二阶 SRA 家族的阶优势在此步长区间未兑现且每步成本 2–3×。早前 L=32 上 SOSRA 的
"偏差减半"在 ±0.4% 统计噪声内不显著，撤回该倾向性结论。T=1 淬火：EM/Heun/SOSRA
成本打平（~22s/t=200），无理由弃用 EM。EM 挂 Z 过程多花 22% → ModelB.jl 增加
`with_Z` 开关（默认 true，纯 EM 运行传 false）。数量级提速仍需谱空间 ETD。

## 文件

- `mc_common.jl` / `mc_phi4_scan.jl` / `mc_sk.jl` — 独立平衡 MC（Metropolis + 嵌入式 Wolff）
- `modelB_runner.jl` — ModelB.jl 核心代码逐字拷贝 + 观测量输出（可选第 5 参数 dt）
- `make_plots.jl` — 汇总图
- `mc_scan_coarse.dat` / `mc_scan_fine.dat` — Binder/χ/⟨φ²⟩ 扫描数据
- `mb_*_obs.dat` / `mb_*_Sk.dat` / `mb_*_final.dat` — Model B 时间序列、结构因子、末态
- `fig_binder_crossing.png`, `fig_chi.png`, `fig_Sk_compare.png`,
  `fig_phi2_vs_T.png`, `fig_snapshots.png`
