# 2D SU(2) 粗到细生成原型 · L16 → L32

本文件是交给 Codex 的实现说明。只新增细化、部分加噪采样和验证功能，复用现有网络，不重新训练，不构造粗细配对，不引入条件网络、公共 IR 表示或 RG 有效作用量。

## 1. 固定物理设置

周期边界，Wilson 作用量为 $S_\beta=\beta\sum_p(1-\operatorname{Tr}U_p/2)$。

| 系综 | L | beta | 格距 | 用途 |
|---|---:|---:|---:|---|
| 细格 | 32 | 2.0 | a | 已有数据及已训练网络 |
| 粗格 | 16 | 0.1408904756906944 | 2a | 另外生成独立平衡构型 |

采用无限体积基本表示弦张力设定格距 [1]，定义 $F(\beta)=a^2\sigma_{\rm str}=-\log[I_2(\beta)/I_1(\beta)]$。
固定物理边长要求 $16a_c=32a_f$，因此求解 $F(\beta_c)=4F(2)$。

```python
import numpy as np
from scipy.special import ive
from scipy.optimize import brentq

def F(beta):
    return -np.log(ive(2, beta) / ive(1, beta))

beta_c = brentq(lambda b: F(b) - 4 * F(2.0), 1e-4, 2.0, xtol=1e-14)
print(beta_c)  # 0.1408904756906944
```

不要用弱耦合近似 `beta_c = beta_f / 4 = 0.5` 替代上述物理尺度匹配。
该匹配不是所有表示、所有可观测量的精确匹配，也不要求粗系综等于细系综的阻塞分布。
无限体积公式用于设定尺度，周期有限体积的观测量仍以独立平衡系综为基准。
这组参数有 $a_f\sqrt{\sigma_{\rm str}}\simeq0.914726$，并不接近连续极限。先测试流程及分布匹配，不能预先宣称解决临界慢化。

## 2. 复用现有实现

读取 `train_su2_dm.py` 和 `sample_su2_em.py`，优先复用其 `qmul`、`conj`、`qexp`、`roll`、`plaquette`、`heat_kernel_step`、`LCNN` 和 HDF5 转换。

- 实际目标以用户的 L32、beta=2 数据及 checkpoint 为准。旧脚本中的全局 `L=16` 和默认数据路径不能作为本次输入。
- 从 checkpoint 读取 `L, beta, sigma, ch, nlayers`，检查 `L==32`、`beta==2`，加载 EMA 权重并使用 `eval()`。元数据不符时停止，不修改标签冒充目标模型。
- HDF5 的 `cfgs` 为复矩阵 `[N,y,x,mu,2,2]`，内部四元数为 `[N,2,x,y,4]`。保持轴顺序和矩阵往返转换一致。
- 群约定为 $U=q_0I+i\mathbf q\cdot\boldsymbol\sigma$，`qmul` 的向量乘积含负叉乘，必须沿用原代码。
- 当前前向过程是 SU(2) 群上热核增量的乘法扩散。不得换成四维高斯相加再归一化，也不得把三维 score 归一化。
- 输入先检查单位性。群乘法后可逐链接重归一化修正舍入误差，记录修正前误差。异常范数或 NaN 必须报错。
- 数据、checkpoint、输出路径和设备均通过命令行传入，不覆盖原文件。CPU 代数测试不要触发原训练模块的 CUDA/Triton 导入副作用。

## 3. Prolongation 的明确构造

输入粗链接 $V$，输出细链接 $W$。$X=(X_0,X_1)$ 是粗格整数坐标，$\hat0=(1,0)$、$\hat1=(0,1)$。
分配四个细格点 $A=2X$、$B=A+\hat0$、$C=A+\hat1$、$D=A+\hat0+\hat1$。所有坐标采用周期边界。

计算粗 plaquette 和主值群根

$$
P_X=V_0(X)V_1(X+\hat0)V_0^\dagger(X+\hat1)V_1^\dagger(X),
\qquad Q_X=P_X^{1/4}.
$$

逐块填充以下链接，遍历全部粗格点即可覆盖所有细链接。

| 细格点 | W0 | W1 |
|---|---|---|
| A | I | I |
| B | V0(X) | I |
| C | Q_X† | V1(X) |
| D | Q_X† V0(X) | Q_X² V1(X) |

这一确定性构造必须满足两段细链接乘积等于原粗链接，并且块内四个细 plaquette 都等于 $Q_X$，后一个等式在表格选定的块内规范下成立。
粗格规范变换提升为同一块内四个细格点的共同变换，检查细化的协变性。
群根使用稳定的四元数 `atan2` 实现并处理单位元极限。接近 `-I` 时记录并检查精度，准确的分支歧义必须报错。不得静默丢弃这些粗构型或固定一个任意颜色方向。

先比较三个版本，不训练 prolongation 网络。

| 版本 | 块内更新轮数 | 全链接更新轮数 |
|---|---:|---:|
| uniform | 0 | 0 |
| interior | 2 | 0 |
| relaxed | 2 | 1 |

块内更新仅允许方向 0 的 `x1` 为奇数的链接，以及方向 1 的 `x0` 为奇数的链接变化。其余链接构成受保护的粗格骨架。
全链接更新允许骨架改变，以修正有限格距分布差异。

局部更新采用对称 Metropolis 提案 $W'=\exp(i\delta\xi^a\sigma^a/2)W$，$\xi\sim N(0,I_3)$，起始取 $\delta=1/\sqrt{\beta_f}=1/\sqrt2$。
接受率为 $\min(1,e^{-\Delta S_{\beta_f}})$，这里始终使用 **beta_f=2**，不能使用 beta_c。
只计算受影响的两个 plaquette；按方向和棋盘奇偶分成四个不冲突子步，一轮包含四个子步。记录接受率，提案宽度可以调整。

最后施加独立逐站点 Haar 规范变换，保持全部规范不变量。此操作只是随机化规范坐标，不是物理涨落。
有限轮局部更新不等于完成热化，其成本必须计入算法比较。

## 4. 部分加噪与反向采样

总体过程为 `V16 → prolongation → W32 → forward noise → reverse diffusion → Uout32`。

从 checkpoint 读取噪声参数 `sigma`。用 $\Sigma(t)$ 表示噪声尺度，其平方 $\Sigma(t)^2=\int_0^t g(u)^2\,\mathrm du$ 是累计方差。$t$ 为网络时间，$\sigma$ 是噪声日程参数，物理弦张力记为 $\sigma_{\rm str}$。下式对应本文 L32 网络使用的原噪声日程。

$$
\Sigma(t)^2=\frac{\sigma^{2t}-1}{2\log\sigma},
\qquad
 t_\star=\frac{\log(1+2\Sigma_\star^2\log\sigma)}{2\log\sigma}.
$$

文中噪声强度统一用 $\Sigma^2$。历史文件名、HDF5 和 CSV 的 `tau`、`tau_start` 字段及旧图中的同名标注均表示 $\Sigma^2$，数值保持原值。`--Sigma2` 接收累计方差，旧参数名 `--tau` 仍可读取已有命令。

**前向**复用训练中的增量核。取 $K=\max(1,\lceil\Sigma_\star^2/\texttt{DELTA\_MAX}\rceil)$、$\Delta=\Sigma_\star^2/K$，重复左乘 `qexp(heat_kernel_step(..., Delta))`。
真实细格和细化场使用同一噪声核、同一累计方差，独立抽样噪声。$\Sigma^2=0$ 直接返回输入。

**反向**将现有 sampler 改为接受初态 `U_start` 和起始时间 `t_start`，不再强制从 Haar 或 `t=1` 开始。
在递减时间网格上令 `dt=t_i-t_{i+1}>0`、`v=g(t_i)**2*dt`，沿用原更新

```text
S = model(U, t_i)
Z = heat_kernel_step(..., v)
U = qmul(qexp(v*S + Z), U)
```

从 `t_star` 积分到正数 `eps`，不向网络传入 `t=0`。确保单步 `v` 未超出热核近似的适用范围。
末步是否只保留漂移、终止 `eps` 和数值步长必须在各对照间一致，并进行收敛测试。
$\Sigma_\star^2=0$ 只比较未加噪初始化，不调用反向网络；正时间采样须满足 `t_star>eps`。
固定粗构型可生成多个随机副本，但它们不是独立的无条件长波样本。

## 5. 按顺序实现三个实验

**A. 前向扫描，不需要 checkpoint。**
对独立细格系综以及三种细化系综，扫描 $\Sigma^2=[0,0.02,0.05,0.1,0.2,0.4,0.8,1.6]$。
先用约 256 个已评估相关性的构型调试；这不是保证充分的统计量。

**B. 先验证现有网络。**
同时运行 Haar 起点生成，以及“真实细格 → 加噪到 $\Sigma_\star^2$ → 反向生成”。后者用于区分模型误差和细化误差。

**C. 细化起点生成。**
对前向扫描选出的噪声水平运行反向生成，并与 A/B 的基准比较。部分采样使用完整 2000 步网格的对应后缀。Haar 起点复用原 sampler 设置。

至少测量以下规范不变量。

- Wilson loops `1×1,1×2,2×2,4×4` 的均值、二阶矩、分布和部分协方差。区分单个 loop 方差与每构型空间平均后的方差。
- plaquette 和 `2×2` loops 的四种奇偶锚点统计，检查块状缺陷；两个方向 Polyakov loops 的二阶矩和分布。
- 粗格 `R×T` loop 与细化/生成后细格 `2R×2T` loop 的相关性，与真实细格在相同噪声下的自身保留相关性比较。

所有分布比较都在 L32 上进行。不要把粗格 `1×1` 与细格 `1×1` 当成同物理面积。
可增加基于上述特征的 MMD 或两样本分类检验，不必训练额外大型网络。未检出差异不等于证明完整分布相等。
按原始构型或 Monte Carlo 链段分组划分验证集和做 bootstrap，同一父构型的噪声副本不能跨训练/测试，也不能用于虚增独立样本数。
本组参数的大循环信号可能很弱，报告绝对误差和信噪比，不对接近零的均值使用不稳定的相对误差。

## 6. 必须通过的检查与输出

单元测试覆盖四元数/矩阵往返、单位性、周期索引、确定性细化的骨架保持与均匀通量、提升后的规范协变性、局部作用量差与全作用量差一致。
随机更新后的协变性测试需要同步旋转提案或比较不变量分布，不能对两个独立随机结果直接逐元素比较。

热核实现另检验理想公式 $\langle W_\Gamma\rangle_{\Sigma^2}=e^{-3n_\Gamma\Sigma^2/8}\langle W_\Gamma\rangle_0$，其中循环含 $n_\Gamma$ 条不重复链接。
两套 loop 均值同时趋零可能只是共同衰减，不是分布匹配成功的充分证据。分别控制统计误差、热核近似和反向积分误差。

建议新增一个独立入口 `refine_su2.py`，包含 `check`、`scan`、`sample` 子命令及单元测试，不改写原训练文件。
命令行接收 `--coarse --fine --checkpoint --out --seed --device`，以及细化版本、$\Sigma^2$、步数、eps 和批量参数。不存在的输入文件直接报错。
输出兼容原 `cfgs` HDF5，另保存父构型编号、随机种子、beta_c/beta_f、L、$\Sigma_\star^2$ 与 `t_start`、积分设置、观测量、接受率及各阶段耗时。

验收目标是在较小 $\Sigma^2$ 下，细化系综与细格加噪系综难以区分，同时仍保留有统计显著性的输入信息。
如果只有很大噪声才匹配，应明确报告失败或不足。有限噪声反向生成与有限步 MC 均不保证精确平衡，后续可做再热化长度扫描。
2026-10-01 已完成实验 A 的 CPU 实现、首轮 $\Sigma^2\le1.6$ 扫描、$\Sigma^2\le10$ 扩展，以及后文记录的实验 B。实验 C 的 $\Sigma^2\simeq3$ 测试记录见文末。实验 A 没有读取 checkpoint，实验 B/C 使用已有网络在 GPU 上采样。

入口为 [refine_su2.py](refine_su2.py)。`check` 和 `scan` 按原群乘法、回路及热核拒绝采样公式进行 NumPy 计算，不导入 Torch 或 Triton。`sample` 执行实验 B，传入 `--coarse` 时执行实验 C 的三个细化版本与 Haar 对照，均延迟导入现有 GPU 采样器。实验 C 默认不执行 Fine start 的反向生成，可在 `--initializations` 中加入 `fine` 启用。实验 B 保持原流程。训练文件不变，原采样器新增初态与起始时间参数，保留原 EM 更新及完整 2000 步设置。

实验 C 可用 `--initializations relaxed` 只生成 relaxed 与 Haar 对照，或用 `--initializations fine relaxed` 加入真实平衡细格加噪后的反向对照。未指定时仍运行三个细化版本，各版本的随机种子保持原编号。

首轮结果见 [15 页比较图](../../results/su2_prolongation_forward/scan.pdf) 和 [均值预览](../../results/su2_prolongation_forward/scan.png)。横轴数值表示累计方差 $\Sigma^2$。均值图等部分横轴在 $0.03$ 以下为线性，以上为对数，保留 $\Sigma^2=0$ 并展开较小噪声区域。曲线连接八个独立加噪的测量点，不代表对同一条噪声轨迹连续观测。

| 图页 | 内容 |
| --- | --- |
| 1 | 四种 Wilson 回路均值随 $\Sigma^2$ 的变化 |
| 2–4 | 单个回路的二阶矩、方差，以及每构型空间平均的方差 |
| 5 | 相对同噪声细格基准的均值绝对差 |
| 6–7 | 六对回路在同一锚点及空间平均后的协方差 |
| 8–9 | plaquette 与 $2\times2$ 回路四种锚点的均值、二阶矩随 $\Sigma^2$ 的变化 |
| 10 | $\Sigma^2=0,0.4,1.6$ 的四种奇偶锚点比较 |
| 11 | 相同物理面积下的输入保留相关性，分别按局部回路和构型空间平均计算 |
| 12–13 | Wilson 回路和两个方向 Polyakov 回路在三个噪声水平下的分布 |
| 14 | 两个方向 Polyakov 回路的均值和二阶矩随 $\Sigma^2$ 的变化 |
| 15 | 分布累积概率与细格基准的最大绝对差随 $\Sigma^2$ 的变化 |

细格基准使用 `trainningdata/cfgs_su2_beta=2.0_32^2.h5`。从 102400 个构型中每隔 400 个取一个，共 256 个。粗格采用原有 `su2_2d_heatbath.jl`，生成 4096 个 $L=16$、$\beta_c=0.1408904756906944$ 的构型，热化 2000 次更新，保存间隔 20 次，每次含一次热浴和三次过松弛。扫描每隔 16 个粗构型取一个，共 256 个。两套输入的父构型编号均已保存。

粗格完整链的 $\langle P\rangle=0.0353722\pm0.0004932$，与无限体积参考值 $0.0351935$ 相差 $0.36$ 个标准误差。Wilson 和 Polyakov 观测量的首次非正截断自相关时间为 $0.5000$ 至 $0.5042$，单位为保存间隔。块长 1、4、16、64 的误差以及固定 20 个间隔的窗口估计见 [链相关性检查](../../results/su2_prolongation_forward/scan_chains.csv)。细格完整链的其余观测量检查复用 [已有数据检查](../../trainningdata/su2_heatbath_102400_checks.csv)。这些诊断未见明显剩余相关，不构成所有观测量独立的证明。

扫描误差按相邻四个入选父构型分组，进行 1000 次 block bootstrap。图中误差棒为一个标准误差，CSV 另含 95% 区间。每个父构型在每个噪声水平只有一个副本。各噪声水平重新独立抽样噪声，比较的两套系综也使用独立噪声。同一父构型对应的空间链接不作为独立误差样本。

**在本次 $\Sigma^2\le1.6$ 的扫描内，三种细化系综均未达到分布难以区分的目标。** `relaxed` 缩小了偏差，但仍有明显差异。

| 系综 | $\langle P\rangle$，$\Sigma^2=0$ | $\langle P\rangle$，$\Sigma^2=1.6$ | $\Sigma^2=1.6$ 相对细格的偏差显著性 |
| --- | --- | --- | --- |
| fine | $0.432829\pm0.000762$ | $0.038466\pm0.000949$ | 基准 |
| uniform | $0.918605\pm0.000225$ | $0.083295\pm0.000891$ | $34.4$ 个标准误差 |
| interior | $0.710203\pm0.000473$ | $0.062853\pm0.000978$ | $17.9$ 个标准误差 |
| relaxed | $0.603295\pm0.000628$ | $0.055515\pm0.000953$ | $12.7$ 个标准误差 |

在每个噪声水平，以所有已测矩、奇偶锚点量、回路乘积和 40 个等宽区间的累积分布构造最大标准化均值差，对完整构型组进行 999 次置换。三种细化方式的八个噪声水平都得到 $p=0.001$，这是该置换次数的最小可报告值。各回路的分布距离和单独置换结果见 [分布与矩比较](../../results/su2_prolongation_forward/scan_comparison.csv)。这不是对完整场分布的充分检验，当前仅 plaquette 均值就已排除匹配。

块状缺陷在 $2\times2$ 回路中尤其明显。`uniform` 和 `interior` 在偶偶锚点准确保留粗 plaquette，但其他锚点的回路分布不同，因此不能将偶偶锚点的匹配推广到全细格。一次全链接更新缩小了锚点差异，同时削弱了输入保留。`relaxed` 在未加噪时，粗 $1\times1$ 与偶偶锚点细 $2\times2$ 的构型空间平均相关系数已降至 $0.355\pm0.051$。

本次相关性检查使用粗 $1\times1$ 对细 $2\times2$，以及粗 $2\times2$ 对细 $4\times4$。细格基准也比较相同偶偶锚点在加噪前后的自身相关。在 $\Sigma^2=0.2$，`uniform` 和 `interior` 的 $2\times2$ 构型平均相关系数分别为 $0.581\pm0.035$、$0.559\pm0.045$，此时分布偏差仍然很大。在 $\Sigma^2=1.6$，三种细化方式的该构型平均相关系数的 95% 区间均包含零。局部回路仍可能检出很弱的相关，不能据此声称保留了统计显著的整体长波信息。数值见 [输入保留结果](../../results/su2_prolongation_forward/scan_retention.csv)。本次没有选出同时满足分布接近与输入保留的噪声水平，也没有据此启动反向采样。

所有代数检查通过，包括矩阵往返、周期路径、主值根与分支错误、骨架保持、均匀通量、确定性规范协变性，以及同步旋转提案后的随机更新协变性。局部作用量差与完整作用量差一致。单位场的热核测试同时检查 `DELTA_MAX=1` 与减半后的结果。实际扫描的 112 个非零时间回路均值检验中，$\langle W\rangle_{\Sigma^2}-e^{-3n_\Gamma\Sigma^2/8}\langle W\rangle_0$ 的最大偏差为 $2.57$ 个标准误差。见 [热核检验](../../results/su2_prolongation_forward/scan_heat_kernel.csv) 和 [检查日志](../../results/su2_prolongation_forward/checks.log)。保存后的全部 32 组构型已重读验证，矩阵与单位性最大误差为 $3.4\times10^{-16}$，回路重算最大差为 $1.6\times10^{-15}$。

块内更新接受率为 $0.6163$，全链接更新为 $0.6644$。在 `mg0796` 上，粗系综生成的程序计时为 $18.7$ 秒，前向扫描含粗链诊断、观测量、统计及数据写出为 $28.3$ 秒，均不包含各自进程启动，扫描计时不含绘图。其中 256 个构型的两轮块内更新为 $0.402$ 秒，一轮全链接更新为 $0.256$ 秒。Julia 使用四个 CPU 线程，NumPy 扫描设置单线程 BLAS。全程未导入 Torch 或 Triton，未使用 GPU。

全部构型及逐构型观测量位于 [scan.h5](../../results/su2_prolongation_forward/scan.h5)。路径为 `ensemble/tau_VALUE/cfgs`，每个 `cfgs` 保持 `[N,y,x,mu,2,2]` 的复矩阵约定。文件还保存输入索引、独立噪声流种子、物理参数、接受率、归一化前误差和 bootstrap 权重。完整均值、方差及协方差表见 [统计结果](../../results/su2_prolongation_forward/scan_statistics.csv)，耗时见 [分阶段计时](../../results/su2_prolongation_forward/scan_timing.csv)。

在仓库根目录重现可运行以下命令。`scan` 使用独占创建，已有同名输出时会报错，复跑应指定新的输出文件。

```bash
julia --startup-file=no -t 4 trainningdata/su2_2d_heatbath.jl \
  --beta 0.1408904756906944 --L 16 --n_cfgs 4096 \
  --n_therm 2000 --n_skip 20 --n_or 3 --seed 20261001 \
  --out results/su2_prolongation_forward

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 src/Prolongation/refine_su2.py check

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 src/Prolongation/refine_su2.py scan \
  --coarse 'results/su2_prolongation_forward/cfgs_su2_beta=0.1408904756906944_16^2.h5' \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --out results/su2_prolongation_forward/scan.h5 \
  --seed 20261001 --device cpu --n-cfgs 256 \
  --Sigma2 0 0.02 0.05 0.1 0.2 0.4 0.8 1.6
```

按后续要求，扫描已扩展到 $\Sigma^2=10$。新结果见 [扩展比较图，共 15 页](../../results/su2_prolongation_forward/scan_tau10.pdf)、[均值预览](../../results/su2_prolongation_forward/scan_tau10.png) 和 [完整构型与观测量](../../results/su2_prolongation_forward/scan_tau10.h5)。原扫描结果保留。

扩展网格为 $\Sigma^2=[0,0.02,0.05,0.1,0.2,0.4,0.8,1.6,2,2.5,3,4,6,8,10]$。继续使用每套 256 个构型及相同父构型编号、细化更新、随机流和统计方法。原有 32 组结果中的构型、观测量和误差均逐项完全复现。新增 28 组，合计 60 组。热核仍按 `DELTA_MAX=1` 拆分，$\Sigma^2=10$ 使用十个增量。按当前训练程序的 $\sigma=14$，$\Sigma^2=10$ 对应网络时间 $t\simeq0.75499$，训练上限仍是 $t=1$、$\Sigma^2\simeq36.945$。

扩展图保留前述各项统计随时间的变化，奇偶锚点和分布快照改为 $\Sigma^2=0,1.6,10$。两图的横轴数值均对应 $\Sigma^2$。

在 $\Sigma^2=10$，三种细化系综与细格基准的整体置换检验均未检出显著差异。plaquette 结果如下。

| 系综 | $\langle P\rangle$，$\Sigma^2=10$ | 相对 fine 的均值差 | 整体置换 $p$ |
| --- | --- | --- | --- |
| fine | $0.000493\pm0.000988$ | 基准 | 基准 |
| uniform | $-0.001267\pm0.000870$ | $-0.001759\pm0.001316$ | $0.415$ |
| interior | $0.001603\pm0.000922$ | $0.001111\pm0.001351$ | $0.257$ |
| relaxed | $-0.001306\pm0.001035$ | $-0.001798\pm0.001430$ | $0.799$ |

误差均为一个标准误差。未检出差异仅针对这些观测量和当前样本量，不证明完整分布相等。完整结果见 [扩展统计表](../../results/su2_prolongation_forward/scan_tau10_statistics.csv) 和 [扩展比较表](../../results/su2_prolongation_forward/scan_tau10_comparison.csv)。整体检验在 $\Sigma^2=3$ 和 $4$ 也不显著，但不能将其当成精确的匹配阈值。例如 $\Sigma^2=3$ 的 `uniform` plaquette 仍相差约 $2.8$ 个标准误差。$\Sigma^2=6$ 的 `uniform` 和 `interior` 整体 $p$ 值分别为 $0.046$ 和 $0.036$，这是尚未对不同时间和版本进行多重比较校正的数值，结果并不随时间单调变化。

大噪声下的分布接近伴随所测输入相关性的衰减。在 $\Sigma^2=10$，粗 $1\times1$ 与偶偶锚点细 $2\times2$ 的构型平均相关系数分别为 $0.015\pm0.071$、$0.028\pm0.062$、$0.005\pm0.062$，依次对应 `uniform`、`interior`、`relaxed`，95% 区间均包含零。三种细化系综的粗 $2\times2$ 与细 $4\times4$ 构型平均相关性的区间也均包含零。细 $2\times2$ 的理想热核保留因子为 $e^{-3\Sigma^2}$，在 $\Sigma^2=3$ 已降至 $1.23\times10^{-4}$，在 $\Sigma^2=10$ 为 $9.36\times10^{-14}$。因此扩展扫描仍没有找到同时保留显著粗格回路信息并实现分布接近的噪声水平。见 [扩展输入相关性](../../results/su2_prolongation_forward/scan_tau10_retention.csv)。

全部新增构型通过重读检查，矩阵与单位性最大误差为 $3.4\times10^{-16}$，回路重算最大差为 $1.7\times10^{-15}$。224 个非零时间回路衰减检验中，最大残差为 $3.28$ 个标准误差。见 [扩展验证记录](../../results/su2_prolongation_forward/scan_tau10_checks.log) 和 [扩展热核检验](../../results/su2_prolongation_forward/scan_tau10_heat_kernel.csv)。全网格 CPU 扫描、统计及写出用时 $58.8$ 秒，不含绘图。仍未使用 GPU。

扩展扫描的命令如下。已有粗格数据直接复用。程序当前默认网格也已更新为上述 15 个点。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 src/Prolongation/refine_su2.py scan \
  --coarse 'results/su2_prolongation_forward/cfgs_su2_beta=0.1408904756906944_16^2.h5' \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --out results/su2_prolongation_forward/scan_tau10.h5 \
  --seed 20261001 --device cpu --n-cfgs 256 \
  --Sigma2 0 0.02 0.05 0.1 0.2 0.4 0.8 1.6 2 2.5 3 4 6 8 10
```

**实验 B 的首批 256 构型验证。**

结果见 [20 页验证图](../../results/su2_network_validation/validation.pdf)、[均值预览](../../results/su2_network_validation/validation.png) 和 [构型与逐构型观测量](../../results/su2_network_validation/validation.h5)。使用 `su2_L32_beta2.0_lcnn_hk_c16l3/ckpt_epoch0300.pt` 的 EMA 权重，元数据为 $L=32$、$\beta=2$、$\sigma=14$、16 通道和 3 层。每组 256 个构型，Haar 起点运行完整 2000 步。

按用户要求，以剩余步数 $N$ 选择起点，所有部分采样直接使用完整 2000 步网格的最后 $N$ 步。终止时间固定为 $\epsilon=10^{-5}$，末步只保留漂移。对于 $N>0$，起始时间满足

$$
t_\star=\epsilon+(1-\epsilon)\frac{N}{2000},
\qquad
\Sigma_\star^2=\frac{14^{2t_\star}-1}{2\log14}.
$$

实际使用并保存原采样器生成的 float32 时间网格。真实细格重新加噪到由该网格计算的 $\Sigma_\star^2$，因此前向噪声和反向起点一致。$N=0$ 单独表示未加噪、未调用网络的输入。步数固定后，$\Sigma_\star^2$ 不再另行取整。1500 步对应 $\Sigma_\star^2\simeq9.735$，与此前扫描到 10 的范围接近。

输入复用实验 A 的 256 个真实细格父构型，编号为 $200,600,\ldots,102200$。独立平衡基准取同一 102400 构型文件中的 $0,400,\ldots,102000$，与输入不重合。两者保存间隔的相关性已在实验 A 中检查。每个起点重新独立抽样前向及反向噪声。误差继续使用四个父构型一组、1000 次 block bootstrap，整体检验使用 999 次块置换。配对变化始终按相同父构型比较输出与原始输入。

独立平衡基准的 $\langle P\rangle=0.433999\pm0.000750$。Haar 2000 步的结果为 $0.431282\pm0.000902$，相差 $2.32$ 个标准误差，整体置换检验 $p=0.302$。真实细格起点的结果如下。表中误差为一个标准误差，$p$ 比较各输出与独立平衡基准。

| 剩余步数 $N$ | 网络时间 $t_\star$ | 累计方差 $\Sigma_\star^2$ | 输出 $\langle P\rangle$ | 整体置换 $p$ |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | $0.432829\pm0.000736$ | $0.158$ |
| 50 | 0.025010 | 0.026736 | $0.433097\pm0.000757$ | $0.615$ |
| 100 | 0.050010 | 0.057231 | $0.433056\pm0.000676$ | $0.886$ |
| 200 | 0.100009 | 0.131732 | $0.433092\pm0.000830$ | $0.556$ |
| 400 | 0.200008 | 0.355029 | $0.432921\pm0.000795$ | $0.981$ |
| 600 | 0.300007 | 0.733565 | $0.432271\pm0.000744$ | $0.592$ |
| 800 | 0.400006 | 1.375261 | $0.432709\pm0.000641$ | $0.646$ |
| 1000 | 0.500005 | 2.463071 | $0.431947\pm0.000814$ | $0.416$ |
| 1200 | 0.600004 | 4.307135 | $0.433248\pm0.000669$ | $0.704$ |
| 1500 | 0.750003 | 9.735275 | $0.432542\pm0.000697$ | $0.987$ |
| 2000 | 1 | 36.945010 | $0.432608\pm0.000780$ | $0.998$ |

本次整体检验均未检出显著差异，适用于所测回路、Polyakov 量、奇偶锚点及其分布特征和当前样本量。十个起点的四种回路均值，相对原始输入的配对变化均小于 $1.74$ 个标准误差。个别更细的统计量仍有偏差。例如 50 步时，$2\times2$ 回路在奇偶锚点 `10` 的二阶矩配对变化为 $-0.002686\pm0.000690$，约 $3.89$ 个标准误差。这是全扫描 440 项配对比较中的未校正结果，不能据此认定网络无偏或确认单项异常。

所有均值、二阶矩、局部方差、构型平均方差和协方差见 [统计表](../../results/su2_network_validation/validation_statistics.csv)。[与独立平衡基准的比较](../../results/su2_network_validation/validation_comparison.csv) 包含均值绝对差、标准误差、整体置换结果和分布距离。[配对变化表](../../results/su2_network_validation/validation_paired.csv) 保留对原始输入的逐项变化。大回路均值接近零，使用绝对误差解释结果。

输入保留见 [回路相关性](../../results/su2_network_validation/validation_retention.csv)。偶偶锚点 $2\times2$ 回路的构型平均相关系数，在反向生成后由 50 步的 $0.854\pm0.018$，降至 200 步的 $0.491\pm0.041$、400 步的 $0.117\pm0.048$ 和 800 步的 $-0.032\pm0.052$。800 步时，所测 $2\times2$ 和 $4\times4$ 回路的局部与构型平均相关性的 95% 区间均包含零。输出回路分布与基准接近，但大噪声下未检出这些回路与原始输入的显著相关性。

验证图第 1 至 5 页依次给出四种回路均值、相对独立基准的差、二阶矩、局部方差及构型平均方差。第 6 页为配对均值变化，第 7 至 8 页为协方差，第 9 页为奇偶锚点统计，第 10 页为 Polyakov 矩，第 11 页为加噪前后及反向后的输入相关性。第 12 至 13 页比较 50、800、2000 步的 Wilson 与 Polyakov 分布，第 14 页给出整体置换结果，第 15 页给出步数与 $\Sigma_\star^2$ 的对应关系。噪声扫描曲线横轴为 $\Sigma_\star^2$。

按后续要求，图中增加了 holonomy 的比较。定义绕两个周期方向的有序乘积

$$
\Omega_x(y)=\prod_{x=0}^{L-1}U_x(x,y),
\qquad
\Omega_y(x)=\prod_{y=0}^{L-1}U_y(x,y),
\qquad
P_\mu=\frac12\operatorname{Tr}\Omega_\mu=\cos\theta_\mu,
\quad \theta_\mu\in[0,\pi].
$$

其本征值为 $e^{\pm i\theta_\mu}$。现有 Polyakov loop 已经测量 holonomy 的迹，因此本征角分布是同一单回路信息的另一种表示。为补充两个方向的相对信息，在共同起点 $(0,0)$ 另测

$$
C=\frac12\operatorname{Tr}
\left[\Omega_x(0)\Omega_y(0)\Omega_x^\dagger(0)\Omega_y^\dagger(0)\right].
$$

$C$ 在规范变换和中心翻转下不变，取值连续，不是整数拓扑荷。两个绕圈矩阵及其交换子与二维环面上的整体约束有关，见 [环面 Polyakov loops 的解析研究](https://arxiv.org/abs/1403.1770)。这里固定共同起点，不将不同基点的矩阵直接相乘。

第 16 页比较本征角展宽 $\langle(\theta_\mu-\pi/2)^2\rangle$、$\langle C\rangle$ 和 $\langle C^2\rangle$。第 17 页给出两个方向的本征角及 $C$ 的分布。第 18 页比较加噪及反向生成后的 holonomy 迹与原始输入的相关性。第 19 页给出 $\langle\overline P_\mu^2\rangle$，其中 $\overline P_\mu$ 是每个构型内对横向位置的平均，区别于原图中的 $\langle\overline{P_\mu^2}\rangle$。第 20 页为这些 holonomy 特征的整体置换检验。参考、输入、Haar 及全部前向和反向构型均保存对应的逐构型特征。

补充结果见 [holonomy 统计](../../results/su2_network_validation/validation_holonomy_statistics.csv)、[分布与矩比较](../../results/su2_network_validation/validation_holonomy_comparison.csv) 和 [输入保留](../../results/su2_network_validation/validation_holonomy_retention.csv)。统计按父构型分组，不将同一构型中的 32 条绕圈路径当成独立样本。补充计算仅使用已保存构型，不调用网络。

这里的输入保留相关性衡量前向加噪及反向生成后还保留多少原始信息。它不是热浴加过松弛链的自相关函数，本次比较不能据此确定 Monte Carlo 弛豫最慢的量。

保存数据包含 `reference`、`input`、`haar_steps2000`，以及十组 `forward_stepsN` 和十组 `fine_stepsN`。各组 `cfgs` 使用原复矩阵轴约定，反向组保存实际积分网格。全部 23 组重读通过，单位性误差不超过 $3.4\times10^{-16}$，矩阵误差不超过 $4.1\times10^{-16}$，观测量重算误差不超过 $3.9\times10^{-15}$。投影前最大的 float32 范数漂移为 $7.06\times10^{-6}$。40 项前向热核回路衰减检验的最大残差为 $2.20$ 个标准误差。完整网格的最大单步增量方差为 $0.098005$，小于 `DELTA_MAX=1`。

11 套反向积分网格均逐点等于完整网格的对应后缀。新增采样入口与手动运行 50 次原 EM 更新的输出完全一致，并保持输入不变。原完整网格、正数终止时间及增量方差限制也通过检查。

在一张 GH200 上，前向加噪合计约 $0.31$ 秒，11 组反向采样合计约 $101.5$ 秒，包含观测量及写出为 $116.8$ 秒。这些计时不含进程启动、编译预热和绘图。检查见 [验证记录](../../results/su2_network_validation/checks.log)，运行见 [采样日志](../../results/su2_network_validation/validation.log)。本次验证限定在完整 2000 步所定义的分辨率，未额外扫描网格密度，因而不能分离网络误差与 EM 离散误差。

在仓库根目录重现实验 B 可运行以下命令。已有同名输出时需要指定新路径。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python3 src/Prolongation/refine_su2.py sample \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --forward results/su2_prolongation_forward/scan_tau10.h5 \
  --checkpoint results/su2_L32_beta2.0_lcnn_hk_c16l3/ckpt_epoch0300.pt \
  --out results/su2_network_validation/validation.h5 \
  --seed 20261002 --device cuda --n-cfgs 256 --eps 1e-5 \
  --steps 50 100 200 400 600 800 1000 1200 1500 2000
```

**实验 B 的 5120 构型验证。**

高统计量结果单独保存，见 [20 页验证图](../../results/su2_network_validation/validation_n5120.pdf)、[均值预览](../../results/su2_network_validation/validation_n5120.png) 和 [构型与逐构型观测量](../../results/su2_network_validation/validation_n5120.h5)。沿用上述 EMA 权重、十个起点及完整 2000 步网格。每组使用 5120 个构型，随机种子改为 `20261004`。

输入从 102400 个真实细格构型中取编号 $10,30,\ldots,102390$，独立平衡基准取 $0,20,\ldots,102380$。每组内部无重复，输入与基准不重合。相邻所选父构型间隔 20 个已保存构型。这次重新计算完整的 5120 个构型，不将首批 256 个结果直接拼接进来。误差和置换检验仍按四个父构型组成的块计算。

独立平衡基准的 $\langle P\rangle=0.433127\pm0.000176$。Haar 2000 步为 $0.432690\pm0.000186$，与基准相差 $1.71$ 个标准误差。各细格起点的结果如下。两个整体检验均比较输出与独立平衡基准，未对起点数作校正。

| 剩余步数 $N$ | 输出 $\langle P\rangle$ | 原有观测量整体 $p$ | holonomy 整体 $p$ |
| --- | --- | --- | --- |
| 0 | $0.433318\pm0.000176$ | $0.398$ | $0.296$ |
| 50 | $0.433474\pm0.000175$ | $0.205$ | $0.206$ |
| 100 | $0.433708\pm0.000177$ | $0.478$ | $0.830$ |
| 200 | $0.433345\pm0.000174$ | $0.814$ | $0.936$ |
| 400 | $0.433214\pm0.000182$ | $0.596$ | $0.611$ |
| 600 | $0.432788\pm0.000179$ | $0.357$ | $0.609$ |
| 800 | $0.432994\pm0.000174$ | $0.053$ | $0.030$ |
| 1000 | $0.432637\pm0.000174$ | $0.439$ | $0.911$ |
| 1200 | $0.432728\pm0.000170$ | $0.438$ | $0.433$ |
| 1500 | $0.432651\pm0.000175$ | $0.277$ | $0.725$ |
| 2000 | $0.432993\pm0.000174$ | $0.934$ | $0.982$ |

原有观测量的整体检验均未达到 $p<0.05$。配对比较能看出更小的变化。100 步时，$1\times2$ 回路的变化为 $(6.38\pm1.53)\times10^{-4}$，即 $4.17$ 个标准误差，plaquette 的变化为 $(3.91\pm1.16)\times10^{-4}$，即 $3.36$ 个标准误差。440 项配对比较中的最大 $|z|=4.39$，对应同一起点的 $\overline W_{11}\overline W_{12}$。这些 $z$ 是逐项结果，应结合多重比较理解。当前统计量已经分辨出值得检查的小偏移，不能将整体检验不显著解释为网络无偏。仍使用相同的 2000 步分辨率，因此不能分离网络误差与 EM 离散误差。

holonomy 的交换子迹在独立基准中为 $0.24498\pm0.00741$，Haar 生成中为 $0.23459\pm0.00742$，两者差异约一个标准误差。Haar 生成的 holonomy 整体 $p=0.495$。800 步的 $p=0.030$ 主要来自 $\theta_y$ 的累积分布，其在 $0.3\pi$ 处比基准高 $0.00428$。这是一个起点上的提示，对十个细格起点作 Bonferroni 校正后为 $0.30$，尚不足以确认全局分布偏差。

绕 $x$ 方向的逐路径 holonomy 迹，在反向生成后与输入的相关系数由 50 步的 $0.5336\pm0.0017$，降至 100 步的 $0.2632\pm0.0024$、200 步的 $0.0507\pm0.0024$ 和 400 步的 $0.0013\pm0.0025$。这里的误差仍按父构型计算。作为比较，偶偶锚点 $2\times2$ 回路的构型平均相关系数在 400 步仍为 $0.1470\pm0.0133$，600 步降至 $0.0002\pm0.0142$。这些结果说明所测绕圈迹在当前扩散过程中较早丢失输入信息，不说明它在 Monte Carlo 更新中弛豫最快或最慢。

原有观测量结果见 [统计表](../../results/su2_network_validation/validation_n5120_statistics.csv)、[独立基准比较](../../results/su2_network_validation/validation_n5120_comparison.csv)、[配对变化](../../results/su2_network_validation/validation_n5120_paired.csv) 和 [输入保留](../../results/su2_network_validation/validation_n5120_retention.csv)。全局量见 [holonomy 统计](../../results/su2_network_validation/validation_n5120_holonomy_statistics.csv)、[holonomy 比较](../../results/su2_network_validation/validation_n5120_holonomy_comparison.csv) 和 [holonomy 输入保留](../../results/su2_network_validation/validation_n5120_holonomy_retention.csv)。

全部 23 组、每组 5120 个构型通过重读检查。最大矩阵误差为 $4.01\times10^{-16}$，原有观测量重算误差为 $3.78\times10^{-15}$，交换子迹的独立 SU(2) 迹恒等式检查误差为 $1.96\times10^{-14}$。所有统计均值、误差、区间及积分网格通过核对。60 项前向 Wilson 和 Polyakov 热核衰减检查的最大残差为 $2.82$ 个标准误差。输入与基准的 holonomy 另检查了链相关性和块长 1、4、16、64 的误差，见 [验证记录](../../results/su2_network_validation/validation_n5120_checks.log)。GH200 上反向采样合计 $1903.1$ 秒，采样、原有观测量及构型写出合计 $2561.2$ 秒，不含编译预热、后续 holonomy 统计及绘图。见 [运行日志](../../results/su2_network_validation/validation_n5120.log)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python3 src/Prolongation/refine_su2.py sample \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --forward results/su2_prolongation_forward/scan_tau10.h5 \
  --checkpoint results/su2_L32_beta2.0_lcnn_hk_c16l3/ckpt_epoch0300.pt \
  --out results/su2_network_validation/validation_n5120.h5 \
  --seed 20261004 --device cuda --n-cfgs 5120 --eps 1e-5 \
  --steps 50 100 200 400 600 800 1000 1200 1500 2000
```

**九节点追加采样。**

按后续要求，单节点追加计算已取消，未完成的中间输出已清理。批处理脚本为 [validation_9node.pbs](validation_9node.pbs)。申请 `regular-g` 的 9 个节点，每节点一个进程，时限一小时，项目为 `gw97`。`short-g` 的节点上限为 8，因此这里使用 `regular-g`。

每个节点独立运行完整的实验 B，每组 5120 个构型，包括 Haar 2000 步、上述十个细格起点、前向数据及 holonomy 统计。继续使用 epoch 300 EMA、$\epsilon=10^{-5}$ 和完整 2000 步网格的对应后缀。节点内的编译缓存使用本地临时目录。

编号偏移 $b=1,\ldots,9$，每批独立基准取 $20j+b$，输入取 $20j+b+10$，其中 $j=0,\ldots,5119$。随机种子为 $20261009+b$。这九批与已完成的偏移 $b=0$ 批次均不重合，输入和基准之间也不重合。每个采样起点新增 $9\times5120=46080$ 个构型，连同已有 5120 个，可达到 51200 个。后续合并时需要按父构型重算误差，不能只平均各批误差。

从仓库根目录提交。

```bash
qsub src/Prolongation/validation_9node.pbs
```

各节点输出使用 `results/su2_network_validation/validation_n5120_offset1` 至 `validation_n5120_offset9` 前缀，分别保存 HDF5、日志、20 页 PDF、PNG 和统计 CSV。已有同名日志或构型文件时会停止，避免覆盖结果。单节点既有运行约需 40 至 45 分钟，一小时时限留给编译、文件读写及统计，但实际耗时仍取决于节点与文件系统负载。

作业 `3470180` 于 2026-10-01 23 时 11 分启动，运行 74 秒后在编译预热阶段退出，退出码为 1。批处理环境使用了 `nvc`，它不接受 Triton C 扩展编译所传入的 `-Wno-psabi`，见 [九节点错误日志](../../results/su2_network_validation/validation_9node_failed_3470180.log) 和 [PBS 输出](../../su2_validation_9n.o3470180)。MPI 在首进程失败后终止了其余节点。九个 HDF5 的构型组均只有 `input` 和 `reference`，尚未生成反向样本。

作业 `3471314` 已使用 `gcc/g++`，但上次失败的同名日志仍然存在，触发 `set -o noclobber`，九个进程均在启动 Python 前退出。见 [PBS 输出](../../su2_validation_9n.o3471314)。HDF5 使用独占创建模式，因此重新运行前也必须处理同名 HDF5。

2026-10-02 已确认九个 HDF5 未生成反向样本并将其清理，九份旧日志合并保存在上述错误日志中。九个输出前缀已可用。PBS 保留 `CC=/usr/bin/gcc`、`CXX=/usr/bin/g++`。GH200 上使用 epoch 300 EMA 和 5120 构型的实际输入，从空缓存完成三次 EM 编译预热，共 45.48 秒，最大单位四元数范数误差为 $2.98\times10^{-7}$，临时编译缓存已删除。见 [编译检查](../../results/su2_network_validation/validation_9node_compiler_checks.log)。提交脚本通过 Bash 语法检查。此次 `qsub` 被工具策略拒绝，提示当前环境全局禁用 HPC 作业提交，尚无新作业号。可在终端执行上面的提交命令。

**实验 C 的 $\Sigma^2\simeq3$、512 构型测试。**

本次比较 `uniform`、`interior`、`relaxed` 三种细化初态，三种版本使用同一批 512 个不同的粗格父构型。同时计算 512 个真实细格起点对照、512 个 Haar 2000 步对照，以及独立的 512 个平衡细格参考构型。模型继续使用 epoch 300 的 EMA 权重。

从完整 2000 步网格选出最接近 $\Sigma^2=3$ 的起点，实际为

$$
N=1070,\qquad t_\star=0.5350046158,\qquad \Sigma_\star^2=3.0012588776.
$$

三个细化版本和真实细格对照均重新加噪到这个累计方差，再沿完整网格的最后 1070 步反向生成。前向热核分为四个增量，每个方差 $0.7503147194$。终止时间为 $\epsilon=10^{-5}$，末步只保留漂移。Haar 对照仍运行完整 2000 步。所有前向和反向噪声独立抽样，种子与父构型编号保存在 HDF5 中。

图中的 `Haar 2000` 表示每条 link 独立从 SU(2) Haar 分布取初值，再用同一个 score net 做完整 2000 步反向 SDE，从 $t=1$ 积分到 $\epsilon=10^{-5}$。它的起始累计方差为 $\Sigma(1)^2=36.945010$。`Fine start` 表示真实细格加噪到上述 $\Sigma_\star^2$ 后的 1070 步反向生成。两者分别控制完整生成与相同噪声起点的网络误差。

`Fine equilibrium` 则是热浴加过松弛 Monte Carlo 得到的独立平衡细格构型，直接测量作为参考，不经过加噪或反向生成。它与 `Fine start` 是不同的系综。

粗格输入从已有 4096 个构型中取 $4,12,\ldots,4092$。真实细格输入取 $100,300,\ldots,102300$，独立参考取 $0,200,\ldots,102200$。细化与局部更新沿用实验 A 的函数和轮数，随后施加逐站点 Haar 规范变换。误差仍按四个父构型一块、1000 次 bootstrap 计算，分布检验使用 999 次块置换。

粗格信息保留按对应物理路径比较。粗 $1\times1$ 与细 $2\times2$、粗 $2\times2$ 与细 $4\times4$ 都使用偶偶锚点。粗格绕圈迹与细格偶数横向位置的绕圈迹比较。两个方向的交换子迹固定在共同起点。真实细格对照则与自身原始输入比较，使用相同的锚点与横向位置。

首轮 512 样本使用下列参数。下文 epoch 300 的 5120 样本测试取消了 Fine start 的反向生成，epoch 4 测试则按后续要求加入这条对照。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python3 src/Prolongation/refine_su2.py sample \
  --coarse 'results/su2_prolongation_forward/cfgs_su2_beta=0.1408904756906944_16^2.h5' \
  --initializations fine uniform interior relaxed \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --forward results/su2_prolongation_forward/scan_tau10.h5 \
  --checkpoint results/su2_L32_beta2.0_lcnn_hk_c16l3/ckpt_epoch0300.pt \
  --out results/su2_prolongation_reverse/reverse_tau3_n512.h5 \
  --seed 20261020 --device cuda --n-cfgs 512 --eps 1e-5 --steps 1070
```

结果见 [18 页比较图](../../results/su2_prolongation_reverse/reverse_tau3_n512.pdf)、[Wilson 均值预览](../../results/su2_prolongation_reverse/reverse_tau3_n512.png) 和 [各阶段构型与观测量](../../results/su2_prolongation_reverse/reverse_tau3_n512.h5)。独立细格参考的 plaquette 为 $0.433414\pm0.000527$。下表误差为一个 bootstrap 标准误差，两个整体检验均相对独立细格参考。

| 起点 | 反向步数 | 输出 $\langle P\rangle$ | 原有观测量整体 $p$ | holonomy 整体 $p$ |
| --- | --- | --- | --- | --- |
| 真实细格加噪 | 1070 | $0.432783\pm0.000556$ | $0.995$ | $0.955$ |
| Haar | 2000 | $0.433041\pm0.000618$ | $0.979$ | $0.378$ |
| uniform 加噪 | 1070 | $0.432886\pm0.000579$ | $0.853$ | $0.078$ |
| interior 加噪 | 1070 | $0.431856\pm0.000477$ | $0.225$ | $0.904$ |
| relaxed 加噪 | 1070 | $0.433164\pm0.000496$ | $0.574$ | $0.886$ |

在 512 个样本下，三个细化输出的整体检验均未达到 $p<0.05$。这不证明完整分布相等或网络无偏。逐项比较中，interior 的 $\langle\overline W_{11}\overline W_{12}\rangle$ 比参考低 $0.001968\pm0.000602$，约 $3.27$ 个标准误差。uniform 的交换子迹均值比参考高 $0.08036\pm0.03307$，约 $2.43$ 个标准误差。这些是多项比较中的未校正结果，不能将整体检验不显著解释为每个量都在两个标准误差以内。

反向前的分布也需要单独判断。真实细格加噪后的 plaquette 为 $0.004805\pm0.000674$，uniform、interior、relaxed 分别为 $0.011079\pm0.000654$、$0.007433\pm0.000674$、$0.006816\pm0.000655$。与相同噪声的真实细格相比，原有观测量的整体 $p$ 分别为 $0.001$、$0.085$、$0.337$。uniform 的 plaquette 差异为 $6.68$ 个标准误差，因此 $\Sigma^2\simeq3$ 时其加噪分布仍能被区分。

粗格信息保留是另一项要求。反向输出的偶偶锚点 $2\times2$ 回路，与原粗格 $1\times1$ 回路的相关性如下。逐点相关性仍按父构型分块估计误差。

| 细化版本 | 逐点相关系数 | 构型空间平均相关系数 |
| --- | --- | --- |
| uniform | $-0.00064\pm0.00260$ | $0.0812\pm0.0390$ |
| interior | $0.00085\pm0.00285$ | $0.0940\pm0.0398$ |
| relaxed | $-0.00087\pm0.00271$ | $0.0134\pm0.0407$ |

多数回路、绕圈迹及交换子迹的相关性与零相容。uniform 和 interior 的上述构型平均相关性有约两个标准误差的正信号，interior 的逐点 $4\times4$ 回路还有 $-0.00726\pm0.00258$ 的负相关。这些来自三个版本共 27 项反向保留比较，未作多重比较校正，尚不足以确认稳定的粗格信息保留。因此本次完成了生成测试，但尚未达到同时匹配分布并显著保留粗格信息的验收目标。

图中第 1 至 5 页为 Wilson 回路均值、相对参考的差、二阶矩、逐回路方差和空间平均方差。第 6 至 8 页为协方差及奇偶锚点差异，第 9 至 11 页为 Polyakov 与 holonomy 矩，第 12 至 13 页为最终分布。第 14 页比较细化、加噪、反向三个阶段，第 15 至 17 页给出对应物理路径的输入保留，第 18 页给出加噪场和最终输出的整体检验。

全部逐项结果见 [统计表](../../results/su2_prolongation_reverse/reverse_tau3_n512_statistics.csv)、[独立参考及两个生成对照的比较](../../results/su2_prolongation_reverse/reverse_tau3_n512_comparison.csv)、[粗格信息保留](../../results/su2_prolongation_reverse/reverse_tau3_n512_retention.csv) 和 [前向热核检验](../../results/su2_prolongation_reverse/reverse_tau3_n512_heat_kernel.csv)。误差按父构型计算，没有把同一构型中的格点或绕圈路径当成独立样本。

全部 14 组细格构型和原粗格通过重读检查。最大矩阵误差为 $3.51\times10^{-16}$，观测量重算误差为 $3.78\times10^{-15}$，交换子迹的独立 SU(2) 迹恒等式误差为 $1.02\times10^{-14}$。float32 采样投影前最大范数漂移为 $7.16\times10^{-6}$。五套反向积分网格均逐点等于完整网格的对应后缀。uniform 和 interior 在随机规范变换后仍保持与粗格对应的物理路径。24 项前向热核衰减检查的最大残差为 $2.25$ 个标准误差，见 [检查记录](../../results/su2_prolongation_reverse/reverse_tau3_n512_checks.log)。

块内与全链接 Metropolis 接受率分别为 $0.614758$ 和 $0.662930$。细化准备约 $3.65$ 秒，GH200 上五组反向采样合计 $142.24$ 秒，采样、原有观测量与写出合计 $147.66$ 秒。这些计时不含进程启动、前置编译预热、后续 holonomy 分析及绘图，见 [运行日志](../../results/su2_prolongation_reverse/reverse_tau3_n512.log)。本次固定使用完整 2000 步的分辨率，没有另做积分收敛扫描，不能分离网络误差与 EM 离散误差。

**实验 C 的 5120 构型与 Haar 2000 比较。**

按后续要求，每个细化版本和 Haar 对照分别重新生成 5120 个构型。沿用 $\Sigma_\star^2=3.0012588776$、1070 步完整网格后缀、epoch 300 EMA 和 $\epsilon=10^{-5}$。本轮不执行 Fine start 的反向生成。图中 Haar 2000、uniform、interior、relaxed 四组并列画点，蓝色虚线和误差带表示 Fine equilibrium。末页的整体检验同时给出相对 Haar 2000 和 Fine equilibrium 的结果。逐项差值及误差保存在比较 CSV 中。真实细格另做前向加噪，用于检查三个细化场的加噪分布。

粗格父构型由原 `su2_2d_heatbath.jl` 重新生成，共 5120 个。热化 2000 次更新，保存间隔 160 次，每次仍含一次热浴和三次过松弛。这个间隔等于首轮从间隔 20 的链中每隔八个构型取样。粗格种子为 `20261021`，生成与统计种子为 `20261022`。三种细化版本共享同一批粗格父构型，各自独立抽样加噪及反向噪声。没有复制粗格构型凑足样本数。

新粗格的 plaquette 为 $0.034837\pm0.000465$，与无限体积值 $0.035194$ 相差 $0.77$ 个标准误差。首次非正截断的自相关时间为 $0.563$ 个保存间隔。完整链生成约需 $175.7$ 秒，见 [粗格生成日志](../../results/su2_prolongation_reverse/coarse_n5120.log)。细格前向对照取编号 $20j+10$，独立平衡参考取 $20j$，其中 $j=0,\ldots,5119$。误差继续使用四个父构型一块的 bootstrap。

```bash
julia --startup-file=no -t 4 trainningdata/su2_2d_heatbath.jl \
  --beta 0.1408904756906944 --L 16 --n_cfgs 5120 \
  --n_therm 2000 --n_skip 160 --n_or 3 --seed 20261021 \
  --out results/su2_prolongation_reverse

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python3 src/Prolongation/refine_su2.py sample \
  --coarse 'results/su2_prolongation_reverse/cfgs_su2_beta=0.1408904756906944_16^2.h5' \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --forward results/su2_prolongation_forward/scan_tau10.h5 \
  --checkpoint results/su2_L32_beta2.0_lcnn_hk_c16l3/ckpt_epoch0300.pt \
  --out results/su2_prolongation_reverse/reverse_tau3_n5120.h5 \
  --seed 20261022 --device cuda --n-cfgs 5120 --eps 1e-5 --steps 1070
```

结果已保存为 [18 页比较图](../../results/su2_prolongation_reverse/reverse_tau3_n5120.pdf)、[Wilson 均值预览](../../results/su2_prolongation_reverse/reverse_tau3_n5120.png) 和 [构型与逐构型观测量](../../results/su2_prolongation_reverse/reverse_tau3_n5120.h5)。第 1 页将 Haar 2000、uniform、interior、relaxed 四组并列。第 2 页单独画 action density，采用本文件的 Wilson 作用量约定

$$
\frac{S}{V}=\beta\left(1-\overline P\right),\qquad V=32^2,\qquad\beta=2.
$$

它由逐构型 plaquette 线性换算，均值误差为 plaquette 误差的两倍，没有增加独立的物理信息。蓝色参考线和误差带始终表示 Fine equilibrium。其 plaquette 为 $0.433127\pm0.000173$，action density 为 $1.133746\pm0.000346$。四组生成结果如下，表中误差为一个标准误差。

| 起点 | 输出 $\langle P\rangle$ | 输出 $\langle S/V\rangle$ | 相对 Haar 的原有观测量整体 $p$ | 相对 Haar 的 holonomy 整体 $p$ |
| --- | --- | --- | --- | --- |
| Haar 2000 | $0.432888\pm0.000182$ | $1.134225\pm0.000364$ | 基准 | 基准 |
| uniform | $0.432805\pm0.000176$ | $1.134391\pm0.000352$ | $0.899$ | $0.438$ |
| interior | $0.432648\pm0.000173$ | $1.134705\pm0.000346$ | $0.463$ | $0.506$ |
| relaxed | $0.432959\pm0.000182$ | $1.134082\pm0.000365$ | $0.994$ | $0.233$ |

三种细化输出相对 Haar 2000 的整体检验均未检出显著差异，所测逐项矩的最大差异为 $2.25$ 个标准误差。相对 Fine equilibrium，原有观测量整体 $p$ 分别为 $0.484$、$0.481$、$0.240$，holonomy 整体 $p$ 分别为 $0.259$、$0.417$、$0.228$。这些结果限定于所测特征和当前样本量，不证明完整分布相等。

本批 Haar 2000 相对 Fine equilibrium 的原有观测量整体 $p=0.155$，holonomy 整体 $p=0.043$，后者提示仍需复核的差异，涉及 $y$ 方向绕圈分布。它是多个系综比较中的未校正结果，不能单凭这一批确认系统误差。保留独立平衡参考可以检查 Haar 基准本身的偏差。本次仍未分离网络误差和 EM 离散误差。

较高统计量也改变了前向匹配的判断。uniform、interior、relaxed 加噪场相对相同噪声的真实细格，原有观测量整体检验均为 $p=0.001$，即 999 次置换的最低可报告值。三者的加噪 plaquette 分别为 $0.010553\pm0.000212$、$0.007928\pm0.000222$、$0.006885\pm0.000217$，真实细格对照为 $0.004543\pm0.000213$。因此不能将首轮 512 样本中未检出的前向差异解释为分布已匹配。

反向后，细格偶偶锚点 $2\times2$ 回路与粗格 $1\times1$ 回路的构型平均相关系数，三个版本依次为 $-0.0135\pm0.0138$、$-0.0359\pm0.0144$、$-0.0273\pm0.0144$。逐点相关系数均与零相容。其余绕圈迹和交换子迹也没有稳定的正相关信号。interior 有少数约两个标准误差的负相关，属于 27 项未校正比较中的结果，尚不足以确认稳定的输入依赖。本轮支持所测最终分布接近 Haar 生成分布，但没有达到同时匹配前向分布并显著保留粗格信息的验收目标。

第 3 至 5 页给出 Wilson 二阶矩和两种方差，第 6 至 8 页给出协方差与奇偶锚点差异。第 9 至 11 页为 Polyakov 和 holonomy 矩，第 12 至 13 页为最终分布，第 14 页为三个生成阶段，第 15 至 17 页为输入保留。第 18 页用圆点表示相对 Haar 2000 的检验，用方点表示相对 Fine equilibrium 的检验。Haar 本身只与 Fine equilibrium 比较。

完整数值见 [统计表](../../results/su2_prolongation_reverse/reverse_tau3_n5120_statistics.csv)、[两个参考的逐项比较](../../results/su2_prolongation_reverse/reverse_tau3_n5120_comparison.csv)、[粗格信息保留](../../results/su2_prolongation_reverse/reverse_tau3_n5120_retention.csv) 和 [前向热核检验](../../results/su2_prolongation_reverse/reverse_tau3_n5120_heat_kernel.csv)。原 512 样本结果保留，本批结果没有与它直接合并。

13 组细格数据及原粗格的全部 5120 个构型通过重读检查。最大矩阵误差为 $4.00\times10^{-16}$，观测量重算误差为 $4.11\times10^{-15}$，交换子迹恒等式误差为 $1.12\times10^{-14}$。float32 采样投影前最大范数漂移为 $7.83\times10^{-6}$。24 项前向热核检查的最大残差为 $2.57$ 个标准误差。四套反向网格、父构型编号、统计误差和对应物理路径均通过核对，见 [检查记录](../../results/su2_prolongation_reverse/reverse_tau3_n5120_checks.log)。

块内与全链接 Metropolis 接受率分别为 $0.615611$ 和 $0.663469$。三种细化初态准备共用 $64.96$ 秒。Haar 2000 反向采样用 $389.57$ 秒，三个 1070 步采样分别用 $206.26$、$206.15$、$206.04$ 秒，四组反向采样合计 $1008.03$ 秒。采样、原有观测量及构型写出合计 $1090.22$ 秒，不含粗格生成、细化准备、前置编译预热、后续 holonomy 分析及绘图，见 [运行日志](../../results/su2_prolongation_reverse/reverse_tau3_n5120.log)。

**epoch 4 网络的 Fine start、Haar 与 Relaxed 比较。**

本轮使用 `results/su2_L32_beta2.0_lcnn_hk_c16l3_n102400_b256_e500/ckpt_epoch0004.pt` 的 EMA 权重。checkpoint 中的 $L=32$、$\beta=2$、$\sigma=14$、通道数 16 和三层网络与上述采样设置一致。每组仍为 5120 个构型。Haar 对照运行完整 2000 步，Fine start 和 Relaxed 从 $\Sigma_\star^2=3.0012588776$ 出发，沿同一完整网格的最后 1070 步反向生成。

图中并列显示 Fine start、Haar 2000 和 Relaxed，蓝色虚线及误差带仍表示 Fine equilibrium。沿用现有 5120 个粗格父构型、细格参考、细化随机流、加噪和反向随机种子以及 bootstrap 权重。Relaxed 的构造仍包含两轮块内更新和一轮全链接更新。Uniform 和 Interior 不作为单独生成结果。输出使用新前缀，保留 epoch 300 的结果。

Fine start 的输入取自同一平衡 Monte Carlo 数据的 $20j+10$ 号构型，独立参考取 $20j$ 号构型，$j=0,\ldots,5119$。前向加噪使用四个方差 $0.7503147194$ 的热核增量，种子为 20282092。直接从已保存的 `forward_steps1070` 补做反向生成，种子为 20272092。Haar 与 Relaxed 的已有构型沿用。下面命令可从头重现三条分支，输出文件须尚未存在。

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python3 src/Prolongation/refine_su2.py sample \
  --coarse 'results/su2_prolongation_reverse/cfgs_su2_beta=0.1408904756906944_16^2.h5' \
  --initializations fine relaxed \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --forward results/su2_prolongation_forward/scan_tau10.h5 \
  --checkpoint results/su2_L32_beta2.0_lcnn_hk_c16l3_n102400_b256_e500/ckpt_epoch0004.pt \
  --out results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004.h5 \
  --seed 20261022 --device cuda --n-cfgs 5120 --eps 1e-5 --steps 1070
```

结果见 [epoch 4 的 18 页比较图](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004.pdf) 和 [Wilson 均值预览](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004.png)。第 2 页保留 action density，第 9 至 11 页保留 winding loops 与 holonomy，第 18 页同时比较 Haar 和 Fine equilibrium。下表误差为一个标准误差，整体检验均相对 Fine equilibrium。

| 系综 | $\langle P\rangle$ | $\langle P^2\rangle$ | 原有观测量整体 $p$ | holonomy 整体 $p$ |
| --- | --- | --- | --- | --- |
| Fine equilibrium | $0.433127\pm0.000173$ | $0.350417\pm0.000120$ | 参考 | 参考 |
| Fine start 1070 | $0.431735\pm0.000169$ | $0.348004\pm0.000116$ | $0.001$ | $0.159$ |
| Haar 2000 | $0.431896\pm0.000182$ | $0.348229\pm0.000131$ | $0.001$ | $0.134$ |
| Relaxed 1070 | $0.431984\pm0.000183$ | $0.348268\pm0.000138$ | $0.001$ | $0.221$ |

Fine start 的 plaquette 均值比独立平衡参考低 $0.001392\pm0.000242$，约 $5.76$ 个标准误差。其二阶矩低 $0.002413\pm0.000167$，约 $14.48$ 个标准误差。Haar 与 Relaxed 的二阶矩分别偏低 $12.34$ 和 $11.76$ 个标准误差。因此即使从真实平衡细格加噪，当前反向流程仍未恢复这些平衡统计量，偏差不能仅归因于细化起点。本次没有分离网络误差和 EM 离散误差。

Fine start 相对 Haar 的原有观测量与 holonomy 整体 $p$ 分别为 $0.406$ 和 $0.300$。Relaxed 相对 Haar 分别为 $0.985$ 和 $0.891$，相对 Fine start 分别为 $0.735$ 和 $0.395$。这三个生成系综之间的整体检验未检出显著差异，但不能由此断言完整分布相同。Fine start 输出的偶偶锚点 $2\times2$ 回路空间平均与原始细格输入的相关系数为 $0.00224\pm0.01363$，也未见显著输入保留。

完整结果保存在 [构型与观测量](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004.h5)、[统计表](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004_statistics.csv)、[逐项比较](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004_comparison.csv)、[输入保留](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004_retention.csv) 和 [前向热核检验](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004_heat_kernel.csv)。Fine start 的 GPU 反向采样耗时 $231.89$ 秒，见 [运行日志](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004.log)。

八组细格数据及原粗格的全部构型通过重读检查。最大矩阵误差为 $4.00\times10^{-16}$，观测量重算误差为 $3.89\times10^{-15}$，交换子迹恒等式误差为 $1.04\times10^{-14}$。三条反向网格、父构型编号、随机种子和统计误差均通过核对。12 项前向热核检查的最大残差为 $2.57$ 个标准误差，见 [检查记录](../../results/su2_prolongation_reverse/reverse_tau3_n5120_epoch0004_checks.log)。

**1024 个训练构型的 epoch 500 网络比较。**

本轮使用 `results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500/ckpt_epoch0500.pt` 的 EMA 权重。网络使用固定 1024 个 L32 平衡构型，从随机初始化以 batch 64 训练 500 个 epoch，共更新 8000 次。训练记录见 [SU(2) 训练说明](../SU2/README.md)。

生成设置与前面的 epoch 4 比较一致。Fine start、Haar 2000 和 Relaxed 各生成 5120 个构型，Fine equilibrium 为参考。两个部分反向过程从 $\Sigma_\star^2=3.0012588776$ 出发，使用完整 2000 步网格的最后 1070 步，$\epsilon=10^{-5}$，末步只保留漂移。保留 action density 和 holonomy，Uniform 与 Interior 不单独生成。

粗格父构型、细格输入、平衡参考、细化过程和各条随机流均沿用 epoch 4 比较的设置。Fine start 的原始细格与平衡参考互不重叠。其中各有 50 和 49 个构型属于这次的 1024 个训练构型，分别占各自 5120 个构型的约 1%。这轮比较保留了原参考系综，参考与训练数据存在少量重叠。

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python3 src/Prolongation/refine_su2.py sample \
  --coarse 'results/su2_prolongation_reverse/cfgs_su2_beta=0.1408904756906944_16^2.h5' \
  --initializations fine relaxed \
  --fine 'trainningdata/cfgs_su2_beta=2.0_32^2.h5' \
  --forward results/su2_prolongation_forward/scan_tau10.h5 \
  --checkpoint results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500/ckpt_epoch0500.pt \
  --out results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.h5 \
  --seed 20261022 --device cuda --n-cfgs 5120 --eps 1e-5 --steps 1070
```

结果见 [1024 构型训练网络的 18 页比较图](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.pdf) 和 [Wilson 均值预览](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.png)。版式与 epoch 4 的图一致。下表误差为一个标准误差，整体检验均相对 Fine equilibrium。

| 系综 | $\langle P\rangle$ | $\langle P^2\rangle$ | 原有观测量整体 $p$ | holonomy 整体 $p$ |
| --- | --- | --- | --- | --- |
| Fine equilibrium | $0.433127\pm0.000173$ | $0.350417\pm0.000120$ | 参考 | 参考 |
| Fine start 1070 | $0.432310\pm0.000169$ | $0.349358\pm0.000115$ | $0.001$ | $0.533$ |
| Haar 2000 | $0.432478\pm0.000182$ | $0.349589\pm0.000131$ | $0.001$ | $0.040$ |
| Relaxed 1070 | $0.432549\pm0.000183$ | $0.349622\pm0.000138$ | $0.001$ | $0.244$ |

相对 epoch 4，三组 plaquette 均值和二阶矩更接近平衡参考。二阶矩仍分别偏低 $6.37$、$4.68$、$4.36$ 个标准误差。$1\times2$ 回路均值的偏差则增大，分别偏低 $6.48$、$5.09$、$5.25$ 个标准误差。不同观测量的改善并不一致，三组原有观测量整体检验均仍为 $p=0.001$。本轮没有分离网络误差与 EM 离散误差。Haar 的 holonomy 整体 $p=0.040$ 是多个系综比较中的未校正结果，应作为需要复核的迹象。

Fine start 相对 Haar 的原有观测量与 holonomy 整体 $p$ 分别为 $0.493$、$0.356$。Relaxed 相对 Haar 分别为 $0.981$、$0.422$，相对 Fine start 分别为 $0.620$、$0.720$。三种生成起点之间的整体检验未检出显著差异。Fine start 输出的偶偶锚点 $2\times2$ 回路空间平均与原始细格输入的相关系数为 $0.00222\pm0.01363$，Relaxed 对应的细 $2\times2$ 与粗 $1\times1$ 回路空间平均相关系数为 $-0.02749\pm0.01443$，未见稳定的正输入保留。

全部结果保存为 [构型与观测量](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.h5)、[统计表](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500_statistics.csv)、[逐项比较](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500_comparison.csv)、[输入保留](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500_retention.csv) 和 [前向热核检验](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500_heat_kernel.csv)。细化准备耗时 $41.11$ 秒。Haar、Fine start、Relaxed 的 GPU 反向采样分别用 $389.41$、$205.95$、$206.03$ 秒。采样、原有观测量及构型写出合计 $852.93$ 秒，不含细化准备、编译预热、holonomy 分析和绘图，见 [运行日志](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.log)。

八组细格数据和原粗格的全部 5120 个构型通过重读检查。最大矩阵误差为 $4.00\times10^{-16}$，观测量重算误差为 $3.89\times10^{-15}$，交换子迹恒等式误差为 $1.04\times10^{-14}$。三条反向网格、种子、父构型编号和统计误差均通过核对。共享的六组粗格、细格及加噪输入与 epoch 4 数据逐元素完全相同。12 项前向热核检查的最大残差为 $2.57$ 个标准误差，见 [检查记录](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500_checks.log)。

**补充 $\Sigma^2\simeq1.5$ 的 Fine start 与 Relaxed。**

继续使用同一个 1024 构型训练的 epoch 500 EMA 网络，以及同一批 Fine 和 Relaxed 初态，各补充 5120 个输出。完整 2000 步网格中最接近 $\Sigma^2=1.5$ 的起点为

$$
N=829,\qquad t_\star=0.4145058692,\qquad \Sigma_\star^2=1.4997139132.
$$

前向加噪使用两个方差 $0.7498569566$ 的热核增量。Fine start 的前向和反向种子分别为 20281851、20271851，Relaxed 分别为 20581851、20571851。反向仍沿完整网格的后缀积分到 $\epsilon=10^{-5}$，末步只保留漂移。

原 HDF5 增加 `forward_steps829`、`fine_steps829`、`relaxed_forward_steps829` 和 `relaxed_steps829` 四组数据。原 1070 步结果和 Haar 2000 沿用。文件名保留 `reverse_tau3_n5120_train1024_epoch0500`，同一比较图更新为 Fine start 的两个起点、Haar 2000、Relaxed 的两个起点，共五组并列结果，Fine equilibrium 继续作为参考。

`sample` 现支持在实验 C 中传入 `--steps 829 1070`，可从头重现这两个起点。统计比较分别以对应起点的加噪 Fine 数据为前向参考，以 Fine equilibrium 和 Haar 2000 为反向参考。原 1070 步比较使用相同的置换随机流。输入保留 CSV 增加 `tau_start` 和 `reverse_steps`，区分各自的初态、加噪和反向阶段。

两组新增结果已加入同一份 [18 页比较图](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.pdf) 和 [Wilson 均值预览](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500.png)。方形标记对应 $\Sigma^2\simeq1.5$，圆形标记对应 $\Sigma^2\simeq3$，Haar 2000 单独并列。下表误差为一个标准误差，整体检验均相对 Fine equilibrium。

| 新增系综 | $\langle P\rangle$ | $\langle P^2\rangle$ | 原有观测量整体 $p$ | holonomy 整体 $p$ |
| --- | --- | --- | --- | --- |
| Fine start 829 | $0.432401\pm0.000183$ | $0.349584\pm0.000119$ | $0.001$ | $0.676$ |
| Relaxed 829 | $0.433677\pm0.000183$ | $0.350255\pm0.000126$ | $0.010$ | $0.239$ |

Fine start 829 与 Haar 2000 的原有观测量和 holonomy 整体 $p$ 分别为 $0.613$、$0.886$。Relaxed 829 相对 Haar 分别为 $0.002$、$0.356$，相对 Fine start 829 分别为 $0.001$、$0.133$。因此在这个较短起点，Relaxed 的局部观测量已与 Haar 和 Fine start 检出差异。其 plaquette 均值比平衡参考高 $2.18$ 个标准误差，二阶矩低 $0.93$ 个标准误差，原有观测量整体检验仍为 $p=0.010$，不能据此判定已经恢复平衡分布。

Fine start 829 的偶偶锚点 $2\times2$ 回路空间平均与原始细格输入的相关系数为 $-0.00433\pm0.01344$。Relaxed 对应细 $2\times2$ 与粗 $1\times1$ 回路空间平均的相关系数为 $0.00534\pm0.01407$，这两项均未见显著输入保留。两组新增 GPU 反向采样分别耗时 $159.11$ 和 $159.48$ 秒。

十二组细格数据及原粗格的全部构型通过重读检查。反向时间网格、种子、观测量和统计误差核对通过。24 项前向热核检查的最大残差为 $2.57$ 个标准误差。原 $\Sigma^2\simeq3$ 及 Haar 2000 的统计、比较、输入保留和热核表项与补充前逐项完全一致，见 [检查记录](../../results/su2_prolongation_reverse/reverse_tau3_n5120_train1024_epoch0500_checks.log)。

**十节点、三个起点的 epoch 500 比较。**

[reverse_10node.pbs](reverse_10node.pbs) 申请 `regular-g` 的 10 个节点，墙钟时间为一小时，每个节点运行一个进程。继续使用 1024 个训练构型的 epoch 500 EMA 网络。每个节点生成七组结果，每组 5120 个构型，合并后图上每一点对应 51200 个构型，七组反向输出合计 358400 个。

| 起点 | 实际 $\Sigma_\star^2$ | 反向步数 | 生成系综 |
| --- | --- | --- | --- |
| 约 1 | $0.9997011647$ | 696 | Fine start、Relaxed |
| 约 1.5 | $1.4997139132$ | 829 | Fine start、Relaxed |
| 约 3 | $3.0012588776$ | 1070 | Fine start、Relaxed |
| Haar | $36.945010$ | 2000 | Haar 2000 |

三个部分反向过程均使用原 2000 步网格的后缀，终止于 $\epsilon=10^{-5}$，末步仅保留漂移。Fine equilibrium 仍作为参考。图中用三角形、方形、圆形分别标记三个部分反向起点，保留 Wilson loops、action density 和 holonomy。

节点编号为 $r=0,\ldots,9$。细格输入取 $20j+r+10$，平衡参考取 $20j+r$，其中 $j=0,\ldots,5119$。两组各有 51200 个不同的原始细格构型，彼此不重叠。其中分别有 509 和 515 个构型属于训练子集。采样基础种子为 $20271002+100r$，130 条前向及反向噪声流的整数种子彼此不同。

每个节点先独立生成 5120 个 L16 粗格父构型，$\beta_c=0.1408904756906944$，种子为 $20371002+r$。热化 2000 次更新，每隔 160 次保存，每次更新包含一次热浴和三次过松弛，Julia 使用 4 个线程。粗格数据随后完整写入各节点的采样 HDF5，临时粗格文件随节点任务结束清理。

十个节点全部成功后，首节点在剩余墙钟时间内自动合并。构型和逐构型观测量完整复制到一个 HDF5，重新计算 1000 次块 bootstrap 和 999 次块置换。细格分支及平衡参考恢复为原 Monte Carlo 编号顺序，粗格与 Relaxed 分支按各自独立链的顺序保留。误差按合并后的父构型计算，没有平均节点误差。合并完成并生成 18 页 PDF 后，删除节点中间数据和图表，保留十份节点日志。输出文件不依赖这些中间文件。

```bash
qsub src/Prolongation/reverse_10node.pbs
```

输出位于 `results/su2_prolongation_reverse`。公共前缀为 `reverse_tau1_1p5_3_train1024_epoch0500_<作业号数字部分>`，节点日志后缀为 `_rank0_n5120.log` 至 `_rank9_n5120.log`，合并结果后缀为 `_n51200.h5`、`_n51200.pdf`、`_n51200.png`、`_n51200.log` 和对应统计 CSV。作业号用于区分重复提交的输出，同样参数重跑使用相同随机种子。

按现有单节点计时，七组 GPU 反向采样约需 23 分钟，另有粗格生成、编译、测量、合并和绘图。一小时包含上述全部阶段，实际耗时仍取决于节点与共享存储负载。最终完整数据约为 135 GB，合并期间还需容纳节点副本。

PBS 已通过 Bash 语法检查，Python 已通过语法检查，Julia 依赖可用。十份小型 CPU 测试数据验证了父构型排序、全部构型及观测量的复制、统计误差重算、重复父构型拒绝，以及删除节点文件后合并数据仍可独立读取。七点图的标签和布局已检查。测试记录见 [reverse_10node_checks.log](../../results/su2_prolongation_reverse/reverse_10node_checks.log)，其中测试数据仅用于检查合并流程，不代表新的物理结果。

作业 `3472755` 于 2026-10-02 01 时 16 分启动，墙钟运行 5 秒后以退出码 1 结束，尚未生成构型。MPI 工作进程原来由 `bash` 直接执行脚本，没有执行登录环境初始化。远端缺少站点模块所需的 `MOD_GIT_ROOTDIR`，导致 `module load julia/1.12.6` 失败，MPI 随即停止整个作业，见 [PBS 错误输出](../../su2_reverse_10n.o3472755)。

现改为通过 `bash -l` 启动每个工作进程，由 Miyabi 的 `/etc/profile.d/zzz.sh` 设置模块环境，并显式指定工作盘上的 `JULIA_DEPOT_PATH`。验证时先清除继承的模块变量，复现原错误，再在已有分配节点上经 MPI 启动登录 shell。Julia 四线程生成 8 个启动测试构型、Python 重读 HDF5 并核对 plaquette、epoch 500 checkpoint 读取及 CUDA 可见性均通过，见 [MPI 启动检查](../../results/su2_prolongation_reverse/reverse_10node_startup_checks.log)。这是单节点 MPI 启动测试，测试构型已清理。

作业 `3472772` 已完成十节点采样及合并。2026-10-02 按要求重绘同一份 [13 页比较图](../../results/su2_prolongation_reverse/reverse_tau1_1p5_3_train1024_epoch0500_3472772_n51200.pdf) 和 [Wilson 均值预览](../../results/su2_prolongation_reverse/reverse_tau1_1p5_3_train1024_epoch0500_3472772_n51200.png)，去掉全部 $\Sigma^2\simeq1$ 的图示，只保留 $\Sigma^2\simeq1.5$、$3$ 及 Haar 2000。HDF5 和四份统计 CSV 保留全部原始结果，文件名沿用原前缀。重绘直接读取已有统计量。

```python
from pathlib import Path
import sys
sys.path.insert(0, "src/Prolongation")
from refine_su2 import plot_refinement

plot_refinement(
    Path("results/su2_prolongation_reverse/reverse_tau1_1p5_3_train1024_epoch0500_3472772_n51200.h5"),
    steps=[829, 1070],
)
```

当前比较图使用 $1\times1$、$1\times2$、$2\times2$ 和 $2\times3$ 回路。从已保存构型补算两个起点涉及的 12 组系综，每组仍为 51200 个构型。新增观测量、分布和统计量保存在原 HDF5 的 `wilson23` 子组，误差使用原有四构型分块和同一组 1000 次 bootstrap 权重。均值、二阶矩、方差、协方差、分布及三阶段变化图均使用 $W_{23}$，整体置换检验也替换相应观测量后重算。原四份 CSV 保留此前的 $W_{44}$ 分析，之前补算的 $W_{33}$ 保存在 `wilson33` 子组，上面的重绘调用使用当前的 $W_{23}$ 结果。

粗细格面积匹配的输入相关性图只保留细格 $2\times2$ 对粗格 $1\times1$。细格 $2\times3$ 没有相同物理尺寸的粗格整数回路，未将它与粗格 $2\times2$ 配对。平衡参考的 $\langle W_{23}\rangle=0.00665721\pm0.00007179$，理论值约为 $0.00660228$。

各向同性的正方形周期格子具有交换两个坐标轴的对称性，因此平衡期望值满足 $\langle W_{21}\rangle=\langle W_{12}\rangle$，单个构型上的两个测量不必相同。补算了平衡参考及五组最终生成系综的 $W_{21}$，按同一构型上的 $W_{21}-W_{12}$ 进行配对 bootstrap。六组差异的最大绝对显著性为 1.51 个标准误差，没有检出这两个方向的均值差异。逐构型空间平均保存为 `wilson23/<ensemble>/W21_per_config`，差值及误差保存为相应组的 `W21_minus_W12_mean` 和 `W21_minus_W12_se` 属性。两个方向可以先在每个构型内平均，再估计误差，构型数仍为 51200。

按要求删除 `Final winding-loop moments`、`Final holonomy eigenangles and commutator`、`Final holonomy distributions`、`Holonomy commutator retention`、`Second moments of spatially averaged winding traces` 和 `Winding-trace retention on even transverse lines` 六页。新增第 3 页有效弦张力后，比较图共 13 页。

`Residual block patterns after reverse diffusion` 按回路起点的坐标奇偶性分成 `00`、`01`、`10`、`11` 四类，比较 $1\times1$ 和 $2\times2$ 回路的均值及二阶矩相对平衡参考的偏差。它检查细化留下的 $2\times2$ 周期性空间差异。四类共同偏离零表示整体统计偏差，四类之间超过统计误差的差别才支持存在细化造成的空间不均匀性。

有效弦张力由各个回路的系综均值提取。

$$
\sigma_{\mathrm{eff}}a^2=-\frac{\ln\langle W(R,T)\rangle}{RT},
\qquad
\sigma_{\mathrm{exact}}a^2=-\ln\frac{I_2(2)}{I_1(2)}=0.8367233062.
$$

理论值为二维 SU(2) 的无限体积面积律结果，见 [Detmold et al. 的式 39 至 42](https://arxiv.org/pdf/2101.12668#page=9)。每次 bootstrap 先计算回路均值，再取对数，图示误差为所得有效弦张力的一个标准误差。五组最终生成系综和平衡参考使用的全部 bootstrap 回路均值均为正。结果保存于 `wilson23/<ensemble>/string_tension_mean`、`string_tension_se` 和 `string_tension_ci95`，依次对应 $1\times1$、$1\times2$、$2\times2$、$2\times3$。

| 系综 | $2\times2$ 的 $\sigma_{\mathrm{eff}}a^2$ | $2\times3$ 的 $\sigma_{\mathrm{eff}}a^2$ |
| --- | --- | --- |
| Fine equilibrium | $0.83564\pm0.00052$ | $0.83534\pm0.00180$ |
| Fine start，$\Sigma^2\simeq1.5$ | $0.84888\pm0.00056$ | $0.85638\pm0.00202$ |
| Fine start，$\Sigma^2\simeq3$ | $0.85002\pm0.00055$ | $0.85560\pm0.00201$ |
| Haar 2000 | $0.84985\pm0.00057$ | $0.85485\pm0.00199$ |
| Relaxed，$\Sigma^2\simeq1.5$ | $0.84508\pm0.00054$ | $0.85130\pm0.00196$ |
| Relaxed，$\Sigma^2\simeq3$ | $0.85001\pm0.00056$ | $0.85878\pm0.00207$ |

这些是不同有限回路给出的有效值，没有把生成系综的各个回路强行拟合为同一个渐近弦张力。Haar 2000 的 $2\times2$ 有效值比理论高约 1.6%。当前比较没有分离网络误差与反向积分误差。

## L64 到 L128 的前向扫描

细格取 $L_f=128$、$\beta_f=200$，粗格取 $L_c=64$、$\beta_c=50.3725225668$。两者满足 $F(\beta_c)=4F(\beta_f)$，其中 $F(\beta)=-\log[I_2(\beta)/I_1(\beta)]$。复用已有热浴系综，每套从 102400 个构型中选取编号 $200,600,\ldots,102200$ 的 256 个构型。扫描种子为 `20261002`。

细化构造和更新轮数与 L32 扫描相同。局部更新使用目标细格耦合 $\beta_f=200$，提案宽度为 $1/\sqrt{200}$。比较 `fine`、`uniform`、`interior`、`relaxed` 四种系综，横轴为 $\Sigma^2$，采用与参考图相同的 15 个累计方差值，范围为 0 到 10。每个噪声点独立加噪。误差按四个父构型一块、1000 次 bootstrap 计算。完整 plaquette 链用于检查自相关，其他回路检查选取后的父构型序列。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 src/Prolongation/refine_su2.py scan \
  --coarse 'trainningdata/cfgs_su2_beta=50.3725225668_64^2.h5' \
  --fine 'trainningdata/cfgs_su2_beta=200.0_128^2.h5' \
  --out results/su2_prolongation_forward/scan_L128_beta200_Sigma2.h5 \
  --seed 20261002 --device cpu --n-cfgs 256 \
  --Sigma2 0 0.02 0.05 0.1 0.2 0.4 0.8 1.6 2 2.5 3 4 6 8 10
```

结果见 [15 页比较图](../../results/su2_prolongation_forward/scan_L128_beta200_Sigma2.pdf)、[均值预览](../../results/su2_prolongation_forward/scan_L128_beta200_Sigma2.png)、[统计表](../../results/su2_prolongation_forward/scan_L128_beta200_Sigma2_statistics.csv) 和 [均值及分布比较](../../results/su2_prolongation_forward/scan_L128_beta200_Sigma2_comparison.csv)。完整构型与逐构型观测量保存在 [HDF5](../../results/su2_prolongation_forward/scan_L128_beta200_Sigma2.h5)。本次图示和统计表使用 $\Sigma^2$，HDF5 同时保存 `Sigma2` 和 `Sigma`。

数值与写出用时约 806 秒，不含绘图。全部代数检查通过。每个测量点重读抽查三个构型，最大矩阵误差为 $3.33\times10^{-16}$，回路重算误差为 $1.78\times10^{-15}$。224 个非零噪声热核衰减检验的最大残差为 3.03 个标准误差。检查结果见 [验证记录](../../results/su2_prolongation_forward/scan_L128_beta200_Sigma2_checks.log)。

按后续要求，首页均值图改为 $1\times1$、$2\times2$、$4\times4$、$8\times8$、$16\times16$、$32\times32$ 六种方形回路，排成两行三列。其余 14 页保留原有内容。三个大回路从已保存构型补算，使用同一组父构型和 bootstrap 权重。逐构型均值与误差保存在原 HDF5 的 `wilson_squares` 组，新增均值也加入原统计 CSV。

在新生成的扫描文件上补算并更新首页。

```python
from pathlib import Path
import sys
sys.path.insert(0, "src/Prolongation")
from refine_su2 import measure_square_scan, plot

path = Path("results/su2_prolongation_forward/scan_L128_beta200_Sigma2.h5")
measure_square_scan(path)
plot(path)
```

补算完成后，重绘只需调用 `plot(path)`。

本次清理删除了作业 `3472772` 的十份节点过程日志，以及作业 `3472755` 仅含启动信息的 rank 0 日志。保留合并日志、各项验证记录和旧作业的 PBS 错误输出。

## L64 到 L128 的反向测试

2026-10-02 从上述 L64、$\beta_c=50.3725225668$ 的 102400 个构型中，用 NumPy 种子 0 无放回随机抽取 512 个，排序后的父编号保存为 `parent_id`。复用 `prolong`、两轮内部 Metropolis 更新和一轮全格点更新，更新耦合为 200。准备种子为 20261002，内部、全格点更新和随机规范变换分别使用 `[seed,10]`、`[seed,11]`、`[seed,3,100]`。

使用新噪声方案训练的 L128、$\beta=200$、512 构型、batch 32、epoch 500 EMA 模型。沿用 L32 的时间步长规则，$\Delta t=(1-10^{-5})/2000$，选取最接近 $\Sigma^2=1.5$ 的起点。

$$
t_\star=0.6405035853385925,\qquad
\Sigma_\star^2=1.4976253607542171,\qquad
N_{\mathrm{reverse}}=1281.
$$

前向加噪使用两个方差 $\Sigma_\star^2/2$ 的热核增量，种子为 20261003。反向种子为 1，512 个构型同时演化到 $10^{-5}$，末步仅保留漂移。直接调用原有 `em_sample`，没有修改训练、采样或细化程序。保存的时间数组与此前 Haar 2000 步结果的后 1281 步完全一致。反向耗时 407.66 秒，准备、采样、观测与构型写出共 470.39 秒。

| 回路 | 解析值 | Haar 起点 | L64 起点 |
| --- | --- | --- | --- |
| $1\times1$ | 0.992509422 | $0.991287728\pm0.000002804$ | $0.991286047\pm0.000003150$ |
| $2\times2$ | 0.970372663 | $0.952677308\pm0.000031814$ | $0.952610616\pm0.000034530$ |
| $4\times4$ | 0.886654073 | $0.696364755\pm0.000329019$ | $0.695703931\pm0.000348506$ |
| $8\times8$ | 0.618040384 | $0.059727162\pm0.000822353$ | $0.058998348\pm0.000856162$ |
| $16\times16$ | 0.145904073 | $0.000150968\pm0.000493190$ | $0.000398908\pm0.000420633$ |

误差为一个标准误差。本次输出按四个父构型分块，Haar 输出使用逐构型误差，1024 个热浴参考按 16 个构型分块。共测量 20 种矩形回路，与 Haar 结果的逐项差异均小于 1.5 个合并标准误差。当前模型在这个噪声起点下没有观察到改善，大回路仍明显偏低。$32\times32$ 的目标均值只有 0.000453，当前误差较大，不能仅凭这一项与目标相容就认为生成正确。

对应位置的粗格 plaquette 与细格 $2\times2$ 回路相关系数，在局部更新后为 $0.798524\pm0.000353$，加噪后为 $0.000914\pm0.000718$，反向后为 $-0.000339\pm0.000640$。这一观测量未显示粗格空间信息的保留。逐构型空间平均的最终相关系数为 $-0.09157\pm0.04434$，约两个标准误差的弱反相关需要独立重复才能确认。

前向 20 种回路的热核衰减检查最大残差为 2.41 个标准误差。全部保存构型为有限值，最大四元数模平方误差为 $3.09\times10^{-7}$，矩阵结构和重读 plaquette 检查通过。三份原程序的 SHA256 与运行前一致。

结果见 [512 个生成构型及各阶段观测量](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_1p5_n512_steps1281_betanoise.h5)、[三页比较图](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_1p5_n512_steps1281_betanoise_comparison.pdf)、[回路统计表](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_1p5_n512_steps1281_betanoise_comparison.csv)、[父构型相关性](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_1p5_n512_steps1281_betanoise_retention.csv)、[采样日志](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_1p5_n512_steps1281_betanoise.log) 和 [分析及检查记录](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_1p5_n512_steps1281_betanoise_analysis.log)。

**继续降低反向起点噪声。**

随后测试 $\Sigma^2\simeq0.8$ 和 $0.2$。两组沿用上述 512 个父构型、EMA 模型、准备与采样种子，重新构造相同的 Relaxed 初态。对应 $2\times2$ 回路与上一组的最大差异为 $7.77\times10^{-16}$。两组各使用一个热核增量，继续保留原时间步长，末步只保留漂移，全部 512 个构型同时演化。

| 目标 $\Sigma^2$ | 实际 $\Sigma^2$ | $t_\star$ | 反向步数 | 反向耗时 |
| --- | --- | --- | --- | --- |
| 0.8 | 0.800468183291 | 0.570504307747 | 1141 | 362.96 秒 |
| 0.2 | 0.200218905171 | 0.417005836964 | 834 | 265.32 秒 |

| 回路 | 解析值 | $\Sigma^2\simeq0.8$ | $\Sigma^2\simeq0.2$ |
| --- | --- | --- | --- |
| $1\times1$ | 0.992509422 | $0.991397683\pm0.000002996$ | $0.992152296\pm0.000002163$ |
| $2\times2$ | 0.970372663 | $0.954244491\pm0.000028448$ | $0.965180062\pm0.000016372$ |
| $4\times4$ | 0.886654073 | $0.712432667\pm0.000294021$ | $0.833002232\pm0.000138902$ |
| $8\times8$ | 0.618040384 | $0.080023527\pm0.000829000$ | $0.438739444\pm0.000634930$ |
| $16\times16$ | 0.145904073 | $0.000330176\pm0.000578299$ | $0.058011572\pm0.000915231$ |

降低到 0.8 后，中小回路有统计显著的改善，但大回路仍严重偏低。降低到 0.2 后改善更大，$4\times4$、$8\times8$、$16\times16$ 的相对偏差分别为 $-6.05\%$、$-29.01\%$、$-60.24\%$，仍未恢复目标分布。相邻两组的差值按相同父构型配对，并按四个父构型分块估计误差。

Relaxed 初态的 $8\times8$ 均值为 0.639587，$16\times16$ 为 0.157210。因此即使起点降到 0.2，当前加噪和反向过程仍把大回路压到了目标值以下。较小起点减少了这一偏差，但这次结果不能说明反向过程已正确恢复平衡系综。

最终对应位置的粗格 plaquette 与细格 $2\times2$ 回路相关系数，在 0.8 组为 $-0.000612\pm0.000680$，在 0.2 组为 $0.012167\pm0.000660$。0.2 组保留了小幅、可检出的空间相关性。其逐构型空间平均相关系数为 $-0.01494\pm0.04601$，与零相容。偶偶锚点 $2\times2$ 回路与全格平均之差，从 Relaxed 初态的 $-0.005594\pm0.000008$ 降到 0.2 组的 $-0.0000213\pm0.0000117$，这项粗格周期结构明显减弱。

两组热核衰减检查的最大残差分别为 1.40 和 1.82 个标准误差，保存构型的最大四元数模平方误差分别为 $3.14\times10^{-7}$ 和 $3.17\times10^{-7}$。矩阵结构、重读 plaquette、父编号和时间网格检查均通过，原训练、采样和细化程序保持不变。

0.8 组结果见 [生成构型](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p8_n512_steps1141_betanoise.h5)、[比较图](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p8_n512_steps1141_betanoise_comparison.pdf) 和 [分析记录](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p8_n512_steps1141_betanoise_analysis.log)。

0.2 组结果见 [生成构型](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p2_n512_steps0834_betanoise.h5)、[三个噪声起点的比较图](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p2_n512_steps0834_betanoise_comparison.pdf)、[统计表](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p2_n512_steps0834_betanoise_comparison.csv)、[父构型相关性](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p2_n512_steps0834_betanoise_retention.csv) 和 [分析记录](../../results/su2_prolongation_reverse/reverse_L64_to_L128_beta200_Sigma2_0p2_n512_steps0834_betanoise_analysis.log)。

**Σ² 降到 0.15、0.1、0.05、0.02。**

上面三组的驱动脚本已经删除。2026-10-03 按运行日志和 HDF5 属性重建为 [reverse_su2_L128.py](reverse_su2_L128.py)，每次运行一个 Σ²。重建时逐项核对了 Σ²≃0.2 的存档。父构型完全相同，uniform、interior、relaxed 的 20 个回路与存档相差不超过 3×10⁻¹⁶。前向噪声由 GPU 上的 `train_su2_dm.heat_kernel_step` 在 `torch.manual_seed(20261003)` 下抽取，重现的加噪回路相差 1×10⁻⁷，是 float32 精度。NumPy 的 `refine_su2.forward` 不是原来的做法，相差 0.025。起点取完整 2000 步网格上 Σ² 最接近目标的时刻，这一规则重现了 1281、1141、834 步。反向直接调用原有 `em_sample`，种子为 1。程序运行时断言父构型和 relaxed 回路与 Σ²≃0.2 的存档一致。

[reverse_L128_4node.pbs](reverse_L128_4node.pbs) 申请 regular-g 的 4 个节点，每个节点算一个 Σ²。作业 3477259 四个节点全部完成，每个节点用时 456 到 580 秒。

| 目标 Σ² | 实际 Σ² | t⋆ | 反向步数 | 反向耗时 |
| --- | --- | --- | --- | --- |
| 0.15 | 0.150008775592 | 0.385506153107 | 771 | 283.49 秒 |
| 0.1 | 0.100213617498 | 0.342006593943 | 684 | 257.83 秒 |
| 0.05 | 0.049911553318 | 0.269007325172 | 538 | 214.61 秒 |
| 0.02 | 0.019962728652 | 0.180508196354 | 361 | 157.25 秒 |

Σ²≃0.02 事先在 mg0831 上单独试跑过，与作业中 mg0759 的结果逐位相同，最终 plaquette 都是 0.992623940588。同一种子下的 GPU 采样跨节点可以复现。

| 回路 | 解析值 | 0.15 | 0.1 | 0.05 | 0.02 |
| --- | --- | --- | --- | --- | --- |
| $1\times1$ | 0.992509 | 0.992236 | 0.992331 | 0.992456 | 0.992624 |
| $4\times4$ | 0.886654 | 0.845527 | 0.859115 | 0.875024 | 0.886465 |
| $8\times8$ | 0.618040 | 0.485376 | 0.534215 | 0.586039 | 0.615983 |
| $12\times12$ | 0.338679 | 0.218456 | 0.262321 | 0.309988 | 0.337166 |
| $16\times16$ | 0.145904 | 0.077657 | 0.101553 | 0.129149 | 0.145417 |

误差按四个父构型分块，$1\times1$ 约 $2\times10^{-6}$，$8\times8$ 约 0.0005，$16\times16$ 约 0.001。大回路随 Σ² 减小单调接近解析值。Σ²≃0.02 时 $16\times16$ 与解析值相差 0.4 个标准误差，$12\times12$ 相差 1.5 个，$8\times8$ 偏低 $0.0021\pm0.0005$。plaquette 则由偏低变为偏高，0.05 时为 0.992456，低于解析值，0.02 时为 0.992624，高出约 54 个标准误差。relaxed 初态的 plaquette 为 0.994861，很小的噪声不足以消除细化留下的短程过度有序。所测范围内没有一个 Σ² 同时使 plaquette 和大回路与解析值一致。

七个粗格起点、Haar 和热浴的比较图由 [plot_su2_reverse_L128.py](plot_su2_reverse_L128.py) 生成，前缀为 `reverse_L64_to_L128_beta200_Sigma2_1p5_to_0p02_n512_betanoise`，包括 Wilson 回路、二阶矩、action density、面积律和有效弦张力五张图。R=1 到 32 全部方形回路的逐构型 $W$ 与 $W^2$ 平均存于同一前缀的 `_squares.h5`，已有系综不重算。热浴的 32 个方形回路与解析值 $r^{R^2}$ 的偏差最大为 1.3 个标准误差。

## 依据

现有文件接口来自 `train_su2_dm.py` 的群运算、时钟、DSM、LCNN、数据加载及 checkpoint 保存，以及 `sample_su2_em.py` 的 EMA 加载、EM 更新、Wilson loops 和矩阵输出。
均匀通量表格及少量随机修正是本原型待验证的设计，不是从这些训练文件中已有的功能。

[1] W. Detmold et al., *Path integral contour deformations for observables in SU(N) gauge theory*, arXiv:2101.12668, Sec. III.1, Eqs. (39)–(42)。采用 beta=4/g² 对应本文作用量，弦张力公式取无限体积极限。
[2] M. G. Endres et al., *Multiscale Monte Carlo equilibration: Pure Yang–Mills theory*, arXiv:1510.04675。仅作为细化后再热化的背景，本任务不要求实现文中的 RG 流程。
