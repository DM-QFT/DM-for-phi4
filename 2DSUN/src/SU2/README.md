# 二维 SU(2) 规范场的训练与生成程序

本文对应 2026 年 10 月 1 日的代码。训练程序用热浴构型学习加噪后概率分布的对数梯度，生成程序从 Haar 随机链出发，沿反向扩散逐步生成近似服从目标规范场分布的构型。

以下文件分别负责不同阶段。

- [su2_2d_heatbath.jl](../../trainningdata/su2_2d_heatbath.jl) 用热浴和过松弛产生训练构型。
- [train_su2_dm.py](train_su2_dm.py) 定义四元数运算、群上的加噪过程、规范等变网络和训练循环。
- [sample_su2_em.py](sample_su2_em.py) 读取训练参数，用反向 Euler–Maruyama 生成构型，计算可观测量并保存 HDF5。
- [compare_su2_wilson.py](compare_su2_wilson.py) 比较已有 DM、热浴构型与解析 Wilson 回路，输出带误差棒的图和数值表。

整体数据流是

```text
热浴构型 HDF5
    ↓ load_cfgs
单位四元数 U0
    ↓ loss_fn 中的群上随机游走
加噪构型 Ut 和最后一步的解析训练目标
    ↓ LCNN 与 Adam 更新
网络参数与 EMA 参数
    ↓ sample_su2_em.py 读取 EMA
Haar 随机链经过反向扩散
    ↓ report 与 to_matrix
可观测量与生成构型 HDF5
```

## 1. 物理问题与数据表示

格子是周期边界的二维 $L\times L$ 格子。每个格点有两个方向的链变量 $U_\mu(x)\in SU(2)$。目标分布相对于各条链的 Haar 测度定义为

$$
p_0(U)\propto e^{-S_W[U]},\qquad
S_W[U]=\beta\sum_x\left[1-\frac12\operatorname{Re}\operatorname{Tr}P(x)\right],
$$

$$
P(x)=U_0(x)U_1(x+\hat 0)U_0(x+\hat 1)^\dagger U_1(x)^\dagger.
$$

当前默认值是 $L=32$、$\beta=2.0$。网络输入只有构型和扩散时间。$\beta$ 通过训练数据决定，并不是网络的条件输入。训练损失也不直接使用 Wilson 作用量或 plaquette。

### 1.1 用四个实数存一条链

代码采用

$$
U=a_0\mathbb 1+i\sum_{a=1}^3a_a\sigma^a
=\begin{pmatrix}
a_0+ia_3 & a_2+ia_1\\
-a_2+ia_1 & a_0-ia_3
\end{pmatrix},\qquad
a_0^2+a_1^2+a_2^2+a_3^2=1.
$$

$\sigma^a$ 是 Pauli 矩阵。程序存储实四元数 $(a_0,a_1,a_2,a_3)$，因此群乘法和指数映射都可以直接计算，不需要复矩阵乘法、矩阵对角化或 `matrix_exp`。

`qmul(p, q)` 实现

$$
(p_0,\boldsymbol p)(q_0,\boldsymbol q)
=\left(p_0q_0-\boldsymbol p\cdot\boldsymbol q,
p_0\boldsymbol q+q_0\boldsymbol p-\boldsymbol p\times\boldsymbol q\right).
$$

叉积前的负号来自这里的 $U=a_0\mathbb 1+i\boldsymbol a\cdot\boldsymbol\sigma$ 约定。`conj` 把后三个分量变号。对单位四元数，它就是 $U^\dagger$。

李代数使用三个实坐标 $\boldsymbol\xi$，生成元取 $T^a=\sigma^a/2$。`qexp(xi)` 计算

$$
\exp\left(i\boldsymbol\xi\cdot\boldsymbol\sigma/2\right)
=\left(\cos\frac\theta2,
\frac{\sin(\theta/2)}{\theta/2}\frac{\boldsymbol\xi}{2}\right),
\qquad \theta=|\boldsymbol\xi|.
$$

### 1.2 张量布局

- 磁盘中的 `cfgs` 是复矩阵，h5py 读到的轴顺序是 `(n, y, x, mu, a, b)`，最后两个轴都是长度 2。
- `load_cfgs` 提取四元数并交换轴，得到 `float32` 张量 `(n, 2, L, L, 4)`。两个格点轴依次是 `x`、`y`。
- 网络输出是 `(B, 2, L, L, 3)`。每条链输出一个李代数三维向量。
- 中间特征是 `(B, ch, L, L, 4)`。这些特征是四元数的实线性组合，不要求长度为 1。
- `to_matrix` 把生成结果转回输入 HDF5 的轴顺序，并以 `complex128` 保存。计算本身仍然使用 `float32`。

`roll(A, mu, k)` 表示在 $x+k\hat\mu$ 处取值，通过周期移位实现。`bc(s)` 将每个构型的标量扩展为 `(B, 1, 1, 1, 1)`，用于广播到全部链及其分量。

## 2. 前向加噪过程

### 2.1 扩散时间和噪声强度

每条链独立地做群上的布朗运动。代码使用的扩散强度及累计扩散方差是

$$
g(t)=\sigma^t,\qquad
\Sigma(t)=\int_0^t g(u)^2\,\mathrm du
=\frac{\sigma^{2t}-1}{2\ln\sigma},\qquad
s(t)=\sqrt{\Sigma(t)}.
$$

它们分别对应 `diffusion_coeff` 和 `marginal_prob_std`。这里的 $s(t)$ 表示群上扩散的尺度，不是四元数四个分量的标准差。默认 `SIGMA = 14.0`，给出

$$
\Sigma(1)\simeq36.9450,\qquad s(1)\simeq6.07824.
$$

在这个生成元约定下，基本表示的布朗运动衰减因子是

$$
e^{-C_F\Sigma/2}=e^{-3\Sigma/8},\qquad C_F=3/4.
$$

到 $t=1$ 时该因子约为 $9.62\times10^{-7}$。生成程序因此用各条链独立的 Haar 分布近似前向过程的终点分布。有限扩散时间下，这仍然是一个近似。

### 2.2 `heat_kernel_step` 如何抽一个增量

一次增量通过左乘作用在链上

$$
U' = \exp\left(i\boldsymbol Y\cdot\boldsymbol\sigma/2\right)U.
$$

`heat_kernel_step(shape, delta)` 返回的是三维坐标 $\boldsymbol Y$，并非已经指数化的群元。参数 $\delta$ 是该步的扩散方差。算法先抽高斯候选

$$
\boldsymbol Y\sim\mathcal N(0,\delta\mathbb 1_3),
$$

再以

$$
\frac{\sin(\theta/2)}{\theta/2},\qquad \theta=|\boldsymbol Y|
$$

作为主指数坐标区域内的接受概率。原因是指数映射的体积因子为

$$
j(\boldsymbol Y)=\left[\frac{\sin(\theta/2)}{\theta/2}\right]^2.
$$

在忽略缠绕像的近似下，热核相对于 Haar 测度的密度是

$$
K_\delta\left(e^{i\boldsymbol Y\cdot\boldsymbol\sigma/2}\right)
\propto\frac{e^{-|\boldsymbol Y|^2/(2\delta)}}{\sqrt{j(\boldsymbol Y)}}.
$$

乘回体积元 $j(\boldsymbol Y)\,\mathrm d^3Y$ 后，坐标密度正好是高斯乘以 $\sqrt j$，所以需要接受拒绝这一步。接受后的坐标分布已经不再是高斯。

`_heat_kernel_draw` 用 Triton 在 GPU 上执行拒绝循环。一个 block 处理 128 个候选，未接受的候选继续重抽。循环不需要 CPU 同步。$\delta=0$ 时直接返回零，这用于训练中的补齐步骤和采样的最后一步。

当前实现没有求和完整热核的缠绕像。训练将每步方差限制为 `DELTA_MAX = 1.0`，使主要概率区域远离指数坐标的边界。源码中的 $e^{-79/\delta}$ 来自单位元附近距离 $4\pi$ 的最近像，不能把它解释成整个群上一致的相对误差界。本文中的热核公式均指这套小步近似。

## 3. `loss_fn` 的训练目标

### 3.1 从干净构型得到加噪构型

每个 batch 中，每个构型独立抽一个时间

$$
t=\epsilon+(1-\epsilon)\left(1-\sqrt{1-u}\right),
\qquad u\sim\operatorname{Uniform}(0,1),\qquad \epsilon=10^{-5}.
$$

忽略很小的下端截断，其密度是 $2(1-t)$，所以小时间被抽到的次数更多。接近 Haar 分布时 score 很小，这样的取样把更多训练放在尚有物理结构的区间。

给定 $t$，程序设置

$$
K=\max\left(1,\left\lceil\frac{s(t)^2}{\mathrm{DELTA\_MAX}}\right\rceil\right),
\qquad \delta=\frac{s(t)^2}{K}.
$$

然后在每条链上抽 $K$ 个独立增量，并计算

$$
U_t=G_KG_{K-1}\cdots G_1U_0,\qquad
G_k=\exp\left(i\boldsymbol Y_k\cdot\boldsymbol\sigma/2\right).
$$

同一构型的全部链共享 $t$、$K$ 和 $\delta$，但噪声向量各自独立。

默认参数下最多需要 37 个有效增量。为了保持编译图的形状固定，代码将子步轴补齐到 `Kmax = 64`，无效位置取 $\delta=0$。所有增量一次抽出，再按两两相乘的方式用 6 轮归约得到有序乘积。群乘法不交换，因此配对顺序保留为 $G_K\cdots G_1$。

### 3.2 为什么监督只用最后一个增量

网络需要学习的是加噪边缘分布的 score

$$
\boldsymbol S_\vartheta(U_t,t)\approx\nabla\log p_t(U_t).
$$

这里每个分量的导数沿单条链的左乘方向定义，即将该链替换为 $e^{i\varepsilon\sigma^a/2}U_\mu(x)$，再对 $\varepsilon$ 求导。

对马尔可夫过程的最后一步，有

$$
\nabla\log p_t(U_t)
=\mathbb E\left[\nabla_{U_t}\log q(U_t\mid U_{K-1})\mid U_t\right].
$$

因此，用最后一步过渡概率的 score 做最小二乘回归，其条件均值就是所需的边缘 score。前面的步骤负责把 $U_t$ 抽到正确分布，监督目标只需最后一步。网络并不接收 $U_0$、$U_{K-1}$ 或 $\boldsymbol Y_K$。

`heat_kernel_score` 计算的目标是

$$
\boldsymbol H_\delta(\boldsymbol Y)
=-\boldsymbol Y\left[
\frac1\delta+
\frac{\frac12\cot(\theta/2)-1/\theta}{\theta}
\right],\qquad \theta=|\boldsymbol Y|.
$$

第一项来自高斯指数，第二项来自群的体积因子。`dlog_sinc_half_over` 在小 $\theta$ 时使用

$$
\frac{\frac12\cot(\theta/2)-1/\theta}{\theta}
=-\frac1{12}-\frac{\theta^2}{720}+O(\theta^4)
$$

避免相近数相减造成精度损失。`Y_last` 选取每个构型的第 $K$ 个有效增量。

### 3.3 损失如何归一化

程序先构造残差

$$
\boldsymbol r=s(t)\left[
\boldsymbol S_\vartheta(U_t,t)-\boldsymbol H_\delta(\boldsymbol Y_K)
\right].
$$

它对三个李代数分量求平方和，对每个构型的 $2L^2$ 条链取平均，再除以 $K$，最后对 batch 取平均。因 $s(t)^2/K=\delta$，可写成

$$
\mathcal L=\mathbb E\left[
\frac{\delta}{2L^2}\sum_{x,\mu}
\left|\boldsymbol S_\vartheta(U_t,t)-\boldsymbol H_\delta(\boldsymbol Y_K)\right|^2
\right].
$$

$1/K$ 改变各个时间的相对权重，不改变固定时间下理想回归解的条件均值。代码对三个分量求和，并没有再除以 3。

源码注释用 3 描述零输出网络的损失尺度。严格等于 3 是小步平直高斯极限的说法，有限步长下存在群曲率修正。训练目标本身也含有条件涨落，所以最优损失一般不为零。实际生成质量需要用可观测量判断。

## 4. `LCNN` 怎样保证规范等变

规范变换为

$$
U_\mu(x)\longmapsto\Omega(x)U_\mu(x)\Omega(x+\hat\mu)^\dagger.
$$

因为噪声从左侧乘到链上，score 应当在链的起点变换

$$
\boldsymbol S_\mu(x)\cdot\boldsymbol\sigma
\longmapsto\Omega(x)
\left[\boldsymbol S_\mu(x)\cdot\boldsymbol\sigma\right]
\Omega(x)^\dagger.
$$

网络先构造以格点为基点的闭合回路，再用保持这个变换规律的运算处理它们。

1. `plaquette(U)` 给出每个格点的有向 plaquette，作为初始的一个四元数通道。
2. `transport(W, U)` 将本地及四个邻居的回路都搬到当前格点，形成 5 倍通道。
3. 每层的 `conv` 对这些通道做实线性组合。标量部分和向量部分使用两套权重，向量的三个分量共享同一套权重。
4. 拼入单位四元数通道后，`qbilinear` 计算可学习的双线性组合 $\sum_{ij}b_{kij}W_iW_j$。
5. 用回路标量部分和时间决定一个 sigmoid 因子，再将它乘到整个四元数上。
6. 最后一层的标量部分与时间特征产生系数，对向量部分加权求和，分别输出两个链方向的 score。

其中，正方向和负方向的平行移动分别是

$$
U_\mu(x)W(x+\hat\mu)U_\mu(x)^\dagger,
$$

$$
U_\mu(x-\hat\mu)^\dagger W(x-\hat\mu)U_\mu(x-\hat\mu).
$$

它们都按 $\Omega(x)W\Omega(x)^\dagger$ 变换。四元数乘法和实线性组合保持这个规律，标量部分则是规范不变量。

时间特征由 `temb` 将 $\log s(t)$ 映射为 64 维向量 $h(t)$。每层的调制因子为

$$
\operatorname{sigmoid}\left(a_cW_{c,0}(x)+b_c(h(t))\right).
$$

sigmoid 有界，用来抑制连续双线性乘法带来的幅度增长。最终 `readout` 产生规范不变的系数 $c_{\mu c}(x,t)$，输出

$$
\boldsymbol S_{\vartheta,\mu}(x,t)
=\frac1{s(t)}\sum_c c_{\mu c}(x,t)\boldsymbol W_c(x).
$$

所以 `model(U, t)` 返回的已经是除过 $s(t)$ 的 score。训练残差中的乘 $s(t)$ 用于调整损失尺度。

默认 16 个通道、3 层，共 33872 个可训练参数。`check_equivariance` 在训练开始时，对两个构型施加随机局域规范变换，并比较网络输出与预期旋转后的输出。这个检查只打印误差，不设置通过阈值。程序没有做规范固定。

## 5. 训练主循环与参数保存

`train_su2_dm.py` 的 `main` 执行以下步骤。

1. 设置随机种子并创建 `OUT` 目录。`load_cfgs()` 一次读入全部构型，并放到默认的 CUDA 设备上。
2. 创建 `LCNN`、Adam 优化器和参数的指数移动平均 EMA。优化器使用 `fused=True`。
3. 用 `torch.compile` 编译 `loss_fn`，采用 `reduce-overhead`、完整图和固定形状。
4. 每个 epoch 随机打乱构型顺序，按 batch 计算损失、反向传播并更新参数。
5. 每次参数更新后更新 EMA。第 $k$ 次更新使用衰减率 $\min(0.999,(1+k)/(10+k))$，保留早期的逐渐增大过程。
6. 打印每个 epoch 的平均损失和累计时间。每 10 个 epoch 及最后一个 epoch 保存参数文件。

最后不足一个 batch 的构型被跳过。当前数据有 10000 个构型，`BATCH = 64` 时每个 epoch 更新 156 次，使用 9984 个构型。每轮重新打乱，所以被跳过的 16 个不是固定的。

保存文件的命名是 `ckpt_epoch0010.pt`、`ckpt_epoch0020.pt`，依此类推。文件包含当前网络参数 `model`、平滑参数 `ema`，以及 `ch`、`nlayers`、`sigma`、`L`、`beta`、`epoch`。

文件不含优化器状态，训练入口也没有继续训练的分支。现有参数文件主要供生成程序读取。训练代码输出日志到终端，目录中的 `train.log` 来自运行时的重定向或外部运行脚本。

## 6. 生成程序如何反向积分

### 6.1 恢复 EMA 网络并建立初态

`sample_su2_em.py` 从参数文件读取网络宽度、层数和 $\sigma$，随后将保存的 EMA 参数复制给模型。实际采样使用 EMA 权重。

`haar_su2` 抽四维独立标准高斯，再除以四维长度，得到 $S^3$ 上均匀分布的单位四元数。每个格点、方向和构型独立抽样。生成不需要从某个训练构型开始。

### 6.2 `em_step` 的更新式

设正步长 $h=t_i-t_{i+1}>0$。代码先计算三维向量

$$
\boldsymbol X=g(t_i)^2\boldsymbol S_\vartheta(U_i,t_i)h+\boldsymbol Z_i,
$$

其中 $\boldsymbol Z_i$ 由 `heat_kernel_step` 抽出，扩散方差是 $g(t_i)^2h$。随后更新

$$
U_{i+1}=\exp\left(i\boldsymbol X\cdot\boldsymbol\sigma/2\right)U_i.
$$

漂移项前是正号，因为这里的 $h$ 表示向较小时间移动的正长度。漂移与随机增量先在李代数坐标中相加，再做一次指数映射。这是反向 SDE 的离散近似，仍有有限步长误差。

默认从 $t=1$ 等间隔走到 $10^{-3}$，共 2000 步。因此

$$
h=0.0004995,\qquad
\max_i g(t_i)^2h=0.097902.
$$

采样时每步直接调用一次 `heat_kernel_step`，没有训练中的子步拆分。若大幅减小 `N_STEPS` 或增大 $\sigma$，需要重新检查单步方差。

最后一步将 `noise` 设为零，保留 score 漂移。它是程序主动采用的末步去噪处理。因此最终结果停在 $t=10^{-3}$，最后一小步也不是完整的随机更新。

### 6.3 批处理与单位化

默认一次生成 250 个构型，共生成 1000 个。每个 batch 从独立 Haar 初态开始，并使用新噪声，不沿用前一批的终态。

`em_step` 被 `torch.compile` 编译。循环外保留 `U` 的存储，用 `U.copy_(...)` 写回一步结果，避免 CUDA graph 后续重放覆盖已经保留的样本。

全部采样完成后，程序先记录

$$
\max_{n,\mu,x}\left|\sqrt{\sum_{a=0}^3 a_a^2}-1\right|,
$$

再将每条四元数除以长度，消除连乘积累的浮点漂移。日志中的 `unitarity drift before projection` 是这个四元数长度偏差，并不是直接计算的矩阵范数。

生成流程没有 Metropolis 接受拒绝校正。这里 `heat_kernel_step` 内部的拒绝抽样只用于产生噪声。因此网络误差、时间离散、终止时间和末步处理都可能影响最终物理分布。

## 7. 可观测量与输出文件

`line(U, mu, n)` 沿一个方向连乘 $n$ 条链。`wilson(U, R, T)` 将四条边闭合成矩形回路，并取四元数的标量部分，即

$$
W(R,T,x)=\frac12\operatorname{Re}\operatorname{Tr}\prod_{\ell\in\partial(R\times T)}U_\ell.
$$

`report` 计算平均 plaquette、$W(1,2)$ 和 $W(2,2)$。先在每个构型内对格点取平均，再在构型之间求均值及标准误差

$$
\operatorname{SE}=\frac{\operatorname{std}(O_1,\ldots,O_N)}{\sqrt N}.
$$

`site std` 单独表示全部局域 plaquette 值的标准差，不是平均 plaquette 的误差。对热浴数据，当前 `report` 没有修正构型间的自相关。

无限体积二维 SU(2) 的对照值是

$$
p=\frac{I_2(\beta)}{I_1(\beta)},\qquad
\langle W(R,T)\rangle=p^{RT}.
$$

代码用 `scipy.special.ive` 的比值计算 $p$，两个 Bessel 函数的共同指数因子相消。在 $\beta=2$ 时

$$
p\simeq0.43312743,\qquad
p^2\simeq0.18759937,\qquad
p^4\simeq0.03519352.
$$

日志中的 `exact` 指无限体积结果，不是任意有限周期格子的严格精确值。`training` 对照默认只读训练文件的前 1000 个构型，不是全部 10000 个。

生成文件保存在参数文件所在目录，命名为 `samples_epoch0100_steps2000.h5`。它包含

- `cfgs`。生成的复矩阵构型，轴顺序与热浴文件相同。
- `plaq`。每个生成构型的平均 plaquette。
- 文件属性。`beta`、`L`、`sigma`、`epoch`、`n_steps`、`plaq_exact`。

两个较大的 Wilson 回路只打印到终端，没有存成数据集。生成种子和终止时间 `EPS` 也没有保存在 HDF5 属性中。程序在输出前还会读取对应 $L$、$\beta$ 的热浴文件来做对照，因此直接运行整个采样脚本仍需要该文件存在。

## 8. 默认参数与运行方式

训练参数集中在 `train_su2_dm.py` 顶部。

- 物理参数为 `L = 32`、`BETA = 2.0`。
- 扩散参数为 `SIGMA = 14.0`、`DELTA_MAX = 1.0`、`EPS = 1e-5`。
- 网络参数为 `CH = 16`、`NLAYERS = 3`。
- 优化参数为 `BATCH = 64`、`EPOCHS = 100`、`LR = 2e-4`、`EMA_DECAY = 0.999`、`SEED = 0`。
- `DATA` 默认指向 `trainningdata/cfgs_su2_beta=2.0_32^2.h5`。
- `OUT` 默认指向 `results/su2_L32_beta2.0_lcnn_hk_c16l3`。

生成参数集中在 `sample_su2_em.py` 顶部，为 `N_CFGS = 1000`、`BATCH = 250`、`N_STEPS = 2000`、`EPS = 1e-3`、`SEED = 1`。训练与生成中的同名常量各自独立。

两个脚本都依赖 CUDA。训练模块导入时就执行 `torch.set_default_device("cuda")`，同时使用 Triton、编译图和融合 Adam。当前检查环境的 Python 是 `/work/gw97/w97000/envs/nenv2/bin/python`，安装有 PyTorch、Triton、torch-ema、NumPy、h5py 和 SciPy。

从项目根目录运行。

```bash
source /work/gw97/w97000/envs/nenv2/bin/activate
cd /work/gw97/w97000/project/DM/2DSUN
python src/SU2/train_su2_dm.py
python src/SU2/sample_su2_em.py
```

训练没有命令行参数，改变实验时直接修改顶部常量。采样唯一读取的命令行参数是第一个位置参数，用来指定参数文件。

```bash
python src/SU2/sample_su2_em.py results/su2_L16_beta2.0_lcnn_hk_c16l3_20261001/ckpt_epoch0100.pt
```

不传参数时，采样读取训练模块当前 `OUT` 和 `EPOCHS` 拼出的路径，不会搜索最近一次实验。当前默认目录没有 `_20261001` 后缀。

`DATA` 和 `OUT` 在源码中是绝对路径。开展新实验时应给 `OUT` 设置新的目录。同名参数文件会被覆盖，采样 HDF5 也以覆盖模式写入，文件名没有包含随机种子。

## 9. 当前已有的一次运行记录

本节根据已有的 [训练日志](../../results/su2_L16_beta2.0_lcnn_hk_c16l3_20261001/train.log)、[采样日志](../../results/su2_L16_beta2.0_lcnn_hk_c16l3_20261001/sample.log) 和 HDF5 内容整理。这些是已有实验结果，编写本文时没有重新训练或进行完整生成。

`su2_L16_beta2.0_lcnn_hk_c16l3_20261001` 使用 10000 个训练构型、33872 个参数，训练 100 个 epoch。日志记录的最后损失为 2.5450，启动时规范等变相对偏差为 $1.9\times10^{-6}$。

默认设置生成了 1000 个构型，每个构型经过 2000 步反向更新。已有记录给出

- 平均 plaquette。热浴前 1000 个构型为 $0.43379\pm0.00080$，生成为 $0.43072\pm0.00078$，无限体积值为 $0.43312743$。
- $W(1,2)$。热浴为 $0.18881\pm0.00103$，生成为 $0.18561\pm0.00109$，无限体积值为 $0.18759937$。
- $W(2,2)$。热浴为 $0.03489\pm0.00106$，生成为 $0.03370\pm0.00109$，无限体积值为 $0.03519352$。

这次生成的 plaquette 均值比无限体积值低约 0.56%，相差约 3.1 个所报标准误差。已有结果说明仍存在需要检验的偏差，单次运行不能区分网络误差、积分误差与统计涨落各占多少。训练 loss 接近稳定也不能代替物理检验。

日志记录生成结束时、单位化之前的最大四元数长度偏差为 $6.6\times10^{-6}$。完整训练数据文件中的 10000 个 plaquette 平均值为 $0.43319369$，其未作自相关修正的标准误差约为 $0.00025188$。

编写本文时另外用前 4 个训练构型核对了读写转换与 plaquette。矩阵转换往返的最大绝对误差约为 $4.2\times10^{-8}$，重算 plaquette 与文件中保存值的最大差约为 $1.2\times10^{-8}$。复用现有规范等变检查，在 CPU 上对随机初始化网络得到约 $1.5\times10^{-6}$ 的相对偏差。这些检查用于核对数据约定与说明中的公式，不构成生成精度的额外验证。

## 10. 对照源码的阅读顺序

先读训练文件的 `load_cfgs` 和 `main`，确定数据怎样进入训练。随后读 `loss_fn`，把 $t$、$K$、$\delta$、`Ut` 和 `Y_last` 对应起来。再读 `heat_kernel_step`、`heat_kernel_score` 和 `LCNN.forward`，分别理解噪声、监督目标和网络输出。

最后读采样文件的 `main`、`em_sample`、`em_step`。前三节确定了它反演的前向过程，第六节给出了更新式。`report`、`wilson` 和 `to_matrix` 则负责检查物理量并恢复磁盘布局。

## 11. Wilson 回路比较

从项目根目录运行。

```bash
/work/gw97/w97000/envs/nenv2/bin/python src/SU2/compare_su2_wilson.py
```

输入路径、回路列表 `RT`、测量批量 `BATCH` 和热浴块长 `HB_BLOCK` 放在脚本顶部。默认读取第九节记录的 `_20261001` 目录中的 1000 个 DM 构型，以及热浴文件中的全部 10000 个构型。这里的热浴文件也用于训练，没有将它视为独立保留的验证集。

默认分别测量全部 $1\le R,T\le4$ 的 16 种矩形回路，不合并交换方向的回路。程序复用现有的 `load_cfgs` 和 `wilson`，以双精度在 CUDA 上分批测量。`load_cfgs` 仍一次读入整个构型文件，新增的 `dtype` 参数只改变测量精度，训练和采样继续默认使用单精度。

每个构型先对全部起点取空间平均。DM 误差为这些构型均值的样本标准差除以 $\sqrt{N}$。热浴按保存顺序将每 50 个构型组成一个块，用块均值的样本标准差除以块数的平方根。若末尾不足一个完整块，则均值和误差都不使用该块。改变热浴数据时，应检查块长增大后误差是否稳定，确保块长足够覆盖自相关。

解析对照为无限体积的 $[I_2(\beta)/I_1(\beta)]^{RT}$。当前 $\beta=2$、$L=16$ 和这些小回路的有限体积修正可忽略。这个近似不能直接用于小体积或接近整个格子大小的回路。

输出位于 DM 构型所在目录，沿用输入文件名并添加 `_wilson` 后缀。PDF 和 PNG 上图显示均值及一个标准误差，下图显示相对于解析值的差及同样的误差棒。上图使用对称对数坐标，零附近保持线性，保留大回路的负估计值和跨零误差棒。CSV 保存各回路的 $R,T,RT$、解析值和两个系综的均值与标准误差。

同一回路的 DM、热浴和解析值使用相同横坐标。图宽采用 7 英寸通栏尺寸，字体、配色、点型、线宽、误差棒端帽及保存分辨率使用项目的 `../plots/cjp.mplstyles`。
