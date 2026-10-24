# 二维 SU(2) 规范场的训练与生成程序

本文对应 2026 年 10 月 23 日的代码。训练程序用热浴构型学习加噪后概率分布的对数梯度，生成程序从 Haar 随机链出发，沿反向扩散逐步生成近似服从目标规范场分布的构型。

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

当前默认值是 $L=128$、$\beta=200.0$。网络输入只有构型和扩散时间。$\beta$ 决定训练数据和本次噪声方案，并不是网络的独立条件输入。训练损失也不直接使用 Wilson 作用量或 plaquette。

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

每条链独立地做群上的布朗运动。用 $t$ 表示扩散时间，用 $\Sigma(t)$ 表示噪声尺度。其平方是热核的方差参数

$$
\Sigma(t)^2=\int_0^t g(u)^2\,\mathrm du.
$$

`marginal_prob_std` 返回 $\Sigma(t)$，`diffusion_coeff` 返回 $g(t)$。代码直接使用积分的解析结果，不执行数值积分。这里的 $\Sigma(t)$ 不是四元数分量或总旋转角的标准差。

前向扫描的横轴统一用累计方差 $\Sigma^2$。同一 $\Sigma^2$ 对应同一热核噪声强度，噪声日程改变时，对应的网络时间 $t$ 可以不同。细化实验历史文件中的 `tau` 字段也表示这个累计方差。

当前训练采用依赖 $\beta$ 的方案

$$
A=\log(1+\beta\Sigma_{\max}^2),\qquad
\Sigma(t)^2=\frac{e^{At}-1}{\beta},\qquad
g(t)^2=\frac{A}{\beta}e^{At}.
$$

最大噪声保持与原方案相同。`SIGMA = 14.0` 用于确定这个端点

$$
\Sigma_{\max}^2=\frac{14^2-1}{2\log14}\simeq36.9450,\qquad
\Sigma_{\max}\simeq6.07824.
$$

在 $\beta=200$ 时，$A\simeq8.90788$，$\Sigma(t)^2=1/\beta$ 对应 $t\simeq0.07781$。原方案的同一噪声尺度对应 $t\simeq0.004935$。保持原有的时间抽样方式和均匀生成时间网格，会增加弱噪声区间的训练样本和生成步数。

参数文件用 `noise_beta` 保存这一方案使用的 $\beta$。缺少此字段的旧参数文件仍使用原方案

$$
g(t)=\sigma^t,\qquad
\Sigma(t)^2=\frac{\sigma^{2t}-1}{2\log\sigma}.
$$

在这个生成元约定下，基本表示的布朗运动衰减因子是

$$
e^{-C_F\Sigma^2/2}=e^{-3\Sigma^2/8},\qquad C_F=3/4.
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

`_heat_kernel_draw` 用 Triton 在 GPU 上执行拒绝循环。一个 block 处理 128 个候选，未接受的候选继续重抽。循环不需要 CPU 同步。$\delta=0$ 时直接返回零，这用于采样的最后一步。训练中的加噪由 `_brownian` 完成，每个增量的抽取方式与此相同，见 3.1 节。

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
K=\max\left(1,\left\lceil\frac{\Sigma(t)^2}{\mathrm{DELTA\_MAX}}\right\rceil\right),
\qquad \delta=\frac{\Sigma(t)^2}{K}.
$$

然后在每条链上抽 $K$ 个独立增量，并计算

$$
U_t=G_KG_{K-1}\cdots G_1U_0,\qquad
G_k=\exp\left(i\boldsymbol Y_k\cdot\boldsymbol\sigma/2\right).
$$

同一构型的全部链共享 $t$、$K$ 和 $\delta$，但噪声向量各自独立。

默认参数下最多需要 37 个有效增量，当前噪声方案下平均约 1.74 个。`brownian` 调用 Triton kernel `_brownian`，每条链依次抽取本构型的 $K$ 个增量，在寄存器中从左侧连乘，得到 $G_K\cdots G_1U_0$。每个增量的抽取与 `_heat_kernel_draw` 相同，Philox 计数按链、子步和重试次数分开。kernel 同时返回 $U_t$ 和最后一个增量 $\boldsymbol Y_K$。一个 block 的 128 条链总在同一构型内，因此 $K$ 在 block 内一致，循环次数由数据决定也不需要 CPU 同步。

2026 年 10 月 23 日以前的代码把子步轴补齐到 64，一次抽出全部增量，再两两相乘归约。那时约 97% 的抽样和乘法落在 $\delta=0$ 的补齐步上。

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
\boldsymbol r=\Sigma(t)\left[
\boldsymbol S_\vartheta(U_t,t)-\boldsymbol H_\delta(\boldsymbol Y_K)
\right].
$$

它对三个李代数分量求平方和，对每个构型的 $2L^2$ 条链取平均，再除以 $K$，最后对 batch 取平均。因 $\Sigma(t)^2/K=\delta$，可写成

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

时间特征由 `temb` 将 $\log\Sigma(t)$ 映射为 64 维向量 $h(t)$。每层的调制因子为

$$
\operatorname{sigmoid}\left(a_cW_{c,0}(x)+b_c(h(t))\right).
$$

sigmoid 有界，用来抑制连续双线性乘法带来的幅度增长。最终 `readout` 产生规范不变的系数 $c_{\mu c}(x,t)$，输出

$$
\boldsymbol S_{\vartheta,\mu}(x,t)
=\frac1{\Sigma(t)}\sum_c c_{\mu c}(x,t)\boldsymbol W_c(x).
$$

所以 `model(U, t)` 返回的已经是除过 $\Sigma(t)$ 的 score。训练残差中的乘 $\Sigma(t)$ 用于调整损失尺度。

默认 32 个通道、3 层，共 146336 个可训练参数。此前 16 通道、3 层模型有 33872 个参数。`check_equivariance` 在训练开始时，对两个构型施加随机局域规范变换，并比较网络输出与预期旋转后的输出。这个检查只打印误差，不设置通过阈值。程序没有做规范固定。

以上公式由三组 Triton kernel 计算，数学定义与参数都不变。

- `qbilinear` 把恒等通道单独拿出，写成 $w_{k00}+\sum_i(w_{k0i}+w_{ki0})W_i+\sum_{i,j\ge1}w_{kij}W_iW_j$，其余通道数为 32。`_qbil_fwd` 对每块格点逐个 $j$ 计算 $A_{kj}=\sum_i w_{kij}W_i$，随即与 $W_j$ 做四元数乘法并累加，$A$ 只在寄存器中。按矩阵乘积写法物化 $A$ 时，$B=64$、$L=128$ 每层约 17.7 GB。`_qbil_bwd_x` 和 `_qbil_bwd_w` 给出对输入和对权重的梯度，所用恒等式为 $\langle p,ab\rangle=\langle p\bar b,a\rangle=\langle\bar a p,b\rangle$。
- 输入通道等于 `ch` 的层用 `TConv`，在寄存器中完成 `transport` 与通道求和，不存 $5\,\mathrm{ch}$ 个平移后的回路。$\mathrm{Ad}$ 只作用于四元数指标，与通道权重可交换。从 $x+\hat\mu$ 平移到 $x$ 的伴随是从 $x-\hat\mu$ 平移到 $x$，所以对输入的梯度复用同一个 kernel，只交换 $\pm\mu$ 权重块并转置。第一层只有 plaquette 一个输入通道，小于 tensor core 的最小宽度 16，仍用 `transport` 和 einsum。
- 矩阵乘积用 `tl.dot` 的 `tf32x3` 模式，把 float32 拆成两个 TF32 分量做三次乘积，精度与 float32 相同。

## 5. 训练主循环与参数保存

`train_su2_dm.py` 的 `main` 执行以下步骤。

1. 设置随机种子并创建 `OUT` 目录。`load_cfgs()` 一次读入全部构型，并放到默认的 CUDA 设备上。
2. 创建 `LCNN`、Adam 优化器和参数的指数移动平均 EMA。优化器使用 `fused=True`。
3. 用 `torch.compile` 编译 `loss_fn`，采用 `reduce-overhead`、完整图和固定形状。
4. 每个 epoch 随机打乱构型顺序，按 batch 计算损失、反向传播并更新参数。
5. 每次参数更新后更新 EMA。第 $k$ 次更新使用衰减率 $\min(0.999,(1+k)/(10+k))$，保留早期的逐渐增大过程。
6. 打印每个 epoch 的平均损失和累计时间。将 1 到 `EPOCHS` 之间的 10 个对数等间隔值取整并去重，在对应 epoch 保存参数文件。训练 500 个 epoch 时保存 1、2、4、8、16、32、63、126、251、500。

最后不足一个 batch 的构型被跳过。当前 L128、$\beta=200$ 训练使用固定的 5120 个构型，`BATCH = 64` 时每个 epoch 更新 80 次，恰好使用全部构型。此前的 L128 实验使用 512 个构型和 batch 32，L32 子集实验使用 1024 个构型和 batch 64，两者每轮均更新 16 次。完整数据实验采用 batch 256，每轮更新 400 次。已有的 L32 epoch 100 和 epoch 300 模型使用的是更早的 10000 个构型，batch 为 64，每轮更新 156 次，随机跳过 16 个构型。

保存文件的命名是 `ckpt_epoch0001.pt`、`ckpt_epoch0002.pt`，依此类推。文件包含当前网络参数 `model`、平滑参数 `ema`，以及 `ch`、`nlayers`、`sigma`、`noise_beta`、`L`、`beta`、`epoch`。

文件不含优化器状态，训练入口也没有继续训练的分支。现有参数文件主要供生成程序读取。训练代码输出日志到终端，目录中的 `train.log` 来自运行时的重定向或外部运行脚本。

## 6. 生成程序如何反向积分

### 6.1 恢复 EMA 网络并建立初态

`sample_su2_em.py` 从参数文件读取网络宽度、层数、$\sigma$ 和 `noise_beta`，随后将保存的 EMA 参数复制给模型。实际采样使用 EMA 权重。

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

默认从 $t=1$ 等间隔走到 $10^{-5}$，共 2000 步。因此

$$
h=0.000499995,\qquad
\max_i g(t_i)^2h\simeq0.16457.
$$

这里使用当前 $\beta=200$ 的噪声方案，原方案的最大单步方差约为 $0.098$。采样时每步直接调用一次 `heat_kernel_step`，没有训练中的子步拆分。程序检查单步方差不超过 `DELTA_MAX`。

最后一步将 `noise` 设为零，保留 score 漂移。它是程序主动采用的末步去噪处理。因此最终结果停在 $t=10^{-5}$，最后一小步也不是完整的随机更新。

### 6.3 构型生成与单位化

一次生成全部 `N_CFGS` 个构型，默认是 1000 个。各构型从独立 Haar 初态开始，并使用独立噪声。

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

日志中的 `exact` 指无限体积结果，不是任意有限周期格子的严格精确值。`training` 对照默认只读当前热浴文件的前 1000 个构型。文件重新生成后，这个标签仍保持原样，但新的 102400 个构型没有用于已有的 L32 epoch 100 和 epoch 300 模型训练。

生成文件保存在参数文件所在目录，命名为 `samples_epoch0100_steps2000.h5`。它包含

- `cfgs`。生成的复矩阵构型，轴顺序与热浴文件相同。
- `plaq`。每个生成构型的平均 plaquette。
- 文件属性。`beta`、`L`、`sigma`、`epoch`、`n_steps`、`plaq_exact`。采用新噪声方案时另存 `noise_beta`。

两个较大的 Wilson 回路只打印到终端，没有存成数据集。生成种子和终止时间 `EPS` 也没有保存在 HDF5 属性中。程序在输出前还会读取对应 $L$、$\beta$ 的热浴文件来做对照，因此直接运行整个采样脚本仍需要该文件存在。

## 8. 默认参数与运行方式

训练参数集中在 `train_su2_dm.py` 顶部。

- 物理参数为 `L = 128`、`BETA = 200.0`。
- 扩散参数为 `SIGMA = 14.0`、`DELTA_MAX = 1.0`、`EPS = 1e-5`。
- 网络参数为 `CH = 32`、`NLAYERS = 3`。
- 优化参数为 `BATCH = 64`、`EPOCHS = 500`、`LR = 2e-4`、`EMA_DECAY = 0.999`、`SEED = 0`。
- `DATA` 默认指向 `trainningdata/cfgs_su2_beta=200.0_128^2_n5120.h5`。
- `OUT` 默认指向 `results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise`。

生成参数集中在 `sample_su2_em.py` 顶部，为 `N_CFGS = 1000`、`N_STEPS = 2000`、`EPS = 1e-5`、`SEED = 1`。训练与生成中的同名常量各自独立。

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

当时的默认设置生成了 1000 个构型，每个构型经过 2000 步反向更新。已有记录给出

- 平均 plaquette。热浴前 1000 个构型为 $0.43379\pm0.00080$，生成为 $0.43072\pm0.00078$，无限体积值为 $0.43312743$。
- $W(1,2)$。热浴为 $0.18881\pm0.00103$，生成为 $0.18561\pm0.00109$，无限体积值为 $0.18759937$。
- $W(2,2)$。热浴为 $0.03489\pm0.00106$，生成为 $0.03370\pm0.00109$，无限体积值为 $0.03519352$。

这次生成的 plaquette 均值比无限体积值低约 0.56%，相差约 3.1 个所报标准误差。已有结果说明仍存在需要检验的偏差，单次运行不能区分网络误差、积分误差与统计涨落各占多少。训练 loss 接近稳定也不能代替物理检验。

日志记录生成结束时、单位化之前的最大四元数长度偏差为 $6.6\times10^{-6}$。当时的训练数据文件中的 10000 个 plaquette 平均值为 $0.43319369$，其未作自相关修正的标准误差约为 $0.00025188$。

编写本文时另外用前 4 个训练构型核对了读写转换与 plaquette。矩阵转换往返的最大绝对误差约为 $4.2\times10^{-8}$，重算 plaquette 与文件中保存值的最大差约为 $1.2\times10^{-8}$。复用现有规范等变检查，在 CPU 上对随机初始化网络得到约 $1.5\times10^{-6}$ 的相对偏差。这些检查用于核对数据约定与说明中的公式，不构成生成精度的额外验证。

## 10. 对照源码的阅读顺序

先读训练文件的 `load_cfgs` 和 `main`，确定数据怎样进入训练。随后读 `loss_fn`，把 $t$、$K$、$\delta$、`Ut` 和 `Y_last` 对应起来。再读 `heat_kernel_step`、`heat_kernel_score` 和 `LCNN.forward`，分别理解噪声、监督目标和网络输出。

最后读采样文件的 `main`、`em_sample`、`em_step`。前三节确定了它反演的前向过程，第六节给出了更新式。`report`、`wilson` 和 `to_matrix` 则负责检查物理量并恢复磁盘布局。

## 11. Wilson 回路比较

从项目根目录运行。

```bash
/work/gw97/w97000/envs/nenv2/bin/python src/SU2/compare_su2_wilson.py
```

输入路径、回路列表 `RT`、测量批量 `BATCH` 和热浴块长 `HB_BLOCK` 放在脚本顶部。默认读取 `su2_L32_beta2.0_lcnn_hk_c16l3` 目录中的 1000 个 DM 构型，以及对应 $L=32$ 热浴文件中的全部 102400 个构型。新的热浴构型没有用于已有的 L32 模型训练。此前保存的比较结果使用旧的 10000 个训练构型作为热浴对照。

默认分别测量全部 $1\le R,T\le4$ 的 16 种矩形回路，不合并交换方向的回路。程序复用现有的 `load_cfgs` 和 `wilson`，以双精度在 CUDA 上分批测量。`load_cfgs` 仍一次读入整个构型文件，新增的 `dtype` 参数只改变测量精度，训练和采样继续默认使用单精度。

每个构型先对全部起点取空间平均。DM 误差为这些构型均值的样本标准差除以 $\sqrt{N}$。热浴按保存顺序将每 50 个构型组成一个块，用块均值的样本标准差除以块数的平方根。若末尾不足一个完整块，则均值和误差都不使用该块。改变热浴数据时，应检查块长增大后误差是否稳定，确保块长足够覆盖自相关。

解析对照为无限体积的 $[I_2(\beta)/I_1(\beta)]^{RT}$。当前 $\beta=2$、$L=32$ 和这些小回路的有限体积修正可忽略。这个近似不能直接用于小体积或接近整个格子大小的回路。

输出位于 DM 构型所在目录，沿用输入文件名并添加 `_wilson` 后缀。PDF 和 PNG 上图显示均值及一个标准误差，下图显示相对于解析值的差及同样的误差棒。上图使用对称对数坐标，零附近保持线性，保留大回路的负估计值和跨零误差棒。CSV 保存各回路的 $R,T,RT$、解析值和两个系综的均值与标准误差。

同一回路的 DM、热浴和解析值使用相同横坐标。图宽采用 7 英寸通栏尺寸，字体、配色、点型、线宽、误差棒端帽及保存分辨率使用项目的 `../plots/cjp.mplstyles`。

## 12. L32 继续训练与 5120 个构型的比较

2026-10-01 用一次性脚本从 epoch 100 继续训练到 epoch 300，脚本事后已删除。全部 300 个 epoch 都使用旧的 10000 个构型。继续训练保持 `BATCH = 64` 和 `LR = 2e-4`，随机种子为 100，并恢复网络与 EMA 参数。原参数文件没有 Adam 状态，因此 Adam 重新初始化。训练循环与 `train_su2_dm.py` 相同，EMA 更新次数从参数文件中的 15600 继续累计，不重新预热。最后一轮损失为 2.5413。新数据已替换旧文件，这次训练不能直接按原数据重跑。

两个模型分别一次生成 5120 个构型，均使用 `SEED = 1`、`N_STEPS = 2000`、`EPS = 1e-3`。时间网格线性等间隔，最后一步仅保留漂移。生成时临时设置 `N_CFGS = 5120`，采样脚本中的默认值仍为 1000。构型文件分别为 `samples_epoch0100_steps2000_n5120.h5` 和 `samples_epoch0300_steps2000_n5120.h5`。

| 物理量 | 无限体积值 | epoch 100 | epoch 300 |
| --- | --- | --- | --- |
| $P$ | 0.43312743 | $0.43146558\pm0.00017748$ | $0.43190880\pm0.00017723$ |
| $W(1,2)$ | 0.18759937 | $0.18647280\pm0.00023417$ | $0.18690994\pm0.00023435$ |
| $W(2,2)$ | 0.03519352 | $0.03420687\pm0.00023480$ | $0.03449467\pm0.00023492$ |

16 种回路的均值与解析值之差的绝对值均减小。均方根偏差由 $7.93034\times10^{-4}$ 降到 $5.65775\times10^{-4}$，下降约 28.7%。这些回路相互相关，不能把它们当作 16 次独立检验。epoch 300 的 plaquette 仍低于解析值约 6.9 个标准误差，尚存在可分辨的偏差。

完整结果见 [比较数据](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/wilson_epoch0100_vs0300_n5120.csv)、[比较图](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/wilson_epoch0100_vs0300_n5120.pdf) 和 [生成日志](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sample_compare_n5120.log)。此前每个模型生成 1000 个构型的结果也保留在同一目录中。

## 13. 终止时间与采样调度的对照

2026-10-01 用一次性脚本做了这组对照，脚本事后已删除。实验使用 epoch 300 的 EMA 参数，固定 $\epsilon=10^{-4}$、5120 个构型和种子 1。分别比较 2000 步和 4000 步下的线性与 log 时间网格，以及最后一步加噪或不加噪。所有更新调用现有的 `sample_su2_em.em_step`。每次同时生成全部 5120 个构型。

log 调度采用二维标量场 [diffusion_lightning.py](../../../diffusion_lightning.py) 中 `_build_time_steps` 的同一表达式 `torch.exp(torch.linspace(0.0, np.log(eps), num_steps + 1))`。以 $s_i=i/N$ 表示，两种时间网格为

$$
t_i^{\rm linear}=1-(1-\epsilon)s_i,\qquad
t_i^{\rm log}=\exp(s_i\log\epsilon).
$$

2000 步时，线性步长为 $4.9995\times10^{-4}$。log 网格的第一步约为 $4.5946\times10^{-3}$，最后一步约为 $4.6158\times10^{-7}$。log 网格将约一半步数放在 $t<0.01$，线性网格在该区间约有 20 步。两种网格都用各自的实际步长 $t_i-t_{i+1}$ 计算漂移和噪声。

所有实验共享 Haar 初态，并在每条轨迹开始时恢复同一 GPU 随机状态。编译后的单步随机数可重复性在运行开始时核对。每种时间网格的末步加噪和不加噪构型从完全相同的倒数第二步状态分叉。误差比较按构型配对，使用差值的样本标准差除以 $\sqrt{5120}$。不同网格和不同步数之间共享随机数流，但不是同一条连续 Brownian 路径。

16 种回路的均方根偏差定义为 $\mathrm{RMS}=\sqrt{\sum_{R,T}(\overline W-W_0)^2/16}$。其统计不确定度用 2000 次按构型联合 bootstrap 估计，NumPy 种子为 20261001。每次对全部回路和实验组使用相同的构型索引，以保留它们的相关性。旧终止时间的历史对照只报告自身统计量，不参与配对差值计算。

物理对照仍以干净分布的 $W_0(R,T)=[I_2(\beta)/I_1(\beta)]^{RT}$ 为主。另列出当前前向扩散约定下的有限时间均值

$$
\Sigma(t)^2=\frac{\sigma^{2t}-1}{2\log\sigma},\qquad
W_t(R,T)=W_0(R,T)\exp\left[-\frac34(R+T)\Sigma(t)^2\right].
$$

指数中的因子来自每条边的基本表示 Casimir $C_F=3/4$ 和回路周长 $2(R+T)$。$t=10^{-4}$ 时 $P_t=0.4330624453$，低于干净分布值约 $6.50\times10^{-5}$。末步不加噪是额外的去噪处理，不应直接视为对完整反向 SDE 的同一种离散。

实验输出集中在 `results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-4`。HDF5 中每个实验组保存构型、逐构型 Wilson 回路、时间网格和实验参数。CSV 保存均值与标准误差，`_effects.csv` 保存配对差值，PDF 展示相对解析值的偏差和末步噪声效应。旧的 $\epsilon=10^{-3}$、2000 步、末步不加噪结果作为单独的历史对照，不参加配对差值计算。

2026-10-01 的八组结果如下。每组 5120 个构型。$P$ 后给出均值的标准误差，RMS 后给出联合 bootstrap 标准差。

| 步数 | 时间网格 | 末步噪声 | $P$ | RMS，单位 $10^{-4}$ |
| --- | --- | --- | --- | --- |
| 2000 | 线性 | 不加 | $0.4325080\pm0.0001772$ | $3.90\pm0.86$ |
| 2000 | 线性 | 加 | $0.4321667\pm0.0001773$ | $4.80\pm0.92$ |
| 2000 | log | 不加 | $0.4317576\pm0.0001760$ | $7.54\pm1.01$ |
| 2000 | log | 加 | $0.4317574\pm0.0001760$ | $7.54\pm1.01$ |
| 4000 | 线性 | 不加 | $0.4326625\pm0.0001741$ | $3.25\pm0.85$ |
| 4000 | 线性 | 加 | $0.4325081\pm0.0001740$ | $3.65\pm0.88$ |
| 4000 | log | 不加 | $0.4323928\pm0.0001768$ | $4.84\pm1.01$ |
| 4000 | log | 加 | $0.4323927\pm0.0001768$ | $4.83\pm1.01$ |

固定线性网格、2000 步和末步不加噪，终止时间由 $10^{-3}$ 改为 $10^{-4}$ 后，本次样本的 RMS 从 $5.66\times10^{-4}$ 降为 $3.90\times10^{-4}$，下降约 31.0%。线性 4000 步、末步不加噪的观测 RMS 最小，相对旧终止时间的历史结果下降约 42.5%。其 plaquette 仍低于干净分布的解析值约 2.67 个标准误差。

末步加噪减去不加噪的配对 plaquette 差值为

| 时间网格 | 2000 步 | 4000 步 |
| --- | --- | --- |
| 线性 | $(-3.4127\pm0.0778)\times10^{-4}$ | $(-1.5438\pm0.0553)\times10^{-4}$ |
| log | $(-1.45\pm2.43)\times10^{-7}$ | $(-1.21\pm1.69)\times10^{-7}$ |

线性网格的末步噪声显著降低 $P$，其影响随步长减小。log 网格的末步很短，开关末步噪声对当前总误差的影响很小，plaquette 差值在配对统计误差以内。末步不加噪提高了本次偏低的线性采样均值，但这项数值处理并没有消除全部偏差。

两种调度的差异在 4000 步时尚不能清楚分辨。固定末步不加噪，log 减线性的 plaquette 差值为 $(-2.697\pm2.463)\times10^{-4}$。线性减 log 的 RMS 差值为 $(-1.58\pm1.29)\times10^{-4}$，其 95% bootstrap 百分位区间包含零。因此本次观测最小值不足以证明线性调度在更高精度下仍显著更优。

固定末步不加噪，log 调度从 2000 步到 4000 步的 $P$ 变化为 $(6.352\pm2.425)\times10^{-4}$，线性调度为 $(1.545\pm2.485)\times10^{-4}$。这些结果显示 log 有步数依赖的迹象，但两档步数尚不足以证明积分已经收敛。实际后续比较可先以线性 4000 步、末步不加噪作为当前参考设置。

每条 2000 步轨迹约 384 秒，4000 步约 768 秒。末步两种处理共享此前的计算。最后一条 log 4000 步轨迹曾在第 500 步失去原进程，随后以相同参数重新生成，已完成的六组数据保留。八组保存的全部矩阵通过有限性和 SU(2) 约束检查。单位化后的长度平方误差均小于 $3.3\times10^{-7}$，抽查构型重算的 plaquette 与保存值之差小于 $10^{-15}$。

完整结果见 [均值与误差](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-4.csv)、[配对差值](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-4_effects.csv)、[两页比较图](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-4.pdf) 和 [运行日志](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-4.log)。进行该对照时，`sample_su2_em.py` 的默认 `EPS` 为 $10^{-3}$。

## 14. 将 EPS 进一步降到 1e-5

2026-10-01 固定 epoch 300、线性 2000 步、末步不加噪、5120 个构型和种子 1，只将终止时间从 $10^{-4}$ 改为 $10^{-5}$。生成复用上一节的对照程序，只把 `EPS` 改为 `1e-5`，步数只取 2000。初态与随机数流沿用上一节设置。对照完成后，已将 `sample_su2_em.py` 的默认 `EPS` 改为 `1e-5`。

| EPS | $P$ | 16 种回路的 RMS |
| --- | --- | --- |
| $10^{-4}$ | $0.4325080\pm0.0001772$ | $(3.902\pm0.863)\times10^{-4}$ |
| $10^{-5}$ | $0.4325715\pm0.0001771$ | $(3.741\pm0.846)\times10^{-4}$ |

观测 RMS 进一步下降约 4.11%，16 种回路中有 11 种的绝对偏差减小。配对 plaquette 差值为 $(6.346\pm0.235)\times10^{-5}$，两个系综逐构型 plaquette 的相关系数为 0.999912。联合 bootstrap 使用 2000 次重采样和种子 20261001。RMS 差值为 $(-1.605\pm0.485)\times10^{-5}$，95% 百分位区间为 $[-2.151,-0.315]\times10^{-5}$。本次对照支持小幅改善。

改善已经比从 $10^{-3}$ 降到 $10^{-4}$ 时小。$\epsilon=10^{-5}$ 的 plaquette 仍低于干净分布解析值约 3.14 个标准误差。线性步长仍约为 $5.0\times10^{-4}$，最后一次 score 计算在 $t\approx5.10\times10^{-4}$，降低 EPS 没有同时解决步长误差。精确有限时间分布从 $10^{-4}$ 到 $10^{-5}$ 的 plaquette 变化约为 $5.85\times10^{-5}$，与本次变化量同量级。

结果见 [比较数据](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-5.csv)、[比较图](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-5.pdf) 和 [运行日志](../../results/su2_L32_beta2.0_lcnn_hk_c16l3/sampling_eps1e-5.log)。同名前缀的 HDF5 保存新构型、旧系综的逐构型回路值和 RMS 的 bootstrap 样本。复用对照程序时同步生成的末步加噪构型也保存在其中。保存的构型通过有限性、SU(2) 约束和 plaquette 往返检查。

## 15. L32 的 1024 构型训练

2026-10-02 使用 [固定的 1024 构型子集](../../trainningdata/cfgs_su2_beta=2.0_32^2_n1024.h5) 从随机初始化开始训练。源文件为完整的 L32、$\beta=2$、102400 构型数据，抽取采用 NumPy 种子 0 均匀无放回抽样。子集保存原始构型编号 `parent_id`，其构型和 plaquette 均与源文件完全一致。

使用原有 `train_su2_dm.py`，仅修改数据路径、batch 和输出目录。网络为 16 通道、3 层，$\sigma=14$，学习率为 $2\times10^{-4}$，EMA 衰减率为 0.999，训练种子为 0。batch 为 64，每个 epoch 对固定 1024 个构型随机打乱一次，更新 16 次，没有丢弃构型。500 个 epoch 共更新 8000 次。

输出目录为 `results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500`，按 1、2、4、8、16、32、63、126、251、500 个 epoch 保存网络与 EMA 权重。训练使用 float32 和原有损失函数，完整记录见 [训练日志](../../results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500/train.log) 和 [检查记录](../../results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500/checks.log)。

500 个 epoch 已全部完成，训练循环耗时 92 秒，最后一轮损失为 2.5324，最后 20 轮平均损失为 2.54193。十个 checkpoint 的网络与 EMA 参数均为有限值，最终 EMA 更新次数为 8000。重新加载后，在四个扩散时间上的 score 输出均通过有限性检查。最终权重见 [epoch 500 checkpoint](../../results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500/ckpt_epoch0500.pt)。

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /work/gw97/w97000/envs/nenv2/bin/python3 -u src/SU2/train_su2_dm.py \
  > results/su2_L32_beta2.0_lcnn_hk_c16l3_n1024_b64_e500/train.log 2>&1
```

## 16. L128、β=200 的 512 构型训练

2026-10-02 从 [L128、β=200 源文件](../../trainningdata/cfgs_su2_beta=200.0_128^2.h5) 当时已完整写入且块索引已落盘的前 24795 个构型中，无放回随机抽取 512 个。抽取使用 `np.random.default_rng(0).choice(24795, 512, replace=False)`。原始编号排序后保存为 `parent_id`，范围为 67 到 24754，编号从 0 开始。固定子集见 [512 个训练构型](../../trainningdata/cfgs_su2_beta=200.0_128^2_n512.h5)。

抽取时源文件仍在生成，HDF5 根块索引尚未刷新。按照已落盘的叶节点索引读取完整构型块，并将结果保存为独立 HDF5。全部 512 个构型与第二次读取的源数据块逐字节相同，最大 SU(2) 矩阵误差为 $3.33\times10^{-16}$。子集 plaquette 均值为 $0.992510908894$，逐构型标准误差为 $2.18\times10^{-6}$。

源文件完成后，使用 h5py 按 `parent_id` 再次读取并核对，全部 512 个构型与源文件完全相同。重算 plaquette 与源文件保存值的最大差为 $1.13\times10^{-14}$。子集的 `plaq` 已替换为源文件的原始保存值，并补齐原始热浴参数属性。

训练复用 `train_su2_dm.py`，从随机初始化开始。网络为 16 通道、3 层，$\sigma=14$，学习率为 $2\times10^{-4}$，EMA 衰减率为 0.999，训练种子为 0。batch 为 32，每轮打乱固定子集并更新 16 次。500 个 epoch 共更新 8000 次。按 1、2、4、8、16、32、63、126、251、500 个 epoch 保存网络与 EMA 权重。

输出目录为 `results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500`。运行记录见 [训练日志](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500/train.log)，数据和参数检查见 [检查记录](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500/checks.log)。

500 个 epoch 已全部完成，训练循环耗时 641 秒，最后一轮损失为 2.0308，最后 20 轮平均损失为 2.01541。十个 checkpoint 的网络与 EMA 参数均为有限值，最终 EMA 更新次数为 8000。重新加载最终网络与 EMA 后，在 $t=10^{-5},0.005,0.1,1$ 上的 score 均为有限值。最终权重见 [epoch 500 checkpoint](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500/ckpt_epoch0500.pt)。

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /work/gw97/w97000/envs/nenv2/bin/python3 -u src/SU2/train_su2_dm.py \
  > results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500/train.log 2>&1
```

## 17. L128、β=200 的噪声方案对比

2026-10-02 保持第 16 节的固定训练子集、网络、优化器、随机种子、batch 32 和 500 epochs，改用第 2.1 节的噪声方案。从随机初始化重新训练，使用 $\Sigma(t)^2=(e^{At}-1)/200$，其中 $A=\log(1+200\Sigma_{\max}^2)=8.90788329039$。最大噪声仍为 $\Sigma_{\max}=6.07824071708$。

训练循环耗时 628 秒，最终损失为 2.0283，最后 20 轮平均损失为 2.02480，共更新 8000 次。十个 checkpoint 的网络与 EMA 权重均为有限值。最终 EMA 在 $t=10^{-5},0.005,0.077812782,0.1,1$ 上通过输出有限性检查。旧 checkpoint 读取后，在同一输入上的输出与修改前源码逐位相同。记录见 [训练日志](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise/train.log) 和 [检查记录](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise/checks.log)。

原方案和新方案均使用 epoch 500 的 EMA 权重、种子 1、2000 个均匀时间步、$\epsilon=10^{-5}$ 和末步仅漂移处理。通过运行时设置 `sample.N_CFGS = 512`，每次调用一次 `em_sample`，全部 512 个构型同时生成。新方案生成及保存耗时 658.9 秒。构型全部有限，保存后的最大四元数长度平方误差为 $3.26\times10^{-7}$。

双精度测量覆盖 $1\le R,T\le4$ 的 16 种回路，以及 $8\times8$、$12\times12$、$16\times16$。热浴对照为源文件编号 36864 到 102336、间隔 64 的 1024 个构型，与训练子集没有交集。热浴误差使用每 16 个选中构型分块。每个构型先取格点平均，再计算系综误差。采样日志中的 `training` 标签对应源文件前 512 个构型，分析另行使用实际训练子集的观测量。

| 回路 | 解析值 | 原方案 | 新方案 | 原方案相对偏差 | 新方案相对偏差 |
| --- | --- | --- | --- | --- | --- |
| $1\times1$ | 0.992509422 | $0.991708985\pm0.000002759$ | $0.991287728\pm0.000002804$ | −0.08065% | −0.12309% |
| $2\times2$ | 0.970372663 | $0.954217624\pm0.000031424$ | $0.952677308\pm0.000031814$ | −1.66483% | −1.82356% |
| $4\times4$ | 0.886654073 | $0.703368188\pm0.00033430$ | $0.696364755\pm0.00032902$ | −20.67163% | −21.46151% |
| $8\times8$ | 0.618040384 | $0.063815688\pm0.00088166$ | $0.059727162\pm0.00082235$ | −89.67451% | −90.33604% |
| $16\times16$ | 0.145904073 | $-0.000200285\pm0.00051809$ | $0.000150968\pm0.00049319$ | −100.13727% | −99.89653% |

这次修改没有改善生成质量。19 种回路中有 18 种的中心值离解析值更远。剩余的 $16\times16$ 回路在两种方案下都与零相容，远低于目标值。新方案的局域 plaquette 标准差为 0.00710621，热浴为 0.00611321。

只用新方案的平均 plaquette 拟合有效耦合，得到 $\beta_{\mathrm{eff}}=171.9191$。据此预测的 $8\times8$ 回路为 0.571193，仍远高于实测的 0.059727，因此偏差不能只解释成一个有效 $\beta$ 的偏移。相同面积的 $\frac12[W(1,4)+W(4,1)]-W(2,2)$ 为 $0.001458696\pm0.000007248$，也表明回路相关性仍有明显偏差。本次对比同时改变训练的噪声分布和有限步长生成在各噪声尺度上的步数，不能单独区分网络近似、训练误差和时间离散误差。

新结果位于 `results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise`。文件包括 [最终 checkpoint](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise/ckpt_epoch0500.pt)、[512 个构型及逐构型观测量](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise/samples_epoch0500_steps2000.h5)、[两页对比图](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise/samples_epoch0500_steps2000_comparison.pdf) 和 [完整数值表](../../results/su2_L128_beta200.0_lcnn_hk_c16l3_n512_b32_e500_betanoise/samples_epoch0500_steps2000_comparison.csv)。图中最后一栏显示 $100(W_{\mathrm{new}}-W_{\mathrm{old}})/W_{\mathrm{exact}}$，误差按相同种子和构型编号配对计算。


## 18. L128、β=200 的 5120 构型与 32 通道训练

2026-10-02 从完整的 102400 个热浴构型中无放回抽取 5120 个。使用 `np.random.default_rng(0).choice(102400, 5120, replace=False)`，排序后的原始编号保存为 `parent_id`，范围为 19 到 102397。固定子集见 [5120 个训练构型](../../trainningdata/cfgs_su2_beta=200.0_128^2_n5120.h5)。全部构型和 plaquette 与源文件逐项完全一致，最大四元数长度平方误差为 $6.66\times10^{-16}$。子集 plaquette 为 $0.992509311293\pm0.000000658362$。

网络通道数由 16 增至 32，层数保持 3。参数量从 33872 增至 146336，约为原来的 4.32 倍。训练复用 `train_su2_dm.py`，从随机初始化开始，保持第 17 节的 β 相关噪声方案、学习率和 EMA 设置。batch 为 64，训练 500 个 epoch，每轮更新 80 次，共更新 40000 次。训练种子为 0，保存轮次仍为 1、2、4、8、16、32、63、126、251、500。

500 个 epoch 已于 2026-10-02 完成，训练循环耗时 24437 秒。最终损失为 2.0318，最后 20 轮平均损失为 2.027500。十个 checkpoint 的网络和 EMA 权重均为有限值，最终 EMA 更新次数为 40000。启动时规范等变相对误差为 $2.3\times10^{-6}$。结果见 [训练日志](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/train.log)。数据核对、参数设置和进程编号见 [检查记录](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/checks.log)。新子集与第 17 节的 1024 个热浴对照构型有 49 个重合，后续独立检验需要从本次训练子集之外选取对照。

### 18.1 512 个构型的反向生成与粗格起点对比

两条路径均使用 epoch 500 的 EMA 权重、种子 1、终止时间 $10^{-5}$ 和末步仅漂移。每条路径设置一次随机种子，按四批、每批 128 个构型调用现有的 `em_sample`，各保存 512 个构型。Haar 起点使用完整的 2000 步均匀时间网格。

粗格起点复用此前的 512 个 L64、$\beta_c=50.3725225668$ 构型和 relaxed 前处理。名义噪声为 $\Sigma^2=0.2$，沿用旧实验网格上的实际值 $\Sigma^2=0.200218905171446$，对应 $\Sigma=0.447458272$、$t_\star=0.417005836963654$ 和 834 个反向步。这里的 0.2 指累计方差。前处理种子为 20261002，加噪种子为 20261003。加噪后 20 种回路的逐构型测量值与旧实验完全一致。

独立热浴对照使用 1024 个构型，原始编号均排除了新旧两套训练子集。Wilson 回路以 float64 计算，先对每个构型的格点求平均，再估计系综误差。Haar、粗格起点和热浴分别按 1、4、16 个构型分块。下表误差为一个标准误差。

| 回路 | 无限体积解析值 | Haar 起点 | $\Sigma^2\simeq0.2$ 粗格起点 | 旧模型粗格起点 |
| --- | --- | --- | --- | --- |
| $1\times1$ | 0.992509422 | $0.991444643\pm0.000002692$ | $0.992184910\pm0.000002153$ | $0.992152296\pm0.000002163$ |
| $4\times4$ | 0.886654073 | $0.710536356\pm0.000321042$ | $0.833888854\pm0.000140458$ | $0.833002232\pm0.000138902$ |
| $8\times8$ | 0.618040384 | $0.070496435\pm0.000819293$ | $0.439786233\pm0.000613238$ | $0.438739444\pm0.000634930$ |
| $12\times12$ | 0.338678729 | $-0.000253027\pm0.000649376$ | $0.179634190\pm0.000876279$ | $0.178880830\pm0.000932418$ |
| $16\times16$ | 0.145904073 | $0.000375411\pm0.000515372$ | $0.059305506\pm0.000866912$ | $0.058011572\pm0.000915231$ |

Haar 的 $8\times8$ 回路由旧模型的 0.05973 增至 0.07050，仍远低于解析值 0.61804。粗格起点的 $8\times8$ 和 $16\times16$ 回路相对旧模型的差异分别为 1.70 和 1.76 个配对标准误差。两组新结果仍有明显的大回路偏差。新旧实验同时改变了通道数、训练构型数和参数更新次数。

全部 1024 个输出构型的分量均为有限值，矩阵共轭结构精确满足 SU(2) 形式。Haar 和粗格起点的最大四元数长度平方误差分别为 $3.12\times10^{-7}$ 和 $3.20\times10^{-7}$。Haar 有一个格点的 plaquette 因单精度舍入超过 1 约 $3.85\times10^{-8}$，直方图按物理区间裁剪分箱端点，回路均值保留原始测量值。

Haar 生成耗时 2283.5 秒。粗格起点生成耗时 974.2 秒，独立 short-g 作业 `3474621` 已正常结束。

结果包括 [Haar 构型](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/samples_epoch0500_steps2000.h5)、[粗格起点构型](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/reverse_L64_to_L128_Sigma2_0p2_n512_steps0834.h5)、[对比图](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/generation_comparison.pdf)、[20 种回路的数值表](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/generation_comparison.csv) 和 [分析记录](../../results/su2_L128_beta200.0_lcnn_hk_c32l3_n5120_b64_e500_betanoise/generation_comparison.log)。HDF5 保存了逐构型观测量、噪声设置、原始粗格编号和独立热浴对照编号。

## 19. 训练与生成的速度优化

2026-10-23 重写了 `train_su2_dm.py` 中的四处计算。网络的数学定义、参数名称和形状都不变，旧 checkpoint 可以直接读取，`sample_su2_em.py` 和 Prolongation 下的脚本无需修改。

1. `qbilinear` 改为三个融合的 Triton kernel，见第 4 节末。
2. 输入通道等于 `ch` 的层用 `TConv` 计算平行移动与卷积，见第 4 节末。
3. 训练加噪改为 `_brownian`，只计算有效子步，见 3.1 节。
4. readout 最后的逐格点收缩由 einsum 改为广播乘积求和。einsum 会被拆成 16 次批量 cuBLAS 调用，前向加反向约 7.4 ms，改写后约 1.9 ms。

在 GH200 上，$L=128$、batch 64、32 通道 3 层、`reduce-overhead` 编译时的计时如下。

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 每次参数更新 | 609 ms | 85.5 ms |
| 每个 epoch，80 次更新 | 48.8 s | 6.8 s |
| 500 个 epoch | 约 6.8 h | 约 57 min |
| 训练峰值显存 | 68.8 GiB | 8.0 GiB |
| 32 个构型的一步 EM 采样 | 69.4 ms | 11.9 ms |
| 32 个构型采样的峰值显存 | 11.8 GiB | 0.6 GiB |

修改前每步约 70% 的时间花在 `qbilinear` 上，主要是物化的中间量 $A$ 及其梯度的读写。修改后 `qbilinear` 的三个 kernel 每步合计约 59 ms，占 69%。`TConv` 约 10 ms，加噪约 0.2 ms。512 个 $L=128$ 构型可以一次生成，每步 183 ms，峰值显存 9.7 GiB，2000 步约 6 分钟。第 18.1 节分四批的 Haar 生成用了 2283.5 秒。

正确性检查如下。

- 用 epoch 500 的权重和 8 个加噪的 $L=128$ 构型，以 float64 运行原代码作为参考。修改前 float32 输出的相对误差为 $3\times10^{-7}$ 到 $3.3\times10^{-6}$，修改后为 $4\times10^{-7}$ 到 $3.0\times10^{-6}$。全部参数梯度的最大相对误差分别为 $9.5\times10^{-6}$ 和 $5.3\times10^{-6}$。随机局域规范变换下的等变偏差为 $1.0\times10^{-6}$。
- 每个 kernel 也单独与 float64 参考比较，输出和两种梯度的相对误差在 $2\times10^{-7}$ 到 $8\times10^{-7}$ 之间。
- 加噪在 12 个 $t$ 上比较新旧实现，$K$ 从 1 到 37。每个 $t$ 抽 20 次，共 655360 条链，比较 $\langle G_0\rangle$、$\langle G_0^2\rangle$、$\langle|\boldsymbol Y_K|^2\rangle$ 和 $\langle\boldsymbol Y_K\cdot\boldsymbol G\rangle$，其中 $G=U_tU_0^\dagger$。48 项中 46 项的差在 2.2 个标准误差以内。两个例外都在 $t=10^{-5}$，即 $\langle G_0\rangle$ 和 $\langle G_0^2\rangle$。此时 $1-G_0\approx1.7\times10^{-7}$，已到 float32 在 1 附近的分辨率，新旧实现都偏离解析值 $e^{-3\Sigma^2/8}$，旧实现偏离更多。其余 $t$ 上两者与解析值之差都在 3 个标准误差以内。
- 用相同种子、数据和超参数重新训练 10 个 epoch，每个 epoch 的平均损失与第 18 节原日志之差不超过 $3\times10^{-4}$。
- 用 epoch 500 的网络参数从 $t=0.1$ 起反向积分 200 步，相同种子下新旧轨迹的最大差为 $1.3\times10^{-6}$。

加噪的随机数分配已经改变，同一种子下的噪声与旧代码不同，分布不变。矩阵乘积的求和顺序也不同。因此新代码不能逐位复现旧运行，只在统计上等价。

剩余时间主要在 `qbilinear` 的三个 kernel。把它们的 `tf32x3` 换成单次 TF32，单个 kernel 约快 1.5 到 2 倍，但相对误差升到约 $10^{-3}$，目前没有采用。
