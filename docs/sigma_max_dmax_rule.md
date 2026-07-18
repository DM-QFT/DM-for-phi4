真正想最小化的目标是逐点强残差的平方(做了参考点减除、按体积归一):

$$r^2 = \big(d + c\big)^2, \qquad d := \mathrm{tr},J(\phi) - \mathrm{tr},J(0),\quad c := -\langle v, F_t\rangle - \partial_t S$$

其中 c 是确定性的、可以精确算;但散度差 d 只能靠 Hutchinson 探针随机估计(strong_direct_2d.py:243-247):每个探针给出 $D_i = d + \eta_i$,$\eta_i$ 是零均值的探针噪声($\mathbb{E}[D_i] = d$,来自随机 Rademacher 方向 + 有限差分)。

朴素做法是取探针平均 $\bar D$ 然后算 $(\bar D + c)^2$。但

$$\mathbb{E}\big[(\bar D + c)^2\big] = (d+c)^2 + \mathrm{Var}(\bar D) = r^2 + \tfrac{1}{m}\mathrm{Var}(\eta)$$

多出来的方差项依赖于网络参数(它由速度场 Jacobian 的非对角结构决定)。如果直接最小化它,梯度会额外推着网络去压低 Jacobian 涨落——一种虚假的正则化,把最优解从真正的 PIK 速度场上拉偏。

代码的做法(strong_direct_2d.py:259-262)是用 U 统计量消掉这个方差项:

$$\widehat{d^2} = \frac{S_1^2 - S_2}{m(m-1)} = \frac{1}{m(m-1)}\sum_{i\neq j} D_i D_j$$

因为不同探针独立,$\mathbb{E}[D_iD_j] = d^2$(不含 $\mathbb{E}[D_i^2] = d^2 + \mathrm{Var}$ 的对角项——分子里恰好把 $S_2=\sum D_i^2$ 减掉了)。于是

$$\mathbb{E}[\text{loss}] = \frac{d^2 + 2dc + c^2}{V^2} = \frac{r^2}{V^2} \ \ge 0$$

期望严格非负,但样本值不是。把 $D_i = d + \eta_i$ 代入展开:

$$\text{loss}\text{sample} = (d+c)^2 ;+; 2(d+c),\bar\eta ;+; \underbrace{\frac{1}{m(m-1)}\sum{i\neq j}\eta_i\eta_j}_{\text{零均值,可正可负}}$$

最直观的例子:m=2,真值 $d+c=0$,两个探针噪声 $\eta_1=+\varepsilon,\ \eta_2=-\varepsilon$,则 loss $= \eta_1\eta_2 = -\varepsilon^2 < 0$。

为什么"必然"如此
训练收敛时 $r^2 \to 0$。假设存在一个既无偏、又恒非负的估计量:在 $r^2=0$ 处它的期望为 0 且不能取负,那它必须几乎必然恒等于 0——但估计量是连续随机噪声(随机探针方向)的函数,这不可能。所以无偏性和非负性不可兼得,这里选择了无偏性(对 SGD 来说这是对的选择:样本 loss 可以为负,但它的梯度是真目标 $\nabla_\theta, r^2/V^2$ 的无偏估计,这正是随机优化唯一需要的性质)。

对读图的推论
回看昨天那张 loss 曲线:

前中期(loss ≫ 噪声):$(d+c)^2$ 主导,loss ≈ 真实 r²,单调下降有意义;
2500 步以后(loss 在 ±2×10⁻³ 变号震荡):$(d+c)^2$ 已经掉到噪声项之下,logged loss 读到的是 $\sum_{i\neq j}\eta_i\eta_j$ 的涨落,即探针噪声底。此后 loss 的数值大小不再反映残差本身,真实 r² 比涨落幅度更小;有意义的收敛指标变成涨落包络是否收窄(增大 --probes 或 batch 会压低噪声底),以及最终的可观测量对比。
这也解释了为什么"loss 早就到零附近了,step-4000 的观测量却还偏 15–22%":loss 已经探不到残差了,但逐点残差的微小剩余在反向 ODE 沿整条轨迹积分后仍会累积成可见的输运偏差——所以后半程的 lr 退火依然在实质性地改进模型,只是 loss 曲线上看不出来。

