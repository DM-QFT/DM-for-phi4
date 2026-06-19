# 为什么 VE-SDE 的最大噪声尺度要匹配数据的最大配对距离

这是 Song & Ermon 2020 *"Improved Techniques for Training Score-Based Generative Models"* 中 **Technique 1** 的核心结论。直觉一句话：**最大噪声必须大到能让先验"看见"所有的数据 mode，否则反向过程从一开始就采不到远离起点的 mode**。下面拆开讲。

---

## 1. VE-SDE 的反向采样起点

VE 方案的前向过程是

$$
x_t = x_0 + \sigma(t)\,\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I)
$$

在 $t=T$ 时，$\sigma(T) = \sigma_{\max}$。条件分布是 $p_T(x|x_0)=\mathcal N(x_0,\sigma_{\max}^2 I)$，因此边际分布是

$$
p_T(x)=\int p_{\text{data}}(x_0)\,\mathcal N(x;x_0,\sigma_{\max}^2 I)\,dx_0.
$$

采样时我们**不能**从这个真实的 $p_T$ 取样（它依赖未知的 $p_{\text{data}}$），只能近似为一个简单先验：

$$
\pi(x) \approx \mathcal N(0,\sigma_{\max}^2 I).
$$

那么这两者要尽量接近，即 $p_T(x)\approx \pi(x)$。

**第一个约束就来了：** 只有当 $\sigma_{\max}$ 远大于数据本身的尺度时，$p_{\text{data}}(x_0)$ 在高斯卷积里被"涂抹"得几乎看不见，$p_T$ 才会接近以 0 为心、$\sigma_{\max}$ 为宽的各向同性高斯。如果数据点散布的直径是 $D_{\max}$，那么这个"涂抹"要求

$$
\sigma_{\max} \gtrsim D_{\max}.
$$

否则 $p_T$ 仍然保留 $p_{\text{data}}$ 的多峰结构，而采样器却从单峰高斯 $\pi$ 出发，**起点分布偏移**会直接污染整个反向链。

---

## 2. Mode 之间的可达性 —— Song & Ermon 的核心论证

更严格的论证来自 score field 在不同 mode 之间是否"通"。设两个数据点 $x_i, x_j$，距离 $D=\|x_i-x_j\|$。在最大噪声尺度下，扰动核 $\mathcal N(\cdot; x_i,\sigma_{\max}^2 I)$ 在 $x_j$ 处的相对密度为

$$
\frac{\mathcal N(x_j; x_i,\sigma_{\max}^2 I)}{\mathcal N(x_i; x_i,\sigma_{\max}^2 I)}=\exp\!\left(-\frac{D^2}{2\sigma_{\max}^2}\right).
$$

- 若 $\sigma_{\max}\ll D$：这个比值指数级趋于 0。意味着以 $x_i$ 为中心的噪声样本几乎"看不到" $x_j$，score $\nabla_x\log p_{\sigma_{\max}}(x)$ 在两个 mode 中间的区域是两个不相交的吸引盆，反向 Langevin/概率流 ODE 从一个盆里**几乎不可能**跨到另一个。后果就是 **mode collapse / 覆盖不全**。
- 若 $\sigma_{\max}\gtrsim D$：两个 Gaussians 显著重叠，score field 是平滑过渡的单一吸引盆。反向链从先验任意位置出发，能渐进收敛到任一 mode。

把这个论证对**所有数据对**做并集，自然得到

$$
\boxed{\sigma_{\max}\gtrsim \max_{i,j}\|x_i-x_j\| = D_{\max}.}
$$

这就是 Song & Ermon 给出的"choose $\sigma_{\max}$ as the maximum pairwise Euclidean distance"规则。

---

## 3. 上界：为什么不能直接拉到无穷大

既然越大越好，为什么不直接 $\sigma_{\max}\to\infty$？三个反作用：

1. **信噪比与训练样本效率**：DSM 损失的有效信号 $\propto \|x_0\|^2/\sigma^2$，$\sigma$ 巨大时网络看到的几乎是纯噪声，回归目标 $-\varepsilon/\sigma$ 方差爆炸，梯度信号很弱。
2. **离散化误差**：反向 SDE 需要从 $\sigma_{\max}$ 退火回 $\sigma_{\min}$。$\sigma_{\max}$ 越大，需要的步数越多，否则每步 $\Delta\sigma$ 过大、Euler-Maruyama 误差累积。
3. **score 网络的容量限制**：噪声越大，score field 越光滑、信号越微弱；但 $\sigma_{\max}\to\infty$ 时 score $\to -x/\sigma^2$ 又趋于无信息。

所以 $\sigma_{\max}\approx D_{\max}$ 是 **"刚好够用的最小值"**，是 Pareto 上最经济的选择。

---

## 4. 维度依赖的精细化

严格地，高斯的"典型半径"在 $d$ 维下是 $\sigma\sqrt d$ 而不是 $\sigma$（concentration of measure）。所以理论上"两个 mode 的高斯壳层是否相交"用的是

$$
\sigma_{\max}\sqrt d \gtrsim D_{\max}\quad\Longrightarrow\quad \sigma_{\max}\gtrsim D_{\max}/\sqrt d.
$$

Song & Ermon 在论文里给出过这个 $\sqrt d$ 修正的版本（Technique 1 的精确表述），用 $3\sigma$ 规则得到 $\sigma_{\max}\ge D_{\max}/(3\sqrt d)$ 之类的下界。实际中**直接取 $\sigma_{\max}\approx D_{\max}$（不除 $\sqrt d$）**仍然是社区默认做法，原因是：

- 它给一个安全余量，覆盖了 mode 在流形上而非点云的情况。
- 数据流形局部维度通常远小于环境维度 $d$，$\sqrt d$ 修正会**过低估**。
- 训练对 $\sigma_{\max}$ 的取值有相当 robust 的范围，过大一倍代价不高。

---

## 5. 回到本仓库的参数化

[CLAUDE.md](../CLAUDE.md) 里写的经验规则是

> **Empirical sigma rule:** pick $\sigma$ so that `std(t=1) ≈ 2·D_max_norm`

注意这里两件事：

**(a) 参数化的差异。** 本仓库的 VE 用 $\sigma^t$ 形式，所以

$$
\text{std}(t=1)^2 = \int_0^1 \frac{d(\sigma^{2t})}{dt}\,dt = \frac{\sigma^2-1}{2\ln\sigma},
$$

这个 `std(t=1)` 才相当于 Song 论文里的 $\sigma_{\max}$。所以仓库代码里的 `sigma` **不是** $\sigma_{\max}$，配置时要先反解出 `std(t=1)`。

**(b) 系数 2 的来源。** 比 Song 的 $1\times D_{\max}$ 多了一倍：

- $D_{\max\_\text{norm}}$ 是已归一化到 $[-1,1]$ 的数据空间内的**最大配对距离**。归一化后数据被压在一个有限盒子里，$D_{\max\_\text{norm}}\le 2\sqrt d$。
- 取 $2\times$ 是经验上的安全余量：保证以**任意** mode 为中心的扰动核都能覆盖**所有**其他 mode（最坏情况下 mode 在直径两端，从其中一端出发要走过完整的 $D_{\max}$，再加 $\sigma$ 量级的高斯"半径"做缓冲）。
- 它也补偿了归一化后的物理/图像数据通常有重尾或外点，纯 $D_{\max}$ 略偏紧。

经验上这条规则的可观察后果：

- $\sigma_{\max}$ 取得偏小：反向采样的初始几步（小 $t$ 大噪声阶段）score 信号弱、$p_T \neq \pi$，表现为生成样本聚集在训练集均值附近，多样性差；在 $\phi^4$ 上会看到 $\langle |M|\rangle$ 等观测量的直方图比 MC 窄。
- $\sigma_{\max}$ 取得过大：训练损失收敛慢，需要更长 schedule / 更多采样步数才能稳定；在 log schedule 下尤其敏感。

---

## 一句话总结

**反向过程的起点高斯必须"宽到能把所有数据 mode 罩进去"，否则跨 mode 的 score 信号在最大噪声尺度上是断的；而数据 mode 间的最大间距就是 $D_{\max}$，所以 $\sigma_{\max}\propto D_{\max}$ 是这个"罩得住"条件的最小充分尺度，本仓库取 $2\times$ 是安全余量。**

---

# 附录：为什么 $\sigma_{\max}$ 必须随维度 $d$ 增长，而不是固定常数

一个很自然的反问：如果数据每个维度的方差都是 $O(1)$（与 $d$ 无关），为什么噪声不能也用一个固定值？为什么必须让 $\sigma_{\max}$ 随 $D_{\max}\sim\sqrt d$ 一起增长？

下面三层把这件事讲清楚。

## A1. 直觉里对的一半

设数据每个维度独立 $\sim\mathcal N(0,1)$：

- 数据本身的**逐维度尺度**是 $O(1)$，与 $d$ 无关。
- 为了"淹没单维度信号"，**逐维度**噪声 std 只要 $\gg 1$ 就够，**也与 $d$ 无关**。

问题出在 score-based diffusion 的衡量标准是**整段欧氏距离**，不是逐维度距离。两个独立 $\mathcal N(0,I_d)$ 样本的距离平方

$$
\|x_i-x_j\|^2 = \sum_{k=1}^d (x_i^{(k)}-x_j^{(k)})^2 \approx 2d \pm O(\sqrt d),
$$

由 CLT 集中在 $\sqrt{2d}$。所以 **$D_{\max}\sim\sqrt d$**，即使每个维度方差固定为 1。这是几何上的"维度灾难"：欧氏距离把 $d$ 个独立的小波动**累加**起来。

## A2. 单纯"先验覆盖数据"只要求 $\sigma\gtrsim 1$

反向采样起点 $\pi = \mathcal N(0,\sigma_{\max}^2 I_d)$。从它采出的样本典型范数是 $\sigma_{\max}\sqrt d$。数据点典型范数也是 $\sqrt d$。

要 prior "罩住" 数据集：

$$
\sigma_{\max}\sqrt d \gtrsim \sqrt d \;\Longrightarrow\; \sigma_{\max}\gtrsim 1.
$$

**若只要求 prior 能 reach 到任意一个 mode，逐维度 $\sigma\approx 1$ 就够了，与 $d$ 无关。** 但这只是必要条件，不是充分条件。

## A3. 真正卡瓶颈的是 mode 之间的"score 联通性"

关键不在 prior 能否 reach 到某个 mode，而在 score field $\nabla\log p_{\sigma_{\max}}(x)$ 在 mode 之间是否**联通**。

$$
p_{\sigma}(x)=\frac{1}{N}\sum_{k=1}^N \mathcal N(x;x_k,\sigma^2 I_d),\qquad
\nabla\log p_\sigma(x) = \frac{\sum_k w_k(x)\,(x_k-x)/\sigma^2}{\sum_k w_k(x)},\; w_k=\mathcal N(x;x_k,\sigma^2 I_d).
$$

这是个 softmax 加权平均。**多 mode 是否融合，看 softmax 是否塌缩成 one-hot。**

两个 mode 中心 $x_i,x_j$ 距离 $D$ 的两个高斯的统计可分性：

$$
\mathrm{KL}\big(\mathcal N(x_i,\sigma^2 I)\,\|\,\mathcal N(x_j,\sigma^2 I)\big) = \frac{D^2}{2\sigma^2},
\qquad
\mathrm{BC} = \exp\!\left(-\frac{D^2}{8\sigma^2}\right).
$$

**关键：两式分母里都没有 $d$ 来对冲 $D^2\sim d$。** 代入 $D\sim\sqrt d$：

$$
\mathrm{KL}\sim\frac{d}{2\sigma^2}.
$$

- 若 $\sigma=O(1)$（你想要的"固定噪声"）：$\mathrm{KL}\sim d$，**随维度线性增长**。两个 mode 的扰动核高维下几乎完全可分（$\mathrm{TV}\to 1$）。softmax 是剧烈的 **winner-take-all**，score 完全指向最近的 $x_k$。
- 若 $\sigma\sim\sqrt d \sim D_{\max}$：$\mathrm{KL}\sim 1$，mode 高斯有 $\Theta(1)$ 重叠，softmax 是温和的 mixture，score 在 mode 之间平滑过渡。

**逐维度 $\sigma=O(1)$ 在低维下足够，但高维下不够。** 它能让 prior 覆盖数据范围（A2），但不足以让 mode 之间的 score 联通（A3）。后者才是反向 SDE 能够探索整个分布的前提。

## A4. 几何直觉：为什么 $D^2/\sigma^2$ 里没有 $\sqrt d$ 自动对冲

很多人会觉得：高维下"距离 $\sqrt d$"是一种**虚假的距离**——它只是把 $d$ 个小波动累加。两个数据点逐维度差别很小，应该"很相似"。

错觉来自一个直觉：低维空间里距离接近的两点"长得像"。但高维空间里，**两个独立采样的 $\mathcal N(0,I_d)$ 样本距离 $\sqrt{2d}$ 是 typical 的**——所有数据点恰好分布在这个 $\sqrt d$ 量级的壳上、彼此 $\sqrt d$ 距离。

要让一个高斯核**同时**覆盖原点和距离 $\sqrt d$ 处的另一点，核的"半径"必须 $\gtrsim\sqrt d$。$\mathcal N(0,\sigma^2 I_d)$ 的有效半径是 $\sigma\sqrt d$，等式 $\sigma\sqrt d\gtrsim\sqrt d$ 确实只给 $\sigma\gtrsim 1$——这是"半径够长"。

但密度比

$$
\frac{p_\sigma(\text{中点}\,|\,x_i)}{p_\sigma(x_i\,|\,x_i)}=\exp\!\left(-\frac{D^2}{8\sigma^2}\right)
$$

里 $D^2/(8\sigma^2)$ 没有 $1/d$ 因子。**几何上"高维高斯有厚壳"不能消除 mode 间的指数压制**。

直观地：高维高斯虽然有半径 $\sigma\sqrt d$ 的壳，但壳**很薄**（厚度 $O(\sigma)$，相对厚度 $O(1/\sqrt d)$）。两个相距 $\sqrt d$ 的厚壳若 $\sigma\sim 1$，壳几乎不相交（壳间距 $\sqrt d$ 远大于壳厚 1）。要让两个厚壳"撞在一起"，须把厚度也提到 $\sqrt d$，即 $\sigma\sim\sqrt d$。

## A5. 一个具体的失败例子

$d=784$（MNIST flatten），数据归一化到 $[-1,1]^d$。

- 两个不同 digit 距离 $D_{\max}\approx\sqrt{784\cdot 4}\approx 56$ 最坏，典型 $\sim 20$。
- 若设"固定" $\sigma_{\max}=1$：
  - mode 高斯之间 $\mathrm{KL}\approx 20^2/2\approx 200$。**完全可分**。
  - $p_T(x)$ 仍是 60000 个清晰可分的小高斯团。
  - prior $\mathcal N(0,I_{784})$ 是原点附近半径 $\approx 28$ 的薄壳，与数据壳大致重合但都是"随机方向"，离任何真实 digit 都很远。
  - 反向 SDE **确定性 denoise** 到最近的训练 digit。没有多样性，等价于 nearest-neighbor retrieval。
- 若 $\sigma_{\max}=20\approx D_{\max}$：
  - $\mathrm{KL}\approx 0.5$。mode 高斯严重重叠。
  - $p_T\approx \pi$，反向 SDE 在所有 mode 之间自由分配概率质量。

## A6. 与本仓库 $\phi^4$ 的连接

$\phi^4$ 数据维度 $d=L^{d_\text{spatial}}$，例如 $L=64$、3D 时 $d=64^3\approx 2.6\times 10^5$。

- 晶格场高度相关（相干长度 $\xi\ll L$），**有效维度** $d_{\text{eff}}\ll d$，约为独立"区块"数。
- $D_{\max\_\text{norm}}$ 直接测量得到——隐式包含了相关性，给出数据流形真实直径。
- 取 $\sigma_{\max}\approx 2\cdot D_{\max\_\text{norm}}$ 自动适配 effective dim，比手动估计 $d_{\text{eff}}$ 再乘 $\sqrt{d_{\text{eff}}}$ 更稳健。这也是经验规则用 $D_{\max}$ 而非 $\sqrt d$ 的原因。

## A7. 附录一句话总结

**逐维度 $\sigma=O(1)$ 足以"淹没单维度信号"，但不足以"让相距 $\sqrt d$ 的 mode 在 score field 里联通"。后者是反向 SDE 的真正瓶颈：两个高斯的可分性 $\propto\exp(-D^2/\sigma^2)$，分母里没有 $d$ 对冲，所以 $\sigma$ 必须随 $D_{\max}\sim\sqrt d$ 一起增长。"维度灾难"在这里以"距离平方在指数上"的形式出现。**

---

# 附录 B：为什么以 $x_i$ 为中心的噪声要覆盖 $x_j$，而不是只破坏 $x_i$ 本身？

这是一个直指 DSM 训练目标和反向采样目标之间不对称性的问题。回答分四层。

## B1. 等价性：在固定的各向同性先验下，"破坏"等于"覆盖"

反向起点是 $\pi=\mathcal N(0,\sigma_{\max}^2 I)$——一个**固定的、跟数据无关的**分布。

"以 $x_i$ 为中心的噪声把 $x_i$ 破坏掉"精确地说是

$$
\mathcal N(x_i,\sigma_{\max}^2 I)\;\approx\;\pi=\mathcal N(0,\sigma_{\max}^2 I).
$$

对 $x_j$ 同理。**由传递性**

$$
\mathcal N(x_i,\sigma_{\max}^2 I)\;\approx\;\mathcal N(x_j,\sigma_{\max}^2 I).
$$

两个 mode 高斯彼此重叠不是独立的额外条件——**它是"每个 mode 高斯都被压扁到同一个先验"的自动推论**。

定量地：若 $\mathrm{KL}(\mathcal N(x_i,\sigma^2 I)\,\|\,\pi)=\|x_i\|^2/(2\sigma^2)$ 很小，那么对所有 $i,j$

$$
\mathrm{KL}\big(\mathcal N(x_i,\sigma^2 I)\,\|\,\mathcal N(x_j,\sigma^2 I)\big)=\frac{\|x_i-x_j\|^2}{2\sigma^2}\le \frac{(\|x_i\|+\|x_j\|)^2}{2\sigma^2}
$$

也跟着小。**"足够大的 $\sigma$ 破坏每个 $x_i$"等价于"足够大的 $\sigma$ 让 mode 高斯互相覆盖"。**

看似可逃避的出路是把 prior 设成多模——比如直接用 $p_T$ 作为反向起点。理论上可以，但你**采不到** $p_T$（它依赖未知的 $p_{\text{data}}$）。一旦 prior 被固定成各向同性高斯（实践中必然如此），上面的传递性就锁死了。

## B2. 采样视角：先验样本不知道自己"来自" $x_i$

DSM 训练里，每个噪声样本 $x_t = x_0+\sigma\varepsilon$ 是从某个具体 $x_0$ 出发的——"出发点"在训练时已知，回归目标 $-\varepsilon/\sigma$ 就是"指回 $x_0$"。

反向采样时是反过来的：

$$
x_T\sim\pi \;\;\xrightarrow{\;\text{reverse SDE}\;}\;\; x_0\sim p_{\text{data}}.
$$

起点 $x_T$ 是匿名的先验样本，它**不知道自己应该回到哪个 $x_k$**。score 网络在 $x_T$ 输出的方向是所有训练点拉力的 softmax 加权：

$$
\nabla\log p_T(x_T)=\sum_k w_k(x_T)\,\frac{x_k-x_T}{\sigma_{\max}^2},\quad w_k(x_T)\propto\exp\!\left(-\frac{\|x_T-x_k\|^2}{2\sigma_{\max}^2}\right).
$$

"以 $x_i$ 为中心的噪声覆盖 $x_j$"的物理含义是：**在 $x_j$ 扰动核的典型支撑区域中，$x_i$ 扰动核也有不可忽略的密度**。等价于在那些区域里 $w_i(x)$ 和 $w_j(x)$ 都不可忽略——score 是 $x_i,x_j$ 拉力的混合。

如果两个噪声不互相覆盖：

- 在 $x_i$ 附近：$w_i\approx 1$，$w_j\approx 0$，score 完全拉回 $x_i$。
- 在 $x_j$ 附近：$w_j\approx 1$，$w_i\approx 0$，score 完全拉回 $x_j$。
- 在"两者都不附近"（先验典型支撑区域）：$w_i,w_j$ 都微小但比例由谁更近决定，softmax 退化成 winner-take-all。

**第三种情况是要命的**——先验典型样本恰好就在"两者都不附近"那里。高维各向同性高斯不集中在任何一个 $x_k$ 附近，所以每个 $x_T$ 被锁定到"碰巧最近的那个 $x_k$"，反向过程退化成**先验样本到训练点的近邻检索**。

反之，若噪声充分覆盖，先验典型样本**同时**处在多个扰动核的支撑里，$w_k$ 是真正的概率分布而非 one-hot，反向轨迹能根据 softmax 概率"投票"到不同 mode——这才是反向 SDE 把 $\pi$ 拉成 $p_{\text{data}}$ 的工作机制。

## B3. 这是"生成"和"记忆"的分界

更深一层：扩散模型为什么能生成**新样本**（不在训练集里）？

答案就在 score field 的"软混合"区。在某个中间噪声尺度上，噪声点 $x$ 同时受多个 $x_k$ 拉力，score 是加权平均

$$
\nabla\log p_\sigma(x)=\frac{\sum_k w_k(x)(x_k-x)}{\sigma^2\sum_k w_k(x)}.
$$

反向 SDE 沿这个**插值的** score 走，最终落到的不是任何具体训练点，而是**邻近多个训练点的概率插值**——"新样本"。

前提是：在那个 $x$ 处，确实有多个训练点的扰动核共同覆盖。**这就是"噪声相互覆盖"的物理意义——它就是模型能插值/泛化的几何前提。**

反之，若训练时不同 $x_i$ 的噪声完全不重叠：

- 每个 $x_t$ 几乎只能由唯一的 $x_0$ 解释，回归目标退化成 $(x_0 - x_t)/\sigma^2$，没有混合项。
- 网络学到的 score field 是分段的：每个噪声区域指向唯一训练点。
- 反向采样从先验样本走到"它命中的那个分段对应的训练点"，输出是训练集的**复述**。

**"覆盖 $x_j$" 不是工程上的方便要求，而是扩散模型从"记忆机器"升级为"生成模型"的必要条件。**

## B4. 训练视角：边际 score vs 条件 score

可能的反驳："DSM 训练里我们只看每个 $x_0$ 的条件 score，跟其它 $x_j$ 无关。"

这对训练**目标的形式**是对的，但对训练**结果**是错的。DSM 损失

$$
\mathcal L=\mathbb E_{x_0,t,\varepsilon}\big\|s_\theta(x_t,t)-(-\varepsilon/\sigma_t)\big\|^2
$$

的最小值点是**边际** score：

$$
s^*(x_t,t)=\nabla\log p_t(x_t),\quad p_t(x_t)=\int p_{\text{data}}(x_0)\,p_t(x_t|x_0)\,dx_0.
$$

网络在某点 $x_t$ 拟合的最优答案是"对所有可能的源 $x_0$ 做后验加权的指回方向"——**所有训练点都参与**。但这种加权只有在 $x_t$ 同时落在多个 $p_t(\cdot|x_0)$ 的支撑中（即噪声相互覆盖的区域）时才**非平凡**。

若噪声不重叠：每个 $x_t$ 几乎只有唯一 $x_0$ 对它有非零条件密度，边际 score 退化成那个唯一 $x_0$ 的条件 score。网络学到的就是 B3 描述的分段函数。

## B5. 附录 B 一句话总结

**反向 SDE 的起点 $\pi$ 是单峰各向同性的，它不携带"我来自哪个 mode"的信息；score 网络必须在 $\pi$ 的典型支撑区域同时对多个训练点"敏感"，这种敏感性的几何前提就是各个 mode 的扰动核在那里相互覆盖。"只破坏 $x_i$"在固定先验下等价于"也覆盖 $x_j$"，两者是同一个条件——前者是 marginal $\approx$ prior，后者是这一条件的传递性推论。**
