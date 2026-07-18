# 反对称 Jacobian 与非守恒项 $r$ 的关系（回应 Gert 的反馈）

> 配套方法文档：[`score_nonconservative_check.md`](score_nonconservative_check.md)。
> 本文澄清诊断量 $\rho$（Jacobian 反对称分数）与 Helmholtz 分解里非守恒项 $r$ 的精确关系，
> 回应 Gert Aarts 提出的一个尖锐且正确的技术质疑。

Gert 的原话（三层意思）：

> "The Jacobian of the exact score is the Hessian of the log probability and hence
> symmetric, correct? I would suggest to make this all explicit and not hide that the
> score is non-conservative, just that the violation is small but quantifiable. I will
> have to think a bit more about this, though, since if the learned score is
> $s = \nabla\log p + r$, with $\nabla\cdot r = 0$ and hence $r = \nabla\times A$ (in 2D),
> it is not clear to me that the violating term is anti-symmetric."

---

## 第一层：exact score 的 Jacobian 是对称的（锚点）

精确 score $s^\star=\nabla\log p$，其 Jacobian

$$
J^\star_{ij}=\frac{\partial s^\star_i}{\partial x_j}
=\frac{\partial^2 \log p}{\partial x_j\,\partial x_i}
=\big[\mathrm{Hess}\,\log p\big]_{ij}
$$

是 $\log p$ 的 Hessian，混合偏导可交换 $\Rightarrow$ **对称**。这是整个诊断的地基：

$$
\boxed{\text{单连通域上：}\quad
\text{conservative}\iff \nabla\times s=0 \iff J[s]\ \text{对称}.}
$$

因此 Jacobian 的反对称部分

$$
A=\tfrac12\big(J-J^\top\big),\qquad A_{ij}=\tfrac12\big(\partial_j s_i-\partial_i s_j\big)
$$

是非守恒的**完整、精确**的障碍量：$A=0\iff s$ 守恒。

## 第二层：表述取向 —— 明说非守恒，量化其小

不要把问题框成"它守不守恒"，而要框成：

> 学到的 score **本来就是**非守恒的（已知事实，非缺陷）；我们在**定量这个很小的违背量**。

三参照结果正好是这个调子（L=128, k=0.28, sigma=640, epoch=10000，$t\to0$）：

| 参照 | $\rho$ | $\rho^2$ | 含义 |
|---|---|---|---|
| baseline $-\partial S/\partial\phi$ | $\sim4\times10^{-8}$ | $\sim10^{-15}$ | 守恒地板（数值噪声） |
| 训练后 score net | $\sim8\times10^{-3}$ | $\sim7\times10^{-5}$ | 实际残留（真、可测） |
| random init | $\sim0.70$ | $\sim0.50$ | 随机天花板 $1/\sqrt2$ |

既不掩饰非守恒为真（高出地板 5 个量级），又量化出它很小。

---

## 第三层（核心）：反对称 Jacobian 与 $r$ 的精确关系

Gert 的疑问：诊断（Jacobian 反对称）是否真的对应非守恒项 $r$？关键在**两类对象被混在一起**。

### 澄清一：$r$ 是向量，"反对称"是矩阵

- $r$ 是**向量场**，性质是无散度 $\nabla\cdot r=0$。"$r$ 反对称"是 type error——反对称是**矩阵**（Jacobian）的属性。
- 真正反对称的是 Jacobian 的反对称部分 $A[s]$。

所以 "the violating term is anti-symmetric" 这句话本身不成立——**这正是 Gert 觉得 not clear 的点**。正确的桥梁不是"$r$ 反对称"，而是下面的恒等式链。

### 澄清二：反对称 Jacobian $=\nabla\times s=\nabla\times r$（不是 $r$ 本身）

对 Helmholtz 分解 $s=\nabla\Phi+r,\ \nabla\cdot r=0$ 取 Jacobian：

$$
J[s]=\underbrace{\mathrm{Hess}\,\Phi}_{\text{对称}}+J[r]
\;\Longrightarrow\;
A[s]=\underbrace{A[\mathrm{Hess}\,\Phi]}_{=0}+A[r]=A[r].
$$

2D 里反对称部分只有一个独立分量，正好是 curl：

$$
A_{12}[s]=\tfrac12\big(\partial_y s_x-\partial_x s_y\big)=-\tfrac12\,(\nabla\times s)_z,
$$

$$
\nabla\times s=\nabla\times(\nabla\Phi)+\nabla\times r=\nabla\times r.
$$

再代入 $r=\nabla\times A=(\partial_y A,\,-\partial_x A)$（$A$ 为标量流函数）：

$$
\nabla\cdot r=\partial_x\partial_y A-\partial_y\partial_x A=0\ \checkmark,
\qquad
\nabla\times r=\partial_x(-\partial_x A)-\partial_y(\partial_y A)=-\nabla^2 A.
$$

于是

$$
\boxed{\;A[s]\ \text{的非零分量}\;=\;-\tfrac12\,(\nabla\times s)\;=\;-\tfrac12\,(\nabla\times r)\;=\;\tfrac12\,\nabla^2 A.\;}
$$

**$\rho$ 度量的是 $\nabla\times s=\nabla\times r$ —— 即 $r$ 的旋度（一个导数），不是 $r$ 本身。** 反对称
Jacobian 与 $r$ 之间差一个 curl 算子，不是恒等。这就是 Gert 没立刻看通的连接。

> 直接回答他那句话：你是对的，$r$ 本身不反对称；反对称的是 Jacobian 的旋度部分，
> 而 $\nabla\times s=\nabla\times r$ 把两者联系起来。

### 澄清三：谐函数歧义（$\rho$ 该看不到的，确实看不到）

$\rho$ 测 $\nabla\times r$ 而非 $r$，会不会有"非零 $r$ 但 $\rho=0$"？

会，当且仅当 $\nabla^2 A=0$（$A$ 调和）。此时 $r$ **既无散度又无旋度** $\Rightarrow$ 调和向量场。但
无旋场在单连通域上**就是梯度场** $r=\nabla\psi$（$\psi$ 调和）——即这种 $r$ **本身就守恒**，应并入
$\nabla\Phi$，根本不代表非守恒。

这正是论文第 7 页的 **harmonic ambiguity**（"adding a harmonic function $h$, $\nabla^2 h=0$,
without affecting the score"）。所以 $\rho$ 忽略它不是 bug，是 feature：$\rho$ 精确捕捉"无法写成
梯度"的那部分，谐函数那部分本就归守恒侧。

### 澄清四：两个必须明说的 caveat

1. **$\rho=0\iff$ 可积分（守恒），$\ne$ score 正确。** 即使 $\nabla\Phi\ne\nabla\log p$（守恒部分
   学错了），只要它仍是梯度，$\rho$ 就是 0。$\rho$ 量的是**非守恒性**，不是 **score 精度**——两件事分开讲。
2. **$\rho$ 测 $\nabla\times r$（导数），不是 $\|r\|$（场本身大小）。** 若要 field-level 的
   $\|r\|/\|s\|$（旋度部分占整场多大比例），需 Helmholtz 投影（解 Poisson）才拿得到；$\rho$ 是
   Jacobian-level 的旋度量。

### 对 Gert 写法的一个小提醒

$s=\nabla\log p+r$ 并**不保证** $\nabla\cdot r=0$：那要求 $\nabla\cdot s=\nabla^2\log p$，一般不成立。
干净做法是对**学到的 $s$ 自己**做 Helmholtz 分解 $s=\nabla\Phi+r$（这总能做），此时 $r$ 是 $s$ 的
无散度部分，$\nabla\times s=\nabla\times r$ 严格成立。不论怎么拆，**反对称 Jacobian $=\nabla\times s$
这条结论不变**。

---

## 如何在正式表述里 "make explicit"

1. **写出定理**：单连通域上 conservative $\iff \nabla\times s=0 \iff$ Jacobian 对称；exact score 的
   Jacobian $=\mathrm{Hess}\log p$ 对称，是零点参照。
2. **明确 $\rho$ 的定义**：

$$
\rho=\frac{\lVert J-J^\top\rVert_F}{2\lVert J\rVert_F}=\text{归一化的}\ \lVert\nabla\times s\rVert,
$$

   并强调它 $=\nabla\times r$，不是 $r$。
3. **独立佐证（直接回应他的疑虑）**：闭合回路环量 $\oint s\cdot dl$（Stokes $\Rightarrow$ 等于 curl 通量），
   是 field/integral-level 的非守恒度量。它与 $\rho$ 结论一致，就从两个独立角度证实"反对称 Jacobian
   确实抓住非守恒"。
4. **（可选）补 field-level $\|r\|/\|s\|$**：解 $\nabla^2 A=-(\nabla\times s)_z$（论文 Eq. 3.13）得流函数
   $A$、再得 $r=\nabla\times A$，报 $\|r\|/\|s\|$。这正好给出 Gert 写法里 $r$ 的大小，和 $\rho$（它的 curl）互补。

---

## 一页纸总结

- exact score：$J=\mathrm{Hess}\log p$ 对称 $\Rightarrow$ 非守恒 $\iff$ Jacobian 非对称。
- 学到的 score：$A[s]=\nabla\times s=\nabla\times r=-\tfrac12\nabla^2 A$。**$r$ 不反对称；反对称的是
  它的 curl（= 反对称 Jacobian）。**
- $\rho$ 忠实度量"非守恒 = 不可积分 = 路径依赖"（$=\nabla\times s$），但**不**度量 $\|r\|$，也**不**度量
  守恒部分的 score 误差；谐函数歧义被正确忽略。
- 结论：score 非守恒为真、可量化、且很小（$\rho\sim10^{-3}$ vs 随机 $0.7$、守恒 $10^{-8}$）。
