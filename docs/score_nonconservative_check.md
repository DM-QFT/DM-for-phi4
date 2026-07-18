# 验证 score net 的非守恒残留

> 参考：Aarts, Habibi, Wang, Zhou, *Combining complex Langevin dynamics with
> score-based and energy-based diffusion models*, arXiv:2510.01328v1.
>
> 目标：定量回答"当前训练好的 DM score net 里还剩多少**非守恒**成分"。

---

## 0. 一个对实现有利的事实

本仓库里 `model(x, t)` 返回的就是反向 SDE 实际用的 score 估计：见
[networks.py:106](../networks.py#L106)，`forward` 最后

$$
s_\theta(x,t) \;=\; \frac{\text{net\_raw}(x,t)}{\operatorname{marginal\_prob\_std}(t)}.
$$

`marginal_prob_std(t)` 只依赖 $t$、不依赖 $x$，是个标量。因此它在下面定义的
"非守恒分数"里会被约掉 —— 指标对这个归一化是**尺度不变**的，不用担心用原始输出
还是用除过的 score。

---

## 1. 判据：什么叫"非守恒"

真正的 score $\nabla\log p_t$ 一定是某标量场的梯度（保守场）。神经网络直接回归向量场
$s_\theta$，**没有结构保证它是梯度**。论文 Eq. (3.8) 做 Helmholtz 分解：

$$
s_\theta(x,t) \;=\; \underbrace{\nabla \Phi_\theta(x,t)}_{\text{守恒 / 梯度部分}}
\;+\; \underbrace{r_\theta(x,t)}_{\text{非守恒残留}},
\qquad \nabla \cdot r_\theta = 0 .
$$

训练目标 Eq. (3.9) 表明守恒与非守恒两部分是**各自独立被最小化**的，所以
$r_\theta$ 不会自动归零 —— 它就是"训练后残留的非梯度成分"。

判断 $r_\theta = 0$ 有三个**等价**判据，对应三种测法：

| 判据 | 表达式 | 性质 |
|------|--------|------|
| Jacobian 对称 | $J_{ij}=\partial s_i/\partial x_j$ 对称 | 局部 |
| 无旋 (curl) | $\partial_x s_y - \partial_y s_x = 0$（2D） | 局部 |
| 闭合回路环量 | $\oint s\cdot dl = 0$ | 全局 |

其中 Jacobian 的反对称部分

$$
A_{ij} \;=\; \tfrac12\!\left(\partial_j s_i - \partial_i s_j\right)
$$

度量非守恒。2 维里 $\operatorname{curl} = \partial_x s_y - \partial_y s_x = 2A_{yx}$，
即论文 Fig. 3 画的那张图就是 $A$ 的唯一独立分量。

---

## 2. 关键提醒：维度

论文是 **1 个自由度** 复化成 $(x,y)$ 两维，所以能画一张 curl 热图。

你的对象维度是

$$
d = L^2 \;(\text{2Dphi4}),\quad L^3 \;(\text{3Dphi4}),\quad C\cdot H\cdot W \;(\text{图像}),
$$

反对称分量有 $d(d-1)/2$ 个 —— 画不出单张 curl 图。

**结论：你需要的不是逐分量可视化，而是把 Fig. 3 的 curl 聚合成一个标量指标。**

---

## 3. 主方法：Jacobian 反对称分数 ρ

定义无量纲、落在 $[0,1]$ 的"非守恒占比"：

$$
\boxed{\;\rho^2 \;=\; \frac{\lVert J - J^\top \rVert_F^2}{4\,\lVert J \rVert_F^2}
\;=\; \frac{\lVert A \rVert_F^2}{\lVert J \rVert_F^2} \;\in[0,1]\;}
$$

- 物理含义：**非守恒能量 / 总场强能量**。
- $\rho = 0$ ⟺ 完全守恒（真梯度场）。
- 分子 $\lVert J - J^\top \rVert_F^2 = 4\lVert A\rVert_F^2$ 就是论文 curl 在**全部** $d(d-1)/2$
  个分量上的总和 —— 即 Fig. 3 的高维聚合版本。

恒等式（用于推导）：对 $J = S + A$（$S$ 对称、$A$ 反对称，$\langle S,A\rangle = 0$）：

$$
\lVert J - J^\top \rVert_F^2 = 4\lVert A\rVert_F^2,\qquad
\lVert J + J^\top \rVert_F^2 = 4\lVert S\rVert_F^2,\qquad
\text{两者之和} = 4\lVert J\rVert_F^2 .
$$

### 3.1 高维怎么估：Hutchinson + 自动微分

不显式构造 $d\times d$ 的 $J$。对各分量独立、协方差为 $I$ 的随机探针 $v$（高斯或 Rademacher）：

$$
\mathbb{E}_v \lVert M v \rVert^2 = \operatorname{tr}(M^\top M) = \lVert M \rVert_F^2 .
$$

取 $M = J - J^\top$ 得分子，取 $M = J$ 得分母。每个探针只需两次自动微分：

**(a) $J^\top v$** —— 一次反向（VJP，就是普通 backprop）。设 $J_{ij}=\partial s_i/\partial x_j$：

$$
\big[\,\texttt{autograd.grad}(s, x, \texttt{grad\_outputs}{=}v)\,\big]_j
= \sum_i v_i\,\frac{\partial s_i}{\partial x_j} = (J^\top v)_j .
$$

**(b) $Jv$** —— double-backprop 技巧（纯反向模式，对 NCSN++ 最稳）：

$$
g(w) = J^\top w \;\xrightarrow{\ \partial/\partial w\ \text{沿}\ v\ }\;
\big[\texttt{autograd.grad}(g, w, \texttt{grad\_outputs}{=}v)\big]_k
= \sum_j v_j\,J_{kj} = (J v)_k .
$$

然后

$$
(J - J^\top)\,v = Jv - J^\top v .
$$

### 3.2 参考实现

```python
import torch

def _perturb_to_t(x0, t, model):
    """把干净组态 x0 前向扰动到噪声层级 t (VE-SDE: x_t = x0 + std(t)·z),
       从而在真实 marginal p_t 上评估 score, 而非只在干净数据点。"""
    std = model.marginal_prob_std_fn(t).view(-1, *([1] * (x0.dim() - 1)))
    return x0 + std * torch.randn_like(x0)

@torch.enable_grad()
def nonconservative_fraction(model, x0_batch, t_value, n_hutch=8, on_pt=True):
    """返回每个样本的 rho^2 = ||J - J^T||_F^2 / (4||J||_F^2) ∈ [0,1].
       rho=0 完全守恒(真梯度场); 越大非守恒残留越多。
       论文 Fig.3 curl 的高维聚合版, 尺度不变(与 /std(t) 归一化无关)。"""
    model.eval()
    device = next(model.parameters()).device
    x0_batch = x0_batch.to(device)
    B = x0_batch.shape[0]
    t = torch.full((B,), float(t_value), device=device)

    x = (_perturb_to_t(x0_batch, t, model) if on_pt else x0_batch).requires_grad_(True)
    red = tuple(range(1, x.dim()))             # 对所有场自由度求和, 保留 batch 维

    num = torch.zeros(B, device=device)        # 累加 ||(J - J^T) v||^2
    den = torch.zeros(B, device=device)        # 累加 ||J v||^2
    for _ in range(n_hutch):
        v = torch.randn_like(x)                # 高斯探针, Cov = I
        s = model(x, t)                        # score, 形状同 x
        JTv = torch.autograd.grad(s, x, grad_outputs=v, retain_graph=True)[0]   # J^T v
        w = torch.zeros_like(s, requires_grad=True)
        g = torch.autograd.grad(s, x, grad_outputs=w, create_graph=True)[0]     # = J^T w
        Jv = torch.autograd.grad(g, w, grad_outputs=v)[0]                       # = J v
        diff = Jv - JTv                        # = (J - J^T) v
        num += (diff ** 2).sum(dim=red)
        den += (Jv ** 2).sum(dim=red)
    return (num / (4.0 * den)).detach().cpu()  # per-sample rho^2
```

用法：

```python
model = DiffusionModel.load_from_checkpoint(ckpt_path, score_model=net).to('cuda').eval()
x0 = load_real_or_sampled_configs(...)         # [B, 1, L, L], 归一化到 [-1,1] 空间

with model.ema.average_parameters():           # 采样时用的就是 EMA 权重
    for t_val in [1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 1.0]:
        rho2 = nonconservative_fraction(model, x0, t_val, n_hutch=16)
        print(f"t={t_val:6.3f}  rho={rho2.mean().sqrt():.4f}  (±{rho2.std().sqrt():.4f})")
```

---

## 4. 独立交叉验证：闭合回路环量

ρ 是**局部**判据（逐点 Jacobian）。再补一个**全局**判据互证（对应 Eq. 3.10 的路径无关性）：

在配置空间随机取一个 2D 平面 $\{u, w\}$（正交单位向量），画一个半径 $r\sim\operatorname{std}(t)$
的小圆并数值积分：

$$
x(\theta) = c + r\big(\cos\theta\, u + \sin\theta\, w\big),\qquad
\text{环量} = \oint s\cdot dl,\qquad
\text{归一化} = \frac{\big|\oint s\cdot dl\big|}{\oint \lVert s\rVert\, dl}.
$$

保守场环量恒为零，非零即路径依赖（= 非守恒）。两个判据结论一致才放心。

```python
@torch.no_grad()
def loop_circulation(model, center, t_value, radius=0.1, n_steps=128, n_loops=32):
    model.eval()
    device = next(model.parameters()).device
    c = center.to(device).flatten(); d = c.numel()
    out = []
    for _ in range(n_loops):
        u, w = torch.randn(2, d, device=device)
        u = u / u.norm()
        w = w - (w @ u) * u; w = w / w.norm()          # 平面内正交基
        th = torch.linspace(0, 2 * torch.pi, n_steps + 1, device=device)[:-1]
        pts = c[None] + radius * (torch.cos(th)[:, None] * u + torch.sin(th)[:, None] * w)
        dl  = radius * (-torch.sin(th)[:, None] * u + torch.cos(th)[:, None] * w) * (2 * torch.pi / n_steps)
        t = torch.full((n_steps,), float(t_value), device=device)
        s = model(pts.view(n_steps, *center.shape[1:]), t).flatten(1)
        circ = (s * dl).sum(dim=1).sum()                # ∮ s·dl
        norm = (s.norm(dim=1) * dl.norm(dim=1)).sum()   # ∮ |s| dl
        out.append((circ.abs() / (norm + 1e-12)).item())
    return torch.tensor(out)
```

> `pts.view(...)` 的形状需按你具体 `x` 的形状微调。

---

## 4.5. 更简单的等价测法（cosine similarity 风格）

主方法 ρ 与下面三个测法**同源** —— 都在测 Jacobian 的反对称部分
$A=\tfrac12(J-J^\top)$，只是表达和成本不同。核心观察：

$$
\text{守恒} \iff J = J^\top \iff A = 0,
$$

所以"守恒性"本质就是"$J$ 和 $J^\top$ 有多像"，这天然是一个 cosine。

### 4.5.1 矩阵级 cosine（= ρ 换皮）

把 $J,\,J^\top$ 当向量做 Frobenius cosine：

$$
\cos\angle(J,J^\top)
= \frac{\langle J,J^\top\rangle_F}{\lVert J\rVert_F\,\lVert J^\top\rVert_F}
= \frac{\sum_{ij}J_{ij}J_{ji}}{\lVert J\rVert_F^2}.
$$

代入 $J = S + A$ 可精确得到

$$
\boxed{\;\cos\angle(J,J^\top)
= \frac{\lVert S\rVert_F^2 - \lVert A\rVert_F^2}{\lVert S\rVert_F^2 + \lVert A\rVert_F^2}
= 1 - 2\rho^2\;}
$$

即 $\cos = 1$ ⟺ 完全守恒，$\rho^2 = (1-\cos)/2$。好处是读数直观（1 = 守恒），
坏处是**不比 ρ 便宜**，仍需 Jacobian 探针。

### 4.5.2 向量级 cosine（最贴近 "cosine similarity"）

对随机探针 $v$ 比较 $Jv$ 与 $J^\top v$ 的方向：

$$
\cos\big(Jv,\,J^\top v\big)
= \frac{(Jv)\cdot(J^\top v)}{\lVert Jv\rVert\,\lVert J^\top v\rVert}.
$$

因为 $J$ 对称 ⟺ 对**所有** $v$ 都有 $Jv = J^\top v$，守恒时这个 cosine 恒为 1。

```python
v = torch.randn_like(x)
s = model(x, t)
JTv = torch.autograd.grad(s, x, v, retain_graph=True)[0]            # J^T v  (反向/VJP)
Jv  = torch.func.jvp(lambda z: model(z, t), (x,), (v,))[1]          # J v    (前向/JVP)
cos = torch.cosine_similarity(Jv.flatten(1), JTv.flatten(1), dim=1) # 接近 1 = 守恒
```

> 注意：cosine 只比**方向**、不比模长。$Jv$ 与 $J^\top v$ 方向一致但模长不同时仍可能非对称。
> 要同时约束方向和模长，用相对差 $\lVert Jv - J^\top v\rVert / \lVert Jv + J^\top v\rVert$
> —— 那就退回 ρ。所以 cosine 是"更宽松但更直观"的版本。

### 4.5.3 有限差分对称性检验（**完全不用 autodiff，最省最稳**）

守恒 ⟺ 混合偏导对称（$u^\top J v = v^\top J u$）。只做**前向 forward**、有限差分：
取随机方向 $u, v$ 与基点 $x$，

$$
a = u\cdot\big[s(x+\epsilon v) - s(x-\epsilon v)\big] \approx 2\epsilon\,u^\top J v,\qquad
b = v\cdot\big[s(x+\epsilon u) - s(x-\epsilon u)\big] \approx 2\epsilon\,v^\top J u.
$$

守恒时 $a = b$。归一化非守恒分数（对 $u,v,x$ 平均，$2\epsilon$ 约掉）：

$$
\frac{\mathbb{E}\big[(a-b)^2\big]}{\mathbb{E}\big[a^2+b^2\big]} = 2\rho^2,
\qquad\Longrightarrow\qquad
\rho^2 \approx \frac12\,\frac{\mathbb{E}[(a-b)^2]}{\mathbb{E}[a^2+b^2]}.
$$

```python
eps = 1e-3
u, v = torch.randn_like(x), torch.randn_like(x)
sdot_v = model(x + eps*v, t) - model(x - eps*v, t)
sdot_u = model(x + eps*u, t) - model(x - eps*u, t)
a = (u * sdot_v).flatten(1).sum(1)     # ∝ uᵀ J v
b = (v * sdot_u).flatten(1).sum(1)     # ∝ vᵀ J u
rho2 = 0.5 * (a - b).pow(2).mean() / (a.pow(2) + b.pow(2)).mean()
```

额外好处：

- **只用 4 次前向**，无反向、无二阶图 —— 最省显存、最快。
- **对网络可微性不敏感**：score-based net 用 LeakyReLU（见 §A / 论文 Table 3），二阶导在折点
  病态，double-backprop 会踩坑；有限差分不在乎。
- 唯一要调 $\epsilon$：归一化 $[-1,1]$ 空间取 $\sim 10^{-3}$，太小被浮点噪声淹没、太大线性化失效，
  扫两三个值看 plateau。

### 4.5.4 选哪个

| 方法 | 成本 | 直观度 | 备注 |
|------|------|--------|------|
| ρ（Frobenius，§3） | 1 JVP + 1 VJP | 中（占比） | 主指标 |
| $\cos(Jv,J^\top v)$ | 1 JVP + 1 VJP | **高**（1=守恒） | 只比方向，最像 cos similarity |
| 有限差分对称性 | **4× forward** | 中 | **最便宜最稳**，LeakyReLU 也安全 |

三者同源，换算关系 $\cos = 1 - 2\rho^2$。**要快要稳用有限差分，要一眼读懂用向量级 cosine。**

---

## 5. 评估协议

1. **在 $p_t$ 上评估**，不是只在干净数据点：用 `_perturb_to_t` 把真实组态扰动到层级 $t$
   再测。非守恒性在数据流形外（训练不足处）通常更大（论文第 7 页：边界处 score 没训好）。
   $t\to 0$ 时 $p_t\approx$ 数据分布，直接用真实组态即可。
2. **扫 $t$（log scale）**，报 $\rho(t)$ 曲线。物理上最关心 $t\to 0$（Eq. 3.17 的
   $\lim_{t\to 0}$，即真实分布的 score）。
3. **多 seed**（论文 $N=10$）：非守恒性部分是训练随机性产物。报 mean±std；严谨的话照搬
   附录 D 的 DerSimonian–Laird，分离 statistical / systematic 误差。
4. **sanity check（必做）**：先在解析已知守恒场上验证代码 —— 纯高斯 target 的
   $s(x,t) = -x/\operatorname{std}(t)^2$，Jacobian 对称，此时 $\rho\approx 0$
   （只剩 Hutchinson 噪声，增大 `n_hutch` 压低）。代码自检通过后真实模型的数才可信。
5. **可选、几乎免费**：你有 log-scale checkpoints，把 $\rho$ 当 epoch 的函数扫一遍，看
   非守恒残留是否随训练收缩 —— 可直接挂进 propagator 演化研究线。

---

## 6. 怎么解读结果

| $\rho$ 量级 | 结论 |
|------------|------|
| $\sim 10^{-2}$ | score 基本守恒，残留可忽略 |
| $\gtrsim 0.1$ | 像论文那样，**score 不能直接积分还原分布**；要分布只能靠采样直方图，或改用 energy-based 参数化 $s=-\nabla E_\theta$ 从结构上强制守恒 |

$\rho^2$ 直接读作"非守恒能量占总场强能量的比例"。
