这个 objective（eq:weak_loss）

$$\mathcal{L}{\mathrm{weak}}[s]=\int_0^1!\dd t,\Big\langle\tfrac12(\partial\phi s)^2+s,\underbrace{\big(\partial_t\hat S_t-\langle\partial_t\hat S_t\rangle_{p_t}\big)}{\equiv,f_t;=;\partial_tS_t;=;-\partial_t\log p_t}\Big\rangle{p_t}$$

不是拍脑袋设计的，它有一条非常干净的推导线。我按三个层次讲：直接构造、完全平方验证、以及它和 Neklyudov Action Matching 的关系。

1. 出发点：要解的方程是散度形式的椭圆方程
我们要找的是梯度形式的输运速度 $v=\partial_\phi s$，使连续性方程成立。把 $\partial_t p_t=-p_tf_t$ 代入连续性方程，待解方程就是

$$\partial_\phi\big(p_t,\partial_\phi s\big)=p_t f_t .$$

这是一个加权 Poisson 方程（$\nabla!\cdot(a\nabla u)=g$ 型，权重 $a=p_t$）。而这类散度形式方程有一个教科书级的事实——Dirichlet 原理：它恰好是凸泛函

$$J_t[s]=\int\dd\phi,p_t\Big[\tfrac12(\partial_\phi s)^2+s,f_t\Big]$$

的 Euler–Lagrange 方程（静电学类比：$\nabla!\cdot(\varepsilon\nabla\varphi)=-\rho$ 极小化 $\int[\frac\varepsilon2|\nabla\varphi|^2-\rho\varphi]$，这里 $p_t$ 扮演介电常数）。对 $t$ 积分就得到 $\mathcal{L}_{\mathrm{weak}}$。你上次让我修的那个符号（eq:weak_euler_lagrange）正是这一步的变分结果。

所以第一层答案是：**动能项 $\frac12(\partial_\phi s)^2$ + 线性源项 $s f_t$ 是让"求导取零"精确复现目标 PDE 的标准配方。**动能项单独会把 $v$ 压成零，源项负责把它"推"到恰好实现所要求的概率变化率 $f_t=-\partial_t\log p_t$。

2. 更直接的推导：从误差泛函完全平方展开
更有启发性的是反过来推。理想的损失当然是与精确速度的 $L^2(p_t)$ 距离

$$\mathcal{D}[s]=\tfrac12\int_0^1!\dd t,\big\langle(\partial_\phi s-v_t^{\mathrm{exact}})^2\big\rangle_{p_t},$$

但它看似不可用——$v^{\mathrm{exact}}$ 未知。展开：

$$\mathcal{D}[s]=\int\dd t\Big[\tfrac12\langle(\partial_\phi s)^2\rangle-\big\langle \partial_\phi s; v^{\mathrm{exact}}\big\rangle\Big]+\underbrace{\tfrac12\int\dd t,\langle (v^{\mathrm{exact}})^2\rangle}_{\text{与 }s\text{ 无关}}.$$

关键在交叉项：$v^{\mathrm{exact}}$ 只线性地出现，而且只通过它唯一已知的性质——满足 $\partial_\phi(p_tv^{\mathrm{exact}})=p_tf_t$——进入。分部积分（用零通量条件 eq:zero_flux 丢掉边界项）：

$$\big\langle\partial_\phi s,v^{\mathrm{exact}}\big\rangle_{p_t}=\int\dd\phi;\partial_\phi s,(p_tv^{\mathrm{exact}})=-\int\dd\phi;s,\partial_\phi(p_tv^{\mathrm{exact}})=-\int\dd\phi;s,p_tf_t=-\langle s,f_t\rangle_{p_t}.$$

未知量完全消失了，只剩已知的 $f_t$。代回去：

$$\boxed{;\mathcal{D}[s]=\mathcal{L}{\mathrm{weak}}[s]+\tfrac12\int_0^1!\dd t,\langle(v^{\mathrm{exact}})^2\rangle{p_t};}$$

即 $\mathcal{L}_{\mathrm{weak}}$ 与真实误差 $\mathcal{D}$ 只差一个常数。这一步同时白送了三个结论：

eq:weak_lower_bound（损失差 = $L^2(p_t)$ 速度误差）就是这个恒等式本身；
最优值 $\mathcal{L}_{\mathrm{weak}}^\star=-\frac12\int\langle (v^{\mathrm{exact}})^2\rangle\dd t<0$——所以训练时 loss 收敛到一个负数（action_matching_0d.py 里打印的 loss 正是负的），其绝对值就是输运动能；
均值扣除 $\langle\partial_t\hat S_t\rangle$ 的必要性：它保证 $\langle f_t\rangle_{p_t}=0$，于是规范方向 $s\mapsto s+c(t)$（$\partial_\phi s$ 不变）对损失零贡献。若不扣除，损失沿这个方向是线性无界的，$\mathcal{L}\to-\infty$——这正是我们在代码里实际踩过的坑（naive AM 发散，代码注释里那句 "would otherwise drive the AM loss to -inf"）。
3. 和 Neklyudov Action Matching 的关系："时间边界项塌缩"
Neklyudov et al. (2023) 的一般 AM 面对的是只有样本、不知道 $p_t$ 表达式的情形。他们做同样的交叉项消去，但由于 $\partial_t\log p_t$ 未知，必须再对 $t$ 做一次分部积分，把 $\int\dd t,\langle s,\partial_t\log p_t\rangle$ 换成时间端点的边界项减去 $\langle\partial_t s\rangle$：

$$\mathcal{L}{\mathrm{AM}}[s]=\E{p_0}[s_0]-\E_{p_1}[s_1]+\int_0^1!\dd t;\E_{p_t}\Big[\tfrac12(\partial_\phi s)^2+\partial_t s\Big].$$

这只需要样本就能估计——这是他们的核心贡献。而在 PIK 设定里路径是解析规定的，$\partial_t\log p_t=-f_t$ 闭式已知，这次时间分部积分就没有必要做（或者说做回去）：用 $\frac{\dd}{\dd t}\E_{p_t}[s_t]=\E[\partial_t s]+\E[s,\partial_t\log p_t]$ 代入上式，$\E_{p_0}[s_0]-\E_{p_1}[s_1]$ 与全导数项精确相消，剩下的就是我们的单一源项形式。这就是 tex 里那句 "the time-boundary terms of the general construction collapse to the single source term" 的确切含义。

对我们来说这个"塌缩"不只是美观：去掉 $\partial_t s$ 和端点期望后，(a) 不再需要对 $t$ 求导（少一次微分）；(b) 消除了端点项与内部时间离散化之间的失配（另一个实际踩过的坑）；(c) 每个 $t$ 切片的损失独立、可任意采样时间。

一句话总结：这个 objective 是"想最小化与未知精确速度的 $L^2(p_t)$ 距离 → 未知量只线性出现且满足已知的散度方程 → 一次分部积分把它整体消掉"的产物；它是 Dirichlet 原理在权重 $p_t$ 下的实例，也是 Action Matching 在解析路径下的特化——三种视角得到同一个泛函。

