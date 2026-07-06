要解的方程
目标是让 PIK 输运方程逐点成立(而不是像 weak 那样在 $p_t$ 加权下成立):

$$\partial_\phi v-v,\partial_\phi\hat S_t=\partial_t\hat S_t-\langle\partial_t\hat S_t\rangle_{p_t},\qquad v=\partial_\phi s_\theta.$$

右边那个 $\langle\partial_t\hat S_t\rangle_{p_t}$ 是唯一需要 $p_t$ 的项(一个只依赖 $t$ 的场无关常数)。整版做法的核心就是把它绕过去。

第 1 步:网络与参数化(和 weak 完全相同)
直接 import action_matching_0d 的 SNet:学一个偶标量势 $s_\theta(\phi^2,t)$(四层 MLP,输入是 $(\phi/L)^2$ 和 $t$,并做了 $s_\theta(0,t)=0$ 的 gauge 固定)。速度取 $v=\partial_\phi s_\theta$,自动是奇函数,且 $v(0,t)=0$ 精确成立。用同一个网络是刻意的——这样 strong vs weak 的差别只来自损失,排除架构干扰。

第 2 步:构造残差,用参考差分消掉 $p_t$ 期望
在 refsub_residual 里,先定义未减常数的量:

$$g_\theta(\phi,t)=\partial_\phi v-v,\partial_\phi\hat S_t-\partial_t\hat S_t.$$

注意我只减了逐点解析的 $\partial_t\hat S_t$,没减那个期望常数。所以 $g_\theta(\phi,t)=[\text{真残差}]+\langle\partial_t\hat S_t\rangle_{p_t}$——它比真残差多了一个场无关常数。然后取

$$r_{\rm ref}(\phi,t)=g_\theta(\phi,t)-g_\theta(0,t).$$

参考点 $\chi=0$ 处 $g_\theta(0,t)=\partial_\phi v(0,t)$(因为 $v(0)$、$\partial_\phi\hat S_t(0)$、$\partial_t\hat S_t(0)$ 都精确为零)。相减时那个常数被消掉——这就是论文 (20) 的差分技巧,也是"零采样"的全部秘密。精确解让 $g_\theta(\phi,t)$ 对所有 $\phi$ 等于同一个常数,于是 $r_{\rm ref}\equiv0$。

代码里 g[CI] 就是 $g_\theta(0,t)$,CI = N//2 是网格上 $\phi=0$ 的那个点(网格对称、$N$ 为奇数,所以那点精确是 0)。

第 3 步:残差里的导数怎么算
$\partial_\phi v=\partial_\phi^2 s_\theta$ 需要对网络求二阶场导,用两次 autograd:


v  = grad(s.sum(), x, create_graph=True)   # ∂_φ s
dv = grad(v.sum(), x, create_graph=True)   # ∂_φ² s = ∂_φ v
因为 MLP 是 pointwise 的(每个 $\phi_i$ 独立过网络),Jacobian 对角,grad(sum) 正好给出逐点导数。$\partial_\phi\hat S_t$ 和 $\partial_t\hat S_t$ 是解析式(dS_dphi、A.dS_dt),直接代入。这里的"多一阶导"正是 strong form 相对 weak 的代价:0D 里就是标量二阶导,极便宜;到高维它会变成散度 $\nabla!\cdot v$,这才是 scaling 痛点。

第 4 步:配点——完全不含 $p_t$
这是和 weak 最本质的区别。权重是固定的均匀网格测度,只限制在 $|\phi|\le L_c=5$:


u = (w * (phi.abs() <= Lc));  u = u / u.sum()
w 是梯形积分权重(等距 $\mathrm{d}\phi$),里面没有 $e^{-\hat S_t}$、没有任何目标分布信息、不随 $t$ 变。限制到 $[-5,5]$ 是为了不让 $\phi^4$ 尾部(残差 $\sim\phi^4$)主导损失,同时覆盖双阱峰 $\pm2$。这就诚实地演示了"配点可以随便放,不需要从 $p_t$ 采样"。

第 5 步:损失与优化
每个优化步随机抽 4 个时间片,各自在全 $\phi$ 网格上算 $r_{\rm ref}$,均匀加权平方和后平均:

$$\mathcal L=\frac1{4}\sum_{t\in{t_a}}\sum_i u_i,r_{\rm ref}(\phi_i,t)^2.$$

普通 Adam(lr=1e-3,你要求去掉的 cosine 已删),梯度裁剪到 1.0,20000 步。优化对含 $\partial_\phi^2 s$ 的损失再求 $\theta$ 梯度,是三重反传,0D 里可忽略。

第 6 步:验证(复用 weak 的,保证可比)
A.validate 对几个 $t$ 比 $v_\theta$ 与精确 $v^{\rm exact}$ 的 $p_t$-加权 RMSE(评估用 $p_t$ 是合法的,训练才不碰);A.transport_test 从 $t=1$ 的 Gauss 跑反向 ODE 生成样本,比 $\langle\phi^2\rangle$、$\langle\phi^4\rangle$。

