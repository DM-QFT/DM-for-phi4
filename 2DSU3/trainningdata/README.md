# 二维 SU(3) 纯规范理论热浴采样

`su3_2d_heatbath.jl` 在 $L\times L$ 周期格子上生成二维 SU(3) 纯规范场（Wilson 作用量）的平衡构型，写入 HDF5，作为扩散模型的训练数据。二维纯规范理论可以精确求解，所以同一个程序也是采样算法的严格检验场。本文先讲物理和算法，再对照代码，最后记录验证结果。

## 1. 模型

链变量 $U_\mu(x)\in SU(3)$，$\mu=1,2$，$x=(x_1,x_2)$，$x_i\in\{1,\dots,L\}$，周期边界。plaquette

$$P(x)=U_1(x)\,U_2(x+\hat 1)\,U_1(x+\hat 2)^\dagger\,U_2(x)^\dagger .$$

Wilson 作用量与 Boltzmann 权重

$$S[U]=\beta\sum_x\Big[1-\tfrac13\,\mathrm{Re}\,\mathrm{Tr}\,P(x)\Big],\qquad
\mathrm d\mu[U]=\prod_{x,\mu}\mathrm dU_\mu(x)\;e^{-S[U]} .$$

$\mathrm dU$ 是 SU(3) 的 Haar 测度。$\beta=2N/g^2=6/g^2$。连续极限对应 $\beta\to\infty$。程序测量的观测量是平均 plaquette

$$\langle P\rangle=\Big\langle\frac1{L^2}\sum_x\tfrac13\,\mathrm{Re}\,\mathrm{Tr}\,P(x)\Big\rangle .$$

二维里每条链只出现在两个 plaquette 中，plaquette 之间几乎独立，这是精确可解的根源，也是热浴在这里退相关极快的原因。

## 2. 单链条件分布与 staple

固定所有其他链，只看 $U\equiv U_\mu(x)$。它出现在 $P(x)$ 和 $P(x-\hat\nu)$ 两个 plaquette 里（$\nu\neq\mu$）。作用量中依赖 $U$ 的部分可以写成

$$S\supset-\frac\beta3\,\mathrm{Re}\,\mathrm{Tr}\,(U\Sigma),\qquad \Sigma=\Sigma_1+\Sigma_2 ,$$

$$\Sigma_1=U_\nu(x+\hat\mu)\,U_\mu(x+\hat\nu)^\dagger\,U_\nu(x)^\dagger,\qquad
\Sigma_2=U_\nu(x+\hat\mu-\hat\nu)^\dagger\,U_\mu(x-\hat\nu)^\dagger\,U_\nu(x-\hat\nu).$$

$\Sigma_1$ 直接来自 $P(x)=U\Sigma_1$。$\Sigma_2$ 来自 $P(x-\hat\nu)$，它含的是 $U^\dagger$，利用 $\mathrm{Re}\,\mathrm{Tr}\,P=\mathrm{Re}\,\mathrm{Tr}\,P^\dagger$ 和迹的循环性把 $U$ 挪到最前面即得。$\Sigma$ 不是 SU(3) 元素，只是两个 SU(3) 矩阵之和。

于是单链的条件分布是

$$p(U)\,\mathrm dU\propto\exp\Big[\frac\beta3\,\mathrm{Re}\,\mathrm{Tr}\,(U\Sigma)\Big]\mathrm dU .$$

热浴算法就是直接从这个分布抽一个新的 $U$，与旧值无关。SU(3) 上没有直接抽样的办法，下面两节解决这个问题。

## 3. Cabibbo–Marinari 子群分解

思路是把 SU(3) 的更新拆成三个 SU(2) 子群的更新。取 $a\in SU(2)$，把它嵌入到 SU(3) 的第 $(i,j)$ 行列，其余对角元为 1，记为 $A=\mathrm{embed}(a,i,j)$。更新 $U\leftarrow AU$。权重对 $a$ 的依赖是

$$\mathrm{Re}\,\mathrm{Tr}\,(AU\Sigma)=\mathrm{Re}\,\mathrm{Tr}\,(AW),\qquad W=U\Sigma .$$

$A$ 只在 $(i,j)$ 块非平凡，所以只有 $W$ 的 $2\times2$ 子块 $W_{\rm sub}$ 参与，其余对角元贡献常数。

任意 $2\times2$ 复矩阵都能唯一分成 $M=Q+Q_\perp$。$Q=q_0\mathbb 1+i\,\vec q\cdot\vec\sigma$（$q$ 实）是实四元数，构成 $\mathbb R\cdot SU(2)$ 这个四维实子空间。$Q_\perp$ 在内积 $\mathrm{Re}\,\mathrm{Tr}(A^\dagger B)$ 下与全部四元数正交。$a\in SU(2)$ 自身也是四元数，因此 $\mathrm{Re}\,\mathrm{Tr}\,(aQ_\perp)=0$，

$$\mathrm{Re}\,\mathrm{Tr}\,(aW_{\rm sub})=\mathrm{Re}\,\mathrm{Tr}\,(aQ)=k\,\mathrm{Re}\,\mathrm{Tr}\,(aV_0),\qquad k=\sqrt{\det Q},\quad V_0=Q/k\in SU(2).$$

投影的分量形式就是代码里的 `su2_project`

$$\alpha=\tfrac12\big(W_{ii}+\overline{W_{jj}}\big),\qquad \beta=\tfrac12\big(W_{ij}-\overline{W_{ji}}\big),\qquad
Q=\begin{pmatrix}\alpha&\beta\\-\bar\beta&\bar\alpha\end{pmatrix},\qquad k=\sqrt{|\alpha|^2+|\beta|^2}.$$

对 SU(2) 元素 $\mathrm{Tr}\,b=2b_0$，所以

$$p(a)\propto\exp\Big[\frac{2\beta k}{3}\,(aV_0)_0\Big].$$

令 $b=aV_0$。Haar 测度左右不变，$\mathrm da=\mathrm db$，于是 $b$ 服从 $e^{\alpha_{\rm KP}b_0}\mathrm db$，$\alpha_{\rm KP}=2\beta k/3$。抽出 $b$ 后 $a=bV_0^\dagger$。这就是 `cm_step` 里 `su2_random(...) * V0'` 一行的来历。

三个子群 $(1,2),(1,3),(2,3)$ 依次更新，每一步对当前条件分布都是精确热浴，因此三步的复合保持目标分布。三个子群生成整个 SU(3)，遍历性成立。这个复合不是 SU(3) 的精确热浴，一次更新后新链与 staple 仍有残余关联，自相关略长于理想热浴，但正确性不受影响。

## 4. SU(2) 热浴与 Kennedy–Pendleton 抽样

SU(2) 元素 $b=b_0\mathbb 1+i\,\vec b\cdot\vec\sigma$，$b_0^2+|\vec b|^2=1$，群流形是 $S^3$，Haar 测度是 $S^3$ 上的均匀分布

$$\mathrm db\propto\delta(b_0^2+|\vec b|^2-1)\,\mathrm d^4b\propto\sqrt{1-b_0^2}\,\mathrm db_0\,\mathrm d\Omega .$$

所以 $b_0$ 的边缘分布是 $\sqrt{1-b_0^2}\,e^{\alpha b_0}$，方向 $\hat b$ 在 $S^2$ 上均匀。记 $\delta=1-b_0\in[0,2]$，密度变成 $\sqrt{\delta(2-\delta)}\,e^{-\alpha\delta}$。

Kennedy–Pendleton 分两步。先从 $\sqrt\delta\,e^{-\alpha\delta}$，也就是 Gamma(3/2, α)，抽 $\delta$。再以概率 $\sqrt{1-\delta/2}$ 接受。两个因子相乘正好是目标密度。Gamma(3/2) 的构造用了三个均匀数 $r_1,r_2,r_3$

$$\delta=-\frac{\ln r_1+\cos^2(2\pi r_2)\,\ln r_3}{\alpha}.$$

其中 $-\ln r_1\sim\mathrm{Gamma}(1)$。$\cos^2(2\pi r_2)$ 服从反正弦律即 Beta(1/2, 1/2)，而 $\mathrm{Gamma}(1)\times\mathrm{Beta}(1/2,1/2)=\mathrm{Gamma}(1/2)$，两项之和是 Gamma(3/2)。接受判据写成 $r_4^2\le1-\delta/2$。$\delta>2$ 时右边为负，自动拒绝。$r_1=0$ 给出 $\delta=\infty$ 同样被拒绝，不会产生 NaN。

接受率在 $\alpha$ 大时接近 1，$\alpha\to0$ 时按 $\alpha^{3/2}$ 下降。$\beta=6$ 时热化后 $k\approx1.2$ 到 $1.8$，$\alpha_{\rm KP}\approx5$ 到 $7$，效率很高。$k$ 严格为零的概率为零。

方向抽样取 $\cos\theta$ 均匀于 $[-1,1]$，$\varphi$ 均匀于 $[0,2\pi)$，$|\vec b|=\sqrt{1-b_0^2}$。矩阵形式

$$b=\begin{pmatrix}b_0+ib_3&b_2+ib_1\\-b_2+ib_1&b_0-ib_3\end{pmatrix},$$

与第 3 节 $Q$ 的参数化一致，$\alpha\leftrightarrow b_0+ib_3$，$\beta\leftrightarrow b_2+ib_1$。

## 5. 过松弛

过松弛是微正则更新，改变链但不改变作用量。在子群里 $\mathrm{Re}\,\mathrm{Tr}\,(aV_0)$ 的最大值在 $a^*=V_0^\dagger$。把当前元素 $a=\mathbb 1$ 关于 $a^*$ 反射得

$$a=a^*\,\mathbb 1^{-1}\,a^*=(V_0^\dagger)^2 .$$

验证作用量不变。$\mathrm{Re}\,\mathrm{Tr}\,[(V_0^\dagger)^2\,kV_0]=k\,\mathrm{Re}\,\mathrm{Tr}\,V_0^\dagger=k\,\mathrm{Re}\,\mathrm{Tr}\,V_0=\mathrm{Re}\,\mathrm{Tr}\,W_{\rm sub}$。反射后新的 $V_0'=V_0^\dagger$，再反射一次回到原点，所以它是对合。它又是 Haar 测度上的等距，因此保持 Boltzmann 分布。单独使用不遍历，与热浴混合后能显著缩短自相关。程序里每次更新是 1 次热浴扫描加 `n_or` 次过松弛扫描。

## 6. 再幺正化

浮点误差让链慢慢偏离 SU(3)。每次链更新后做一次 Gram–Schmidt。前两行正交归一，第三行取 $r_3=\overline{r_1\times r_2}$。行正交归一保证幺正。行列式

$$\det U=r_1\cdot(r_2\times r_3)=(r_1\times r_2)\cdot r_3=(r_1\times r_2)\cdot\overline{(r_1\times r_2)}=|r_1\times r_2|^2=1,$$

最后一步用了复向量的 Lagrange 恒等式 $|a\times b|^2=|a|^2|b|^2-|\langle a,b\rangle|^2$。所以 det 严格为 1，不需要再除相位。实测不做再幺正化时 $4\times10^6$ 次连乘后偏离也只有 $6\times10^{-12}$，但这一步只花 30 ns。

## 7. 棋盘格并行

单链的条件分布只依赖它的 staple。方向相同、$x_1+x_2$ 奇偶相同的两条链不会出现在同一个 plaquette 里。$U_\mu(x)$ 只与 $U_\mu(x\pm\hat\nu)$ 共享 plaquette，后者奇偶相反。所以同一类 $(\mu,\text{parity})$ 的链可以同时更新，结果等价于按任意顺序串行更新。一次扫描依次处理 4 类。

线程之间各用一个独立的 Xoshiro 生成器，`Threads.@threads :static` 把行 $x_2$ 到线程的分配固定下来，相同 seed 和线程数下结果可复现。

$L$ 必须是偶数。奇数 $L$ 时跨周期边界的 $x_1+x_2$ 与 $x_1+x_2+L$ 奇偶相同，同一类里会出现共享 plaquette 的链，并行更新产生数据竞争。程序没有检查这一点。

## 8. 精确解

在周期 $L\times L$ 格子上做特征展开 $e^{(\beta/3)\mathrm{Re}\,\mathrm{Tr}\,U}=\sum_r f_r\,\chi_r(U)$，逐链积分后环面上的配分函数是

$$Z=\sum_r\Big(\frac{f_r}{d_r}\Big)^{L^2}.$$

$L\to\infty$ 时平凡表示主导，$\ln Z/L^2\to\ln f_0=\ln Z_1(\beta)$，其中

$$Z_1(\beta)=\int_{SU(3)}\mathrm dU\,e^{(\beta/3)\mathrm{Re}\,\mathrm{Tr}\,U},\qquad
\langle P\rangle=\frac1{L^2}\frac{\partial\ln Z}{\partial\beta}\;\longrightarrow\;\frac{\mathrm d\ln Z_1}{\mathrm d\beta}.$$

有限体积修正相对大小为 $\big(f_{\rm fund}/(3f_0)\big)^{L^2}\approx\langle P\rangle^{L^2}$，$L=8$、$\beta=6$ 时约 $10^{-24}$，可以忽略。

单格积分有闭式。U(3) 上 $\int\mathrm dU\,e^{c\,\mathrm{Tr}(U+U^\dagger)}=\det[I_{i-j}(2c)]_{i,j=1..3}$。这里 $e^{(\beta/3)\mathrm{Re}\,\mathrm{Tr}\,U}=e^{(\beta/6)\mathrm{Tr}(U+U^\dagger)}$，$c=\beta/6$，Bessel 函数的宗量是 $\beta/3$。SU(3) 比 U(3) 多一个 $\det U=1$ 的约束，把 $\delta$ 函数展成 $\sum_q(\det U)^q$ 后每一项把 Bessel 指标平移 $q$

$$Z_1(\beta)=\sum_{q\in\mathbb Z}\det\big[I_{i-j+q}(\beta/3)\big]_{i,j=1..3}.$$

程序用 SpecialFunctions 的 `besselix(n, x) = e^{-x}I_n(x)`，三行各带一个 $e^{-\beta/3}$，算出的是 $e^{-\beta}Z_1$。它对 $\ln$ 的导数比真值少 1，`plaq_exact` 加回去。$q$ 求和截断在 $|q|\le20+6\sqrt{\beta/3}$，$I_q$ 随 $q$ 超指数衰减，早已收敛。导数用步长 $10^{-3}$ 的中心差分，精度约 $10^{-7}$。

两个独立检验。强耦合展开给 $\langle P\rangle=\beta/18+\beta^2/216+\dots$，用到 $\int(\mathrm{Re}\,\mathrm{Tr}\,U)^2\mathrm dU=1/2$ 和 $\int(\mathrm{Re}\,\mathrm{Tr}\,U)^3\mathrm dU=1/4$。$\beta=0.1$ 时公式给 0.00560185，展开给 0.005602。另外用 Weyl 积分公式在两个特征值角度上直接数值二重积分，$\beta$ 从 0.1 到 30 与 Bessel 公式一致到 $10^{-11}$。

## 9. 程序结构

| 函数 | 作用 | 对应章节 |
|---|---|---|
| `limitbound`, `wrap`, `E` | 周期边界。`wrap` 对 `CartesianIndex` 的每个分量做 `limitbound`，`E` 是两个单位位移 | 1 |
| `staple(U, μ, c, L)` | 两个 staple 之和 $\Sigma$ | 2 |
| `su2_project(W, i, j)` | $(i,j)$ 块到 $\mathbb R\cdot SU(2)$ 的投影，返回 $k$ 和 $V_0$ | 3 |
| `embed(a, i, j)` | SU(2) 元素嵌入 SU(3)，写成对单位矩阵的四次 `setindex` | 3 |
| `kp_a0(rng, α)` | Kennedy–Pendleton 抽 $b_0$ | 4 |
| `su2_random(rng, a0)` | 给定 $b_0$，方向均匀，返回 SU(2) 矩阵 | 4 |
| `cm_step(V, W, i, j, β, rng, heatbath)` | 一个子群的热浴或过松弛，返回更新后的 $(V, W)$ | 3, 5 |
| `update_link!` | 三个子群依次调用 `cm_step`，再幺正化后写回 | 3, 6 |
| `reunit(V)` | Gram–Schmidt 加叉积 | 6 |
| `sweep!` | 4 类棋盘格的多线程扫描 | 7 |
| `plaquette(U, L)` | 平均 plaquette | 1 |
| `Z1_su3`, `plaq_exact` | 单格积分与精确 $\langle P\rangle$ | 8 |
| `mean_err_tau` | 均值、误差、积分自相关时间 | 10 |
| `main` | 参数、热化、采样、HDF5 输出 | 10 |

`update_link!` 里三个子群写成三行字面调用而不是循环。索引是编译期常量时 `su2_project` 和 `embed` 被完全折叠，整条链内联成一段直线代码。循环写法慢 1.7 倍。

链存成 `Array{SMatrix{3,3,ComplexF64,9}, 3}`，形状 `(2, L, L)`，`U[μ, c]` 用 `CartesianIndex` 取值。3×3 的 SMatrix 乘法约 12 ns 且不分配内存，是这个规模下 Julia 里最快的表示。

## 10. 使用

```bash
julia -t 8 su3_2d_heatbath.jl --beta 6.0 --L 16 --n_cfgs 10000 \
      [--n_therm 500] [--n_skip 5] [--n_or 3] [--seed 1] [--out .]
```

从冷启动（所有链为单位矩阵）做 `n_therm` 次更新热化，然后每隔 `n_skip` 次更新保存一个构型，共 `n_cfgs` 个。一次更新是 1 次热浴扫描加 `n_or` 次过松弛扫描。`include` 这个文件只定义函数，`main()` 只在作为脚本运行时执行。

输出文件 `cfgs_su3_beta=<β>_<L>^2.h5`。

| 数据集 | 内容 |
|---|---|
| `cfgs` | ComplexF64。Julia 里形状 `(3, 3, 2, L, L, n_cfgs)`，`cfgs[b, a, μ, x, y, n] = (U_μ(x,y))_{ab}`。h5py 按行主序读到形状 `(n_cfgs, L, L, 2, 3, 3)`，`cfgs[n, y, x, mu, a, b] = (U_mu(x,y))_{ab}`，即最后两维就是通常的矩阵下标 |
| `plaq` | 每个保存构型的平均 plaquette |
| 属性 | `beta, L, n_therm, n_skip, n_or, seed, plaq_exact` |

写盘前对每个 3×3 块做一次转置，正是为了让 h5py 一侧得到自然的 `[a, b]` 顺序。每个构型 $288L^2$ 字节，$L=16$、一万个构型约 0.74 GB。

结束时打印 $\langle P\rangle$ 及误差、积分自相关时间 $\tau_{\rm int}$ 和与精确值的偏离（以 $\sigma$ 计）。误差按 $\sigma^2=2\tau_{\rm int}\,\mathrm{var}/N$ 估计，$\tau_{\rm int}$ 的求和窗口取到第一个非正的自相关系数。二维里保存构型之间基本独立，$\tau_{\rm int}\approx0.5$。

性能（GH200 节点，2026-09-29）。每条链每次更新单线程约 300 ns，16 线程下 $L=64$ 约 31 ns。默认参数（$L=16$，$\beta=6$，一万个构型，`n_skip=5`，`n_therm=500`）16 线程含写盘共 15 s。

## 11. 验证记录

2026-09-29 完成的检验。

- 精确公式。Bessel 行列式与 Weyl 积分在 $\beta=0.1$ 到 30 一致到 $10^{-11}$，小 $\beta$ 与强耦合展开一致。
- 模拟与精确值。$\beta=6$，$L=8$，30000 个构型，$\langle P\rangle=0.422545(188)$，偏 $+0.07\sigma$。$L=16$，10000 个构型，$0.422338(163)$，偏 $-1.2\sigma$。$\beta=2$，$L=8$，10000 个构型，$0.128276(334)$，偏 $-1.1\sigma$。精确值分别为 0.4225317 和 0.1286282。
- 输出布局。用 h5py 按上表的索引约定重算 plaquette，与存盘的 `plaq` 一致到 $10^{-15}$。幺正性和 det 偏离 $10^{-15}$。
- 单链精确解。取 $\Sigma=2\cdot\mathbb 1$、$\beta=3$，单链模型的权重就是 $\beta=6$ 的单格积分。本程序的链更新、Gaugefields.jl 的便携内核、MetaQCD.jl 风格的内核各做 $4\times10^6$ 次更新，都在 $1.3\times10^{-4}$ 的误差内与 0.4225317 一致。
- 与 Gaugefields.jl 整格 2D 热浴对比，plaquette 一致，本程序单线程快 4.6 倍。

## 12. 注意事项

- $L$ 取偶数。
- $\beta$ 很小或 staple 近乎抵消时 $\alpha_{\rm KP}$ 小，Kennedy–Pendleton 的拒绝次数增多，只影响速度不影响正确性。
- Cabibbo–Marinari 不是 SU(3) 的精确热浴，`n_or` 次过松弛负责压自相关。`n_or=3` 下保存构型之间已经独立。
- 相同 seed 但不同线程数给出不同的随机流。
- 全部构型按顺序流式写入，程序中途终止时已写入的部分仍然可读。
