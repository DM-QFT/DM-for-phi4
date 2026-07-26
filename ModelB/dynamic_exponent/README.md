# 2D Model B 动力学临界指数

有限尺寸观测量是固定总磁化 \(M=0\) 扇区中的最低轴向模：

\[
C_L(t)=\frac{\langle\phi_{k_{\min}}(t_0+t)
\phi_{k_{\min}}^*(t_0)\rangle}{\langle|\phi_{k_{\min}}|^2\rangle},
\qquad k_{\min}=2\pi/L .
\]

在 \(0.8\ge C_L\ge0.2\) 内拟合 \(C_L(t)=e^{-t/\tau_L}\)，再拟合
\(\tau_L\propto L^z\)。二维 Model B 的渐近结果为
\(z=4-\eta=3.75\)。

## 三个脚本

固定 \(M=0\) 条件平衡初态：

```bash
julia -O3 dynamic_exponent/make_m0.jl \
  L N kappa lambda warmup thin nstep seed output.jld2
```

约束 HMC 将 Fourier 质量矩阵的零模速度置零，并使用二维 Ising
\(G^{-1}(k)\propto k^{1.75}\) 预条件；Wolff 翻转不能用于固定 \(M\)
扇区。输出同时给出最低模相邻样本相关系数 `rho1`。

GPU 动力学只记录两个最低轴向模：

```bash
julia -O3 dynamic_exponent/measure_modes.jl \
  input_m0.jld2 output_modes.jld2 tmax sample_dt dt seed
```

脚本直接调用未修改的 `solver/ModelB_fixed.jl`。每个观测时刻先在 GPU
上做横向求和，只传输 \(2LN\) 个数，不保存完整时空场。

拟合：

```bash
zsh -ic 'nenv2 && python dynamic_exponent/fit_z.py \
  output_dir modes_L8.jld2 modes_L16.jld2 modes_L32.jld2 ...'
```

三个脚本合计约 300 行。

## 临界点与计算设置

最终 Binder crossing 使用 \(L=128,256,512\)：

- \(\lambda=0.022:\ \kappa_c\simeq0.27087\)，动力学用
  `0.27084–0.27088` 传播临界点系统误差；
- \(\lambda=1:\ \kappa_c=0.34034(8)\)。

\(\lambda=0.022\) 的生产步长为 `dt=0.02`；`L=32` 在
`dt=0.005–0.02` 内相容，四尺寸 `dt=0.01,0.02` 的 \(z\)
相差 `0.008`。较硬的 \(\lambda=1\) 使用 `dt=0.0025`，
`dt=0.01` 会数值发散。

主计算保持 \(L^2N\gtrsim10^6\)：

| \(L\) | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|
| \(N\) | 16384 | 4096 | 1024 | 256 | 128 |

在 GH200 上，\(L=128,N=128\) 为 `82.75 μs/step`，达到同 solver
饱和吞吐的约 87%；在线观测后的稳态开销约 2.5%。
