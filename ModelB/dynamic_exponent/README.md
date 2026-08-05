# Model B 的 \(L=32\) 色散 \(\tau(k)\)

本目录只保留二维 Model B 在 \(L=32,\lambda=1\) 的动量色散分析。
不使用不同 \(L\) 的 \(\tau(L)\) 做有限尺寸标度拟合 \(z\)，也不采用固定
\(M=0\) 初态。

初态由无约束的 Wolff+FAHMC canonical 系综产生，因此不同轨迹的总磁化
\(M\) 按平衡分布涨落。守恒 Model B 演化随后分别保持每条轨迹自己的 \(M\)，
这不等于把整个 ensemble 固定在 \(M=0\) 扇区。

## 计算流程

1. `scan_pseudocritical.jl` 扫描
   \(\chi_{|m|}=L^2(\langle m^2\rangle-\langle|m|\rangle^2)\)，
   `analyze_pseudocritical.py` 确定 \(L=32\) 的赝临界点。
2. `make_canonical.jl` 在指定耦合生成无固定磁化约束的 canonical 初态。
3. `save_configs.jl` 用内部步长 `dt=0.005` 演化并只保存实空间场。
4. 场历史完整写入后，`analyze_dispersion.py` 才逐时间片 FFT，分别计算
   轴向 `(n,0)` 和对角 `(n,n)` 的 \(\tau_{1/e}(k)\)。
5. `analyze_tmax_stability.py` 检查最低模对演化长度及前后时间窗的稳定性。

动量使用
\[
\hat k^2=\sum_\mu4\sin^2(k_\mu/2),
\]
\(\tau_{1/e}\) 是归一化自相关首次到达 \(1/e\) 的对数插值时间。图中的
\(\hat k^{-4}\) 与 \(\hat k^{-3.75}\) 只是色散参考线，不是有限尺寸拟合。

## 数据

- `results/critical_points.tsv` 和 `critical_crossings.json` 只记录设置真临界
  耦合所需的静态信息，不用于动力学 FSS 拟合。
- `results/canonical_l1/can_L32.jld2` 是真临界细批场的 canonical 初态。
- `results/dispersion_L32/` 保存真临界和赝临界的实空间历史；FFT 后的
  真临界结果在 `critical/`，赝临界结果在 `pseudocritical_t6000/`。

参数、误差、守恒检查和文件清单见
[`results/dispersion_L32/README.md`](results/dispersion_L32/README.md)。
