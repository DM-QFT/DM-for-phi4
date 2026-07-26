# 实空间有限时间分布测试

这里的 Euler--Maruyama、stochastic Heun 和生产 SOSRA 使用完全相同的
`ModelB_fixed.jl` 实空间漂移与守恒 link noise。测试不使用 FFT、谱推进、
Kawasaki sweep 或辅助动量动力学。

`test_methods.jl` 检查确定性一步、实空间观测量和逐配置守恒量。
`compare_distributions.jl` 在 `t=0:0.1:10` 比较不同方法和步长的有限时间
边缘分布，并把相对 `SOSRA, dt=0.0005` 的结果写入 `output/`。

六个逐配置统计量是守恒均值、固定格点场值、空间平均 `ϕ²`、空间平均
`ϕ⁴`、最近邻相关和作用量密度。标量分布使用 KS 与标准化一维
Wasserstein 距离；六维联合统计使用 16 个固定随机方向的
sliced-Wasserstein 距离。第二份独立的细步长 SOSRA 给出有限样本噪声底。
这些统计量用于筛选积分器，不等价于证明 1024 维全场分布相同。

```bash
julia performance_tests/realspace_distribution/test_methods.jl
julia performance_tests/realspace_distribution/compare_distributions.jl
```

实测结论见 `results.md`。`output/stats_*.jld2` 保留 101 个时刻的逐配置
统计量，完整逐时指标与汇总分别在 `time_metrics.tsv`、
`joint_metrics.tsv` 和 `aggregate_metrics.tsv`。
