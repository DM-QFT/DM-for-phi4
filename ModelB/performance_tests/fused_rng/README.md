# Model-B RNG data-path tests

这个测试把每条有向 link 的随机数定义为
`Philox(link_index, step, seed)`，在 SOSRA stage 1 中按索引重算当前、左侧
和下侧 link。相邻格点因此使用完全相同的共享 link，噪声仍是离散散度。

实现调用 CUDA.jl 当前默认 GPU RNG 所使用的 Philox4x32-10 和 Float32
Box--Muller primitives，不使用 reference 程序中的 `rand()` 或有状态
Tausworthe generator。`solver/` 未修改。

```bash
julia performance_tests/fused_rng/test_fused_philox.jl
julia performance_tests/fused_rng/benchmark_rng_fill.jl
julia performance_tests/fused_rng/benchmark_fused_philox.jl
julia performance_tests/fused_rng/benchmark_shapes.jl
julia performance_tests/fused_rng/compare_l32_distributions.jl
```

测试还包括三种不融合 stage 1 的方案：预分配 padded cuRAND buffer、公开
`CUDA.RNG` 和直接偶数长度 cuRAND。实际 `L=32,Nensemble=5120` 的主要
瓶颈是 CUDA.jl 高层 cuRAND wrapper 每步 padding、临时分配和 D2D copy。

直接为每个 site 重算三组 Philox 会增加寄存器和重复计算，并不会加速。
有效 fusion 使用 `32×8` tile：block 内 link noise 只生成一次并放入 shared
memory，首个 warp 补齐左侧和下侧 halo。实测结论见 `results.md`。
