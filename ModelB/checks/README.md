# Model B fixed SOSRA checks

The active implementation is the fixed-step, additive-noise SOSRA solver in
`../ModelB_fixed.jl`.  It accepts a `Float32` CuArray with shape
`(L, L, Nensemble)`, where `L` is a power of two.  The SciML-compatible wrapper
is in `../ModelB_fixed_sciml.jl`.  The general SciML SOSRA implementation is
retained in `../ModelB.jl`.

This directory contains only:

- `test_fixed_sciml.jl`: correctness, reproducibility, conserved mean, and
  `L=32,64,128` checks;
- `benchmark_fixed_sciml.jl`: paired direct/SciML L=128 benchmark;
- `fixed_sciml_results.md`: numerical, HMC, and performance results.

## Run

```bash
julia --startup-file=no --project=@. checks/test_fixed_sciml.jl
```

Minimal use from Julia:

```julia
using CUDA, Statistics
include("ModelB_fixed.jl")
u0 = CUDA.randn(Float32, 128, 128, 1024)
u0 .-= mean(u0; dims=(1, 2))
buffers = modelb_fixed_buffers(u0; seed=UInt64(20260719))
modelB_2d_fixed!(u0, buffers; tspan=(0.0f0, 2.0f0), dt=0.002f0)
CUDA.synchronize()
```
