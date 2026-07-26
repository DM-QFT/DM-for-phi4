# Model B fixed SOSRA checks

The active implementation is the fixed-step, additive-noise SOSRA solver in
`../data_generation/ModelB_fixed.jl`.  It accepts a `Float32` CuArray with shape
`(L, L, Nensemble)`, where `L` is a power of two.  The SciML-compatible wrapper
is in `../data_generation/ModelB_fixed_sciml.jl`.  The general SciML SOSRA
implementation is retained in `../data_generation/ModelB.jl`.

This directory contains only:

- `test_fixed_sciml.jl`: correctness, reproducibility, conserved mean, and
  `L=32,64,128` checks;
- `test_generic_modelb.jl`: explicit-seed reproducibility for the general
  SciML implementation;
- `benchmark_fixed_sciml.jl`: paired direct/SciML L=128 benchmark;
- `fixed_sciml_results.md`: numerical, HMC, and performance results.

## Run

```bash
julia --startup-file=no --project=@. checks/test_fixed_sciml.jl
julia --startup-file=no --project=@. checks/test_generic_modelb.jl
```

Minimal use from Julia:

```julia
using CUDA, Statistics
include("data_generation/ModelB_fixed.jl")
u0 = CUDA.randn(Float32, 128, 128, 1024)
u0 .-= mean(u0; dims=(1, 2))
buffers = modelb_fixed_buffers(u0; seed=UInt64(20260719))
modelB_2d_fixed!(u0, buffers; tspan=(0.0f0, 2.0f0), dt=0.002f0)
CUDA.synchronize()
```

To keep intermediate GPU states through the SciML wrapper:

```julia
using CUDA
include("data_generation/ModelB_fixed_sciml.jl")

prob = modelB_2d_fixed_problem(u0; tspan=(0.0f0, 10.0f0))
buffers = modelb_fixed_buffers(prob.u0; seed=UInt64(20260719))
sol = solve(
    prob, ModelBFixedSOSRA(; buffers);
    dt=0.002f0, saveat=0.5f0, save_start=true,
    seed=UInt64(20260719),
)
```

Every `saveat` time must be an integer number of `dt` steps.  `sol.u` contains
independent CuArray snapshots; omit `save_start=true` to exclude `t=0`.  At
L=128 with 1024 ensembles each snapshot is 64 MiB, so `0:0.5:10` occupies
1.3125 GiB.
