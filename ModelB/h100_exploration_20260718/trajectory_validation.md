# Fixed-step trajectory validation

Configuration: `L=128`, `Nensemble=1024`, `Float32`, `SOSRA`,
`adaptive=false`, `dt=0.002`, ten accepted steps (`t=0.02`).  Every comparison
starts from the same `u0` and reseeds the CUDA RNG with `271828`.  Production
and experimental kernels are compiled and warmed before comparison.

| Comparison | Exact elementwise equality | Max absolute difference | RMS difference |
|---|---:|---:|---:|
| production RSwM3 async vs sync | true | 0 | 0 |
| direct-index + fused-noise RSwM3 vs production | true | 0 | 0 |
| production RSwM1 vs production RSwM3 | true | 0 | 0 |
| direct-index + fused-noise RSwM1 vs production RSwM1 | true | 0 | 0 |
| direct-index + fused-noise RSwM1 vs production RSwM3 | true | 0 | 0 |

All solves returned `Success`, accepted exactly 10 steps, and had finite
outputs.  The largest absolute conserved-mode residual across the runs was
`3.31878662e-4` before division by the `128^2` sites.

The raw terminal output can be reproduced with:

```bash
julia --project=. h100_exploration_20260718/validate_solver_variants.jl
```

This test establishes that callback synchronization removal, RSwM1 in a
non-adaptive/no-rejection solve, the L=128 direct-index drift, and fused
divergence/scale do not change this fixed-step trajectory.  Explicitly
changing the random-number generator is a separate statistical comparison and
is not expected to preserve a trajectory bit-for-bit.
