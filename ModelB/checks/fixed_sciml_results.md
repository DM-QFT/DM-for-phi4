# ModelBFixedSOSRA SciML validation

Revalidated on `mg0223` (NVIDIA GH200 120GB), Julia 1.12.6,
SciMLBase 3.35.1, and StochasticDiffEq 7.1.1.

## Correctness

```bash
julia --startup-file=no --project=@. checks/test_fixed_sciml.jl
```

All tests pass.  They cover `L=32,64,128`; at every size the specialized drift
is elementwise identical to the Cartesian-index reference drift.  The fused
noise/stage-1 kernel also produces `dW`, `q`, `k1`, and `h01` elementwise
identical to independent periodic-divergence and drift calculations.  With the
same Philox seed, both the allocation-owning and
buffer-reusing SciML paths are elementwise identical to `modelB_2d_fixed!`.
Repeated solves with the same seed are also elementwise identical.  The maximum
ensemble mean-field change in the four-ensemble test is below `2e-6`.

The wrapper calls the same backend that was used for the existing HMC
stationarity validation.  The root backend and that validation backend remain
elementwise identical even for a real HMC input with 2500 steps, so the HMC
observable results in `h100_exploration_20260718/hmc_validation.md` apply
unchanged to this SciML interface.

## L=128, 1024 ensembles, 1000 steps

```bash
julia --startup-file=no --project=@. checks/benchmark_fixed_sciml.jl
```

Sixteen paired wall-clock measurements alternated direct/SciML execution order.
Both sides include the initial-state copy, seed reset, reused 640 MiB buffers,
the full GPU solve, and synchronization.

| Measurement | Direct fixed | SciML + reused buffers |
|---|---:|---:|
| Median wall time | 0.606951 s | 0.606973 s |
| Median paired ratio (SciML/direct) | | 1.000107 |
| Paired-ratio IQR | | 0.999315--1.000315 |

All sixteen final arrays were elementwise identical.  The difference is inside
normal run-to-run variation: the SciML interface and direct fixed interface have
the same measured hot performance.

The default `ModelBFixedSOSRA()` path owns fresh buffers.  Its first call took
0.731709 s and its subsequent median was 0.608341 s.  Reuse is therefore
recommended for repeated large solves even though CUDA's memory pool makes the
hot allocation cost small.

The variable-`L` implementation stores `Val(L)` in its buffers so each
power-of-two lattice compiles to constant shifts and masks.  It also forms the
SOSRA stage noise `q = sigma*(dW + dZ/sqrt(3))/2` once, and evaluates
`drift(u)` in that same kernel.  This removes the old increments launch and the
stage-1 read of `q`.  An eight-pair L=128 comparison against the old hard-coded,
separate-`dZ`/stage-1 backend measured 0.605811 s versus 0.666668 s (paired ratio
0.908753).  All trajectories were elementwise identical.  The fused kernel
uses 40 registers per thread with 768 threads and a 264-block cap; stages 2 and
3 remain 32-register, 1024-thread kernels.

With 1024 ensembles and 1000 steps, five hot solver-only runs gave:

| L | Median solve time | Time per step |
|---:|---:|---:|
| 32 | 0.044770 s | 0.044770 ms |
| 64 | 0.167411 s | 0.167411 ms |
| 128 | 0.604523 s | 0.604523 ms |

All three final states were finite, and the largest ensemble mean-field change
was below `1e-7`.

## Scalar-return drift versus in-place drift

`modelb_fixed_drift` returns one `Float32` value that remains in a register; it
does not allocate an array.  A same-kernel test that changed only the syntax to
an inline `drift_store!(k1, ...)` used the same 40 registers, remained
elementwise identical, and had a paired time ratio of 1.0036.  It therefore
provided no performance benefit.

A true in-place materialization requires one kernel to write all of `k1` and a
second kernel to read it while assembling stage 1.  After tuning each variant,
the stage-1 portion took 0.2588 ms instead of 0.2013 ms, 28.5% slower.  In the
complete 1000-step solver the materialized version was 9.3% slower.  Its output
was still elementwise identical, confirming that the difference is the extra
kernel boundary and global-memory read rather than a numerical change.

## Supported scope

- Float32, `(L, L, Nensemble)` CuArray state, with power-of-two `L >= 4`
- fixed positive `dt` and forward `tspan` containing an integer number of steps
- final-state output, with optional initial-state output
- optional explicit `seed`

Adaptive stepping, callbacks, dense/intermediate output, non-power-of-two
lattices, and generic `SDEProblem` functions/noise are intentionally rejected.
