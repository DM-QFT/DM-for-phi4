# ModelBFixedSOSRA SciML validation

Revalidated on `mg0223` (NVIDIA GH200 120GB), Julia 1.12.6,
SciMLBase 3.35.1, and StochasticDiffEq 7.1.1.

## Correctness

```bash
julia --startup-file=no --project=@. checks/test_fixed_sciml.jl
```

All 38 tests pass.  They cover `L=32,64,128`; at every size the specialized drift
is elementwise identical to the Cartesian-index reference drift.  The fused
noise/stage-1 kernel also produces `dW`, `q`, `k1`, and `h01` elementwise
identical to independent periodic-divergence and drift calculations.  With the
same Philox seed, both the allocation-owning and
buffer-reusing SciML paths are elementwise identical to `modelB_2d_fixed!`.
Repeated solves with the same seed are also elementwise identical.  The maximum
ensemble mean-field change in the four-ensemble test is below `2e-6`.

The same backend was also tested for stationarity on 512 evenly spaced L=128
critical HMC configurations without subtracting each configuration's mean.
They were evolved to `t=5` with `dt=0.002` for two independent Philox
repetitions at `kappa=0.2705`, `lambda=0.022`, and `gamma=T=1`.  The relative
drifts (paired z scores) were
`+0.000306%` (`0.004`) for the action, `-0.017045%` (`-0.525`) for `phi2`,
`-0.000759%` (`-0.011`) for `phi4`, and `-0.021997%` (`-0.597`) for the
nearest-neighbor observable.  None is statistically distinguishable from
zero.  The largest magnetization-density change was `5.97e-8`.

For 33,554,432 Philox normal samples, the measured mean magnitude was below
`4.4e-4`, variance was in `[0.9992, 1.0008]`, and cross-component correlation
magnitude was below `2.9e-4`.

## L=128, 1024 ensembles, 1000 steps

```bash
julia --startup-file=no --project=@. checks/benchmark_fixed_sciml.jl
```

Sixteen paired wall-clock measurements alternated direct/SciML execution order.
Both sides include the initial-state copy, seed reset, reused 640 MiB buffers,
the full GPU solve, and synchronization.

| Measurement | Direct fixed | SciML + reused buffers |
|---|---:|---:|
| Median wall time | 0.606200 s | 0.606036 s |
| Median paired ratio (SciML/direct) | | 0.999148 |
| Paired-ratio IQR | | 0.998666--0.999740 |

All sixteen final arrays were elementwise identical.  The difference is inside
normal run-to-run variation: the SciML interface and direct fixed interface have
the same measured hot performance.

The default `ModelBFixedSOSRA()` path owns fresh buffers.  Its first call took
0.732215 s and its subsequent median was 0.607603 s.  Reuse is therefore
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

## Further simplification and optimization audit

The out-of-place constructor now uses `CuArray{Float32}(u0)` directly.  For an
L=128, 1024-ensemble Float32 `CuArray`, the independent initial-state copy fell
from 0.4731 ms to 0.2596 ms while remaining elementwise identical and leaving
the input untouched.  The direct and SciML paths also share one fixed-step
count check, and the explicit Philox constructor relies on CUDA.jl's identical
default-current-stream behavior instead of repeating the stream argument.

The following hot-path candidates were measured and rejected:

- Passing stage 1's already-computed geometry into the drift kept the same 40
  registers and had a paired ratio of 1.000; the compiler already removes the
  apparent duplicate scalar indexing work.
- Reusing one array for `q` and `k2` is numerically valid and saves 64 MiB, but
  the store order is unexpectedly fragile.  Overwriting `q` before writing
  `h02` made stage 2 take 1.059 ms instead of 0.152 ms because it introduced a
  strong same-address load/store ordering dependency before the old `q` value's
  final consumer.  Moving that store to the end restored 0.150 ms and gave a
  full-solver paired ratio of 0.992.
  Despite that small gain, the production code keeps separate `q` and `k2`
  arrays because their mathematical roles and lifetimes remain explicit.
- A full thread/block sweep retained 768 threads, 264 blocks for the 40-register
  fused stage 1 and 1024 threads, 264 blocks for stages 2 and 3.  Alternatives
  were either slower or within sub-percent timing noise.
- Philox dynamic ordering reduced the current complete solve from 0.6060 s to
  0.5930 s (2.1%), but its seed-to-array ordering may change across GPU models
  and cuRAND releases.  It remains an opt-in experiment rather than the
  reproducible default until the HMC/statistical validation is repeated for
  that stream.

## Supported scope

- Float32, `(L, L, Nensemble)` CuArray state, with power-of-two `L >= 4`
- fixed positive `dt` and forward `tspan` containing an integer number of steps
- final-state output, with optional initial-state output
- optional explicit `seed`

Adaptive stepping, callbacks, dense/intermediate output, non-power-of-two
lattices, and generic `SDEProblem` functions/noise are intentionally rejected.
