# ModelBFixedSOSRA SciML validation

Revalidated on `mg0074` (NVIDIA GH200 120GB), Julia 1.12.6,
CUDA.jl 6.3.0, SciMLBase 3.49.1, and StochasticDiffEq 7.1.4.

## Correctness

```bash
julia --startup-file=no \
  --project=/work/gw97/w97000/envs/modelb_julia_benchmark \
  tests/test_fixed_sciml.jl
```

The fixed-solver test suite covers `L=32,64,128`.  At every size the specialized
drift agrees with the Cartesian-index reference within 16 Float32 epsilons
times the reference scale.  This tolerance covers last-bit arithmetic-order
changes across CUDA compilers.  With fixed link values, independent
periodic-divergence and drift calculations verify `q`, `k1`, `h01`, the
partial final state in `h02`, both stage-2 outputs, and the final state.  The
tests also confirm that
the buffers no longer contain `dW` or `k2` arrays.  With the same Philox seed,
both the allocation-owning and buffer-reusing SciML paths are elementwise
identical to `modelB_2d_fixed!`.
Repeated solves with the same seed are also elementwise identical.  The maximum
ensemble mean-field change in the four-ensemble test is below `2e-6`.

The wrapper accepts a positive scalar `saveat` interval or a collection of
times aligned with `dt`.  The wrapper checks verify every saved state
against the same-seed direct trajectory, independent snapshot storage, scalar
and unsorted/repeated vector schedules, `maxiters` endpoints, conserved means,
and rejection of non-aligned save times.

The same backend was also tested for stationarity on 512 evenly spaced L=128
critical HMC configurations without subtracting each configuration's mean.
They were evolved to `t=5` with `dt=0.002` for four independent Philox
repetitions at `kappa=0.2705`, `lambda=0.022`, and `gamma=T=1`.  The relative
drifts (paired z scores) were
`+0.006534%` (`0.112`) for the action, `-0.013732%` (`-0.522`) for `phi2`,
`-0.004993%` (`-0.086`) for `phi4`, `-0.020501%` (`-0.676`) for the
nearest-neighbor observable, and `+0.014413%` (`0.247`) for
`mean(abs2(dS/dphi))`.  The paired z scores for the lowest nonzero and
quarter-lattice momentum structure factors were `1.450` and `0.212`.
None is statistically distinguishable from zero.  The largest
magnetization-density change was `9.11e-8`.

The production `dt=0.002` HMC endpoint files were generated without an active
mean projection.  A Float64 CPU reduction over all 10,240 matched trajectories
at `kappa=0.28` gives the following per-trajectory magnetization-density drift:

| Time | mean abs drift | maximum abs drift |
|---:|---:|---:|
| 1 | `1.186e-8` | `5.506e-8` |
| 10 | `3.670e-8` | `1.757e-7` |
| 100 | `1.190e-7` | `6.253e-7` |

At `t=100`, the 99th and 99.9th percentiles are `3.936e-7` and `4.980e-7`,
and the signed ensemble average is `1.543e-10`.  The drift therefore behaves
like unbiased Float32 roundoff and remains negligible on the field scale.  The
fixed solver does not apply a separate zero-mode correction.

## SDE cross-check at t=100

The same 512 equilibrium HMC configurations at `L=64` were evolved from
`t=0` to `100`.  Both SOSRA implementations used `h=0.00625`, Diffrax ShARK
used `h=0.004`, and Diffrax EM used `h=0.0004`.  Diffrax used an explicit
`StepTo` grid in each of ten physical intervals.  The entries below are
relative shifts from the initial equilibrium ensemble in percent.

| Method | Accuracy runs | action | phi2 | phi4 | nearest |
| --- | --- | --- | --- | --- | --- |
| SOSRA (This work) | 1 | -0.824 | -0.257 | -0.726 | -0.122 |
| SOSRA (StochasticDiffEq.jl) | 1 | -0.535 | -0.183 | -0.480 | -0.074 |
| ShARK (Diffrax) | 1 | -0.654 | -0.199 | -0.590 | -0.121 |
| EM (Diffrax) | 1 | +0.904 | -0.099 | +0.123 | -0.373 |

At `t=10`, the one-run SOSRA shifts were `-0.689%`, `-0.288%`, `-0.791%`,
and `-0.274%` for the fused implementation.  The corresponding
StochasticDiffEq.jl shifts were `-0.825%`, `-0.327%`, `-0.858%`, and
`-0.227%`.

At `t=10`, the one-run ShARK shifts were `-0.733%`, `-0.159%`, `-0.545%`,
and `-0.046%` in the same column order.  Their magnitudes closely match the
DPM-Solver-v3 single-transition shifts of `+0.719%`, `+0.194%`, and
`+0.532%` for the first three observables.  The timing comparison therefore
uses similar one-transition accuracy rather than requiring ShARK to be more
accurate than the learned transition.

At `t=10`, the one-run Diffrax EM shifts were `+0.775%`, `-0.064%`,
`+0.137%`, and `-0.299%` in the same column order.  Its action shift matches
the DPM-Solver-v3 value within the one-run statistical uncertainty.  The
timing benchmark uses one hot run per method.

## Nonequilibrium L=64 and N=512 cross-check

The Fig. 4 quench distribution was generated as `0.1*randn` with Xoshiro seed
`2026081007`.  The spatial mean of every trajectory was subtracted.  The
calculation used `kappa=0.85`, `lambda=T=gamma=1`, and evolved the ensemble
from `t=0` to `t=100`.  SOSRA (This work) with `h=0.0001` supplied the
reference ensemble.  DPM-Solver-v3 used 32 NFE for every physical interval
of 10 and PyTorch `max-autotune`.  Each reported time is one hot run with
startup and compilation excluded.

The reference means at `t=100` were `-2.531344`, `1.859338`, and `3.998877`
for the action density, `phi2`, and `phi4`.  Step sizes were chosen using the
root mean square of the three relative differences.  DPM-Solver-v3 gives
`0.128%`.  The calibrated primary methods span `0.081%` to `0.125%`.

```text
method                          h or NFE       time       action     phi2      phi4      RMS
DPM-Solver-v3 (This work)       32 per 10       3.258 s    -0.078    +0.098    +0.183    0.128
SOSRA (This work)               0.0025          3.280 s    -0.075    -0.061    -0.157    0.107
SOSRA (StochasticDiffEq.jl)     0.0025         30.668 s    -0.146    -0.021    -0.102    0.104
ShARK (Diffrax)                 0.00125        17.765 s    +0.007    -0.052    -0.130    0.081
EM (Diffrax)                    0.0002         38.748 s    +0.209    -0.047    -0.025    0.125
EM (StochasticDiffEq.jl)        0.0002        256.437 s    +0.163    -0.005    +0.049    0.098
EM (torchsde)                   0.0002        356.891 s    +0.368    -0.099    -0.076    0.225
```

The action, `phi2`, and `phi4` columns are signed relative differences from
the reference in percent.  The reference itself took 81.514 s.  The two
additional EM implementations use the step calibrated with Diffrax.  Their
full runs establish a measured 250 to 360 s scale for other general EM
implementations.  Both timings are GPU results and do not assert a universal
lower bound for every CPU or GPU code.

The consolidated values are stored in
`results/gpu_solver_nonequilibrium_calibrated.tsv`.  The reference and all
per-trajectory calibrated observables are stored together in
`results/gpu_solver_nonequilibrium_calibrated_metrics.npz`.

For 33,554,432 Philox normal samples, the measured mean magnitude was below
`4.4e-4`, variance was in `[0.9992, 1.0008]`, and cross-component correlation
magnitude was below `2.9e-4`.

## L=128, 1024 ensembles, 1000 steps

```bash
# benchmark_fixed_sciml.jl 已删除，以下数字为当时的测量结果
```

Sixteen paired wall-clock measurements alternated direct/SciML execution order.
Both sides include the initial-state copy, seed reset, reused 512 MiB buffers,
the full GPU solve, and synchronization.

| Measurement | Direct fixed | SciML + reused buffers |
|---|---:|---:|
| Median wall time | 0.574137 s | 0.574239 s |
| Median paired ratio (SciML/direct) | | 1.000215 |
| Paired-ratio IQR | | 0.999032--1.000731 |

All sixteen final arrays were elementwise identical.  The difference is inside
normal run-to-run variation: the SciML interface and direct fixed interface have
the same measured hot performance.

The default `ModelBFixedSOSRA()` path owns fresh buffers.  Its first call took
0.705292 s and its subsequent median was 0.575291 s.  Reuse is therefore
recommended for repeated large solves even though CUDA's memory pool makes the
hot allocation cost small.

For `tspan=(0,2)` and `saveat=0.5`, four L=128, 1024-ensemble CuArray snapshots
were retained.  Eight alternated pairs measured 0.574311 s for final-only output
and 0.574432 s with intermediate saving, a paired ratio of 1.000267.  Every
final state was elementwise identical.  The copies therefore add no measurable
compute overhead at this interval; their material cost is GPU memory.  Four
snapshots occupy 256 MiB, 192 MiB more than final-only output.

## Partial final-state accumulation

Stage 1 now stores
`u + dt*alpha1*k1 + sigma*dW` temporarily in `h02`.  Stage 2 evaluates `k2` in a
register, adds its final-state contribution into `u`, then overwrites `h02` with
the actual stage-3 input.  Stage 3 only reads `u` and `h02` and adds the `k3`
contribution.  Stage 1 deliberately does not modify `u`: its stencil reads
neighboring values across blocks, so changing `u` before the kernel ends would
introduce a race.

This removes the global `dW` and `k2` arrays.  At L=128 with 1024 ensembles the
buffers fall from 640 MiB to 512 MiB.  Before changing the production file, 12
alternated pairs measured the old path at 0.605480 s and the independent
candidate at 0.575713 s, a paired ratio of 0.950820 (4.92% faster).  After the
change, the production code and the still-independent candidate were
elementwise identical after 1, 10, and 1000 steps; their 12-pair timing ratio
was 0.999369.

The new Float32 accumulation order is not bitwise identical to the old final
sum.  Against the old implementation with the same random stream, the maximum
difference after 1000 steps was `3.70e-6`, RMS difference was `4.77e-7`, and
the largest mean-field change was `3.54e-8`.  The deterministic stage tests and
the HMC stationarity test above independently check that this is rounding from
an algebraically equivalent SOSRA update rather than a changed method.

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
| 32 | 0.043308 s | 0.043308 ms |
| 64 | 0.157446 s | 0.157446 ms |
| 128 | 0.574149 s | 0.574149 ms |

All three final states were finite, and the largest ensemble mean-field change
was below `1.6e-7`.

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
- Reusing `q` itself as `k2` was numerically valid but introduced a fragile
  same-address dependency.  The adopted scheme instead keeps `q`, holds `k2`
  only in a register, and uses `h02` only after its previous value has been read.
- A fresh thread/block sweep retained 768 threads and a 264-block cap for the
  40-register fused stage 1, and 1024 threads with the same cap for the two
  32-register stages.  Alternative thread counts were 0.4--13.7% slower and
  alternative common block caps were 9.5--18.6% slower in the complete solve.

## Supported scope

- Float32, `(L, L, Nensemble)` CuArray state, with power-of-two `L >= 4`
- fixed positive `dt` and forward `tspan` containing an integer number of steps
- final endpoint plus optional `dt`-aligned intermediate CuArray snapshots
- optional initial-state output with `save_start=true`
- optional explicit `seed`

Adaptive stepping, callbacks, dense output, non-power-of-two lattices, and
generic `SDEProblem` functions/noise are intentionally rejected.
