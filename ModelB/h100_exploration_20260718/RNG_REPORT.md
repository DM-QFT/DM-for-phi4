# L128 × 1024 RNG exploration on GH200/H100

## Scope and reproducibility

- Workload: `Float32`, `(128,128,1024)`, fixed SOSRA `dt=0.002`, `t=2`,
  1000 accepted steps, `adaptive=false`.
- Full-solver comparisons use the already validated direct L128 drift and fused
  conserved-noise kernel, asynchronous callbacks, and RSwM1.  Thus the table
  isolates the RNG/noise-concurrency choice instead of re-measuring unrelated
  production overhead.
- Software: Julia 1.12.6, CUDA.jl 6.2.0, cuRAND 10.4.3,
  DiffEqNoiseProcess 5.33.1, StochasticDiffEq 7.1.1.
- Every solver row was run in a fresh Julia process.  Rep 1 includes residual
  allocation/large-shape compilation effects; `warm_median` is median(rep 2,3).
- Source production guard: `ModelB.jl` SHA256 remained
  `5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3`.

Raw machine-readable results are in `rng_microbench.tsv`,
`rng_statistics.tsv`, and `rng_solver_benchmark.tsv`.  Re-run with:

```bash
julia --project=. h100_exploration_20260718/benchmark_rng.jl micro 1024
julia --project=. h100_exploration_20260718/benchmark_rng.jl solver \
  packed_library_philox 2.0 3
```

## Result

The best simple candidate is a single packed allocation with an explicit
cuRAND Philox4x32-10 generator.  Its full-solver warm median is **2.433 s**,
versus **3.275 s** for the current split/default-RNG optimized solver: a
**25.7% time reduction (1.346× throughput)**.  Relative to the measured
5.165 s synchronous production-style baseline, the combined experimental
path is **2.12× faster**.

The reason is generator choice, not packing alone:

| candidate | two SOSRA-noise callbacks | full solver |
|---|---:|---:|
| current/default split | 1.177 ms | 3.275 s |
| packed/default | 1.181 ms | 3.303 s |
| packed/cuRAND Philox | 0.308 ms | 2.433 s |
| packed/cuRAND MRG32K3A | 0.409 ms | 2.513 s |
| packed/GPUArrays Philox | 0.343 ms | 2.442 s |

Packing the x/y vector fields changes two API calls into one but generates the
same number of normals, so with XORWOW/default it is neutral.  cuRAND Philox
is about 6.6× faster for the raw packed fill (0.0785 vs 0.5214 ms) and 3.8×
faster for two complete fill+divergence callbacks.  The remainder of the solve
is drift and SOSRA work, which limits the end-to-end gain to 25.7%.

## Statistical and conservation checks

For 33,554,432 raw normals per candidate, all usable generators had means
within `4.4e-4` of zero, variances in `[0.9992,1.0008]`, and x/y correlations
with magnitude below `2.9e-4`.  These deviations are consistent with the
sampling scale.

For the unscaled two-dimensional lattice divergence, the theoretical variance
is 4.  Measurements were `[3.9982,4.0029]`.  Its global mean was below
`1.8e-10` in magnitude.  The maximum absolute per-ensemble sum was
`9.16e-5` to `1.30e-4`; this is Float32 reduction roundoff over 16,384 sites,
not a lost zero mode.  Every full solve returned `Success`, took 1000 accepted
and zero rejected steps, stayed finite, and retained maximum absolute mass at
roughly `5e-4` to `7e-4` after the complete trajectory.

Changing RNG changes the bit stream, so final `maxabs` values are not supposed
to be bitwise equal.  These tests establish standard-normal moments,
cross-component independence, exact stencil semantics, finite integration,
and conserved zero mode.  Before treating the new stream as production ground
truth, final physical observables should still be compared to the HMC ensemble
with uncertainty bars; that is a scientific validation separate from this
performance benchmark.

## Concurrency and ordering experiments

- Two explicit default-RNG streams improve the full solver to 2.984 s (8.9%),
  but add events and two generators.  With Philox they give 2.444 s, slightly
  worse than the simple one-stream packed path.
- Double-buffer prefetch improves default RNG to 3.016 s (7.9%).  With Philox,
  generation is already short and prefetch/event overhead makes it 2.452 s.
- cuRAND Philox `default`, `best`, and `legacy` ordering all take about
  0.079 ms per packed fill.  `dynamic` measured 0.0729 ms (7% faster) in the
  raw microbenchmark.  A later specialized fixed-SOSRA full-solver timing gave
  medians of 0.657945 s (`dynamic`) and 0.673820 s (`default`), a 2.36% gain.
  It has not received the full statistical/HMC/power suite (nor a separate
  generic-SciML solver campaign), so it remains exploratory rather than the
  recommended result.
- cuRAND MTGP32 and MT19937 returned `CURAND_STATUS_TYPE_ERROR` for this
  device-array normal-fill path and were excluded.  CUDA.jl NativeRNG was
  slower than default (1.432 ms for two callbacks), matching its role as an
  exploratory rather than production generator.

## Important dispatch detail

In DiffEqNoiseProcess 5.33.1, the specialization
`wiener_randn!(rng, ::AbstractGPUArray)` calls `randn!(array)` and deliberately
does not pass `rng`.  Consequently, merely putting an explicit Philox object
in the existing callback does not select it.  The experimental callback calls
`Random.randn!(explicit_rng, packed_array)` directly.  The implementation is
in `rng_variants.jl`; `ModelB.jl` was not edited.

## Recommendation

For the next integrated experiment use `packed_noise_process128` with
`rng_kind=:library_philox`, `RSwM1`, one CUDA stream, and the fused divergence.
Do not add dual streams or double buffering for this generator.  Keep the
seed explicit in run metadata, because this path intentionally changes the
random-number stream relative to CUDA's default XORWOW path.
