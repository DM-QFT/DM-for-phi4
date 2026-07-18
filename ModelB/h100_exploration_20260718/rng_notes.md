# RNG and concurrency candidates

This note is based on the locally installed CUDA.jl 6.2.0,
DiffEqNoiseProcess 5.33.1, and StochasticDiffEq 7.1.1. No production file is
modified. The runnable candidates are in `rng_variants.jl`.

## Important dispatch finding

The current callback looks explicit but does not use the RNG it passes:

```julia
DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
```

For a GPU array, DiffEqNoiseProcess 5.33.1 selects its
`wiener_randn!(::AbstractRNG, ::AbstractGPUArray)` specialization. That method
ignores the first argument and calls `randn!(xi_x)` without an RNG. For a dense
Float32 `CuArray`, CUDA.jl then selects its cuRAND-library method backed by the
task-local `cuRAND.library_rng()`.

Consequences:

- The current ModelB path makes two cuRAND-library calls per noise callback.
- `CUDA.default_rng()` is actually CUDA.jl's GPUArrays Philox4x32-10 RNG, but
  it is not used by that call path.
- To compare generators honestly, call `Random.randn!(explicit_rng, array)`
  directly. `rng_variants.jl` does this.
- A `NoiseProcess`'s ordinary CPU RNG/reseed setting does not reseed the GPU
  RNG captured by these experimental callbacks. Rebuild the noise process for
  each independently seeded benchmark/trajectory.

## Candidates implemented

### 1. One packed x/y call

`packed_noise_process128` allocates a dense `(128,128,Nensemble,2)` buffer.
The two vector components are its contiguous halves, so one `randn!` call
generates both. The divergence kernel reads the two halves directly and also
fuses the following scale or bridge operation.

At `Nensemble=1024`, one component has `2^24` Float32 values and the packed
buffer has `2^25`. This matters because CUDA.jl's cuRAND-library normal fill
pads non-power-of-two lengths to the next power of two; the target shape needs
no padding or temporary copy.

One packed call is distributionally valid: x and y consume disjoint parts of
one pseudorandom sequence. It need not be bitwise identical to two separate
cuRAND calls because generator ordering may depend on call geometry.

### 2. Explicit generator sweep

`make_gpu_rng` supports:

| name | implementation | role |
|---|---|---|
| `library_default` | cuRAND pseudo default | exact API family used today |
| `library_xorwow` | cuRAND XORWOW | likely fastest legacy control |
| `library_philox` | cuRAND Philox4x32-10 | counter-based library candidate |
| `library_mrg32k3a` | cuRAND MRG32k3a | independent library comparison |
| `library_mtgp32` | cuRAND MTGP32 | additional pseudorandom control |
| `library_mt19937` | cuRAND MT19937 | additional pseudorandom control |
| `gpuarrays_philox` | CUDA.jl/GPUArrays Philox4x32-10 kernel | may trade more SM work for library overhead |
| `native_philox2` | CUDA.jl NativeRNG Philox2x32 | exploratory control only |

For a cuRAND `LibraryRNG`, `rng_ordering` can additionally be `best`,
`default`, `seeded`, `legacy`, or `dynamic`. Not every generator/order pair is
guaranteed to be supported, so the benchmark driver should catch and report an
unsupported pair rather than abort the sweep. First test the generator's
unchanged ordering (`nothing`), then `best` and `dynamic`; expand the ordering
sweep only if either improves throughput. Quasi/Sobol generators are omitted
because substituting quasi-random values for independent Wiener increments is
a change of statistical method, not a performance-only optimization.

CUDA.jl documents NativeRNG as a testing/performance-comparison mechanism and
recommends GPUArrays RNG for production kernel-based generation, so do not
select `native_philox2` merely because a microbenchmark is favorable.

Changing generator changes trajectories. Validate equilibrium observables,
normal moments, x/y cross-correlation, and conserved zero mode; do not require
sample-by-sample equality.

### 3. Concurrent x/y fills

`dual_stream_noise_process128` uses two independent generators, two
non-blocking streams, and CUDA events. It launches the x and y normal fills
concurrently, then places device-side waits on the solver's stream before the
divergence kernel. A consumed event prevents either producer from overwriting
a field while the previous divergence still reads it.

This can increase instantaneous power yet still be slower: each large normal
fill may already occupy the same SM/memory resources, and two generators add
event and launch overhead. Treat it as an experiment, not the default choice.

### 4. Double-buffered RNG prefetch

`prefetch_packed_noise_process128` keeps two packed fields. Raw standard
normals do not depend on `u`, `dt`, `q`, or `h`, so it generates the next field
on a producer stream while the main stream performs the current divergence and
the rest of the SOSRA work. Ready/consumed events enforce both dependencies.

This is more promising than overlapping x and y fills because RNG work can
overlap different integer/broadcast/stencil work rather than contend with an
identical RNG kernel. Cost: one extra packed field, or 128 MiB for the target
`L=128, Nensemble=1024, Float32` workload. The constructor primes two fields;
end-to-end timing must include this startup work, while steady-state timing
should also be reported separately.

### 5. Independent ensemble shards

`run_ensemble_shards` divides the third dimension and runs each shard in a
separate Julia task and non-blocking CUDA stream. Each task has a distinct
seed. Start with 2 shards and then 4. Do not concatenate full solution objects
during timing; the helper returns a vector of shard solutions.

Time evolution within a trajectory cannot be parallelized this way. The gain
comes from phase desynchronization: one shard may generate random values while
another runs drift or SOSRA broadcasts. This can raise useful concurrency and
power. Risks are higher total live memory, more kernel launches, allocator
contention, and lower efficiency of each smaller kernel.

## Benchmark matrix and acceptance rules

Run the following in order, with warm-up excluded and with identical target
work (`L=128`, total ensembles 1024, Float32):

1. RNG-only fill throughput:
   current two fills versus one packed fill, for all explicit RNG candidates.
2. Whole noise callback:
   current, packed, dual-stream, and packed-prefetch variants.
3. Short SOSRA integration (`dt=0.002`, enough steps to reach steady state).
4. Full `t=2` solve for the survivors.
5. Shards = 1, 2, 4, first with the best ordinary RNG and then with prefetch.

For multi-stream tests, a timing event on only the main stream is insufficient.
Synchronize all involved streams/device at each measured boundary. Record:

- wall time and completed ensemble-steps/s;
- average and peak GPU/module power;
- joules per completed ensemble-step and energy-to-solution;
- SM active, occupancy, DRAM active, INT/FP32 activity;
- allocation footprint and any OOM/reclaim behavior.

Select by time-to-solution first and energy-to-solution second. Higher watts
without higher completed-work throughput is contention, not improved H100
utilization.

Numerical/statistical checks:

- every output is finite and ensemble means remain conserved to tolerance;
- packed/split divergence kernels agree for identical supplied xi arrays;
- generated normal mean/variance and x-y correlation are acceptable over a
  large sample;
- final action, magnetization/zero mode, two-point or structure-factor
  observables agree across multiple seeds within Monte Carlo uncertainty;
- HMC ground-truth comparisons use the same observable definitions and enough
  independent samples to quote error bars.

## Expected priority

The recommended test priority is:

1. packed one-call + fused divergence/scale;
2. packed double-buffer prefetch;
3. 2-way ensemble sharding;
4. explicit XORWOW/Philox/GPUArrays sweep within the best layout;
5. dual x/y streams;
6. 4-way sharding and combined shard+prefetch only if memory remains safe.

Minimal integration with `common.jl`:

```julia
include("common.jl")
include("rng_variants.jl")
using .H100RNGVariants

packed_builder = (t0, u) -> packed_noise_process128(
    t0, u;
    rng_kind=:library_xorwow,
    rng_ordering=:best,
    seed=20260718,
    synchronize_callbacks=false,
)
sol = solve_sosra(
    u0;
    noise_builder=packed_builder,
    synchronize_callbacks=false,
    dt=0.002f0,
)
```

Replace `packed_noise_process128` with
`prefetch_packed_noise_process128` or
`dual_stream_noise_process128` for those candidates. After any multi-stream
solve, use `CUDA.device_synchronize()` at the measurement boundary.

Do not optimize for reaching the 700 W ceiling. A useful variant should raise
power only because it performs more completed work concurrently; its decisive
result is lower seconds and joules for the same statistical workload.
