# Concurrent ensemble-shard results

Configuration: total shape `(128,128,1024)`, fixed SOSRA, `dt=0.002`, `t=2`,
direct drift, fused conserved-noise callback, RSwM1, asynchronous callbacks.
The total ensemble count is held fixed while 1, 2, 4, or 8 independent solver
tasks use non-blocking CUDA streams. Julia used four host threads. Slices are
prepared outside the timed region; each solver retains the same initial-state
copy cost as the one-shard case.

| RNG | Shards | Warm median | Relative to one shard |
|---|---:|---:|---:|
| implicit/default | 1 | 3.294272 s | 1.000x |
| implicit/default | 2 | 3.102490 s | 1.062x |
| implicit/default | 4 | 2.641159 s | 1.247x |
| implicit/default | 8 | **2.419754 s** | **1.361x** |
| persistent cuRAND Philox | 1 | 2.430631 s | 1.000x |
| persistent cuRAND Philox | 2 | 2.301258 s | 1.056x |
| persistent cuRAND Philox | 4 | **2.300749 s** | **1.056x** |
| persistent cuRAND Philox | 8 | 2.387284 s | 1.018x |

All 24 full solves were finite. Every shard accepted 1000 steps and the
largest absolute per-ensemble spatial-sum residual was below `7.83e-4`.

Sharding helps the slow default RNG because drift from one shard can overlap
RNG/solver phases from another. Once persistent Philox removes most RNG
latency, two streams capture nearly all remaining benefit; four is statistically
tied and eight adds contention. Even the best general-SciML sharded result
(`2.300749 s`) is about 3.42 times slower than the specialized five-operation
fixed-SOSRA path (`0.673414 s` in the sustained power run).

Recommendation: if retaining the generic SciML solver, use two Philox shards
only when the extra task/stream complexity is acceptable. For the fixed
`L=128` production workload, prefer the specialized solver. Raw output and
the reproducible driver are `ensemble_shards_raw.log` and
`benchmark_ensemble_shards.jl`.
