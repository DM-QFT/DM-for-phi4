# GH200/H100 power and utilization results

## Scope and measurement integrity

All measurements were made on `mg0939`, an NVIDIA GH200 120 GB (132 SM), with
`L=128`, 1024 ensembles, `Float32`, fixed `SOSRA()`, `adaptive=false`,
`dt=0.002`, and `t=2` unless a row is explicitly labelled as a microbenchmark.
No other compute process was present during the formal suite.

Every solver path completed one full-size, full-length prewarm outside the
measurement markers.  The CUDA memory-pool cache was retained, and the finite
and maximum reductions used in the measured loop were also precompiled.
Marker overhead beyond the sum of solver times was 0.10--0.12 s per complete
20--23 s run.  DCGM and `nvidia-smi` were sampled independently at 100 ms.

The production file remained unchanged:

```text
5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3  ModelB.jl
```

`nvidia-smi` reports requested/default/max GPU limits of 900 W, while DCGM's
actual enforced-power-limit field was **700 W** in every formal run.  Therefore
700 W, not 900 W, is the attainable GPU limit on this node.

## Solver results

Power columns are DCGM's rolling GPU power, which agrees with the independent
`nvidia-smi power.draw` stream.  Energy per solve is rolling mean GPU power
integrated over the complete marker window and divided by completed solves.
The GH200 total-energy counter is kept in the raw JSON but is not used here: on
this system it visibly includes roughly 90--110 W of Grace/module overhead.

| Solver path | t=2 time (s) | Speedup | GPU power mean / median / p95 / max (W) | GPU energy / solve (J) | Mean SM clock (MHz) | Power-cap time |
|---|---:|---:|---:|---:|---:|---:|
| Production sync RSwM3 | 5.1577 | 1.00x | 414.5 / 425.2 / 429.6 / 430.1 | 2151 | 1980 | 0 |
| Production drift, RSwM1 async | 4.3224 | 1.19x | 424.5 / 435.8 / 437.5 / 438.0 | 1845 | 1980 | 0 |
| Direct drift + fused noise, RSwM1 | 3.2853 | 1.57x | 413.9 / 424.3 / 428.2 / 429.4 | 1369 | 1980 | 0 |
| Shared-memory drift + fused noise, RSwM1 | 3.3197 | 1.55x | 414.0 / 424.3 / 427.7 / 431.0 | 1380 | 1980 | 0 |
| Direct + packed persistent cuRAND Philox, RSwM1 | 2.4266 | 2.13x | 534.9 / 549.4 / 556.1 / 556.6 | 1308 | 1980 | 0 |
| Custom fixed SOSRA + packed Philox | **0.6734** | **7.66x** | **681.3 / 698.7 / 699.6 / 700.0** | **461** | 1921 | **19.21 s** |

The time column reports hot inner timers.  The production timer includes its
per-call problem/noise/workspace/state-copy construction; the custom timer
reuses an allocated workspace and excludes `copyto!` and RNG reseeding.  As a
boundary check, complete marker wall time per solve (including reset, checks,
and logging) is 5.1905 s for production and 0.6764 s for custom, still 7.67x.
One-time custom-workspace allocation is outside the prewarmed formal window;
the intended repeated-trajectory use reuses that workspace.

Relative to production, the custom fixed-step path reduces wall time by 86.9%
and estimated GPU energy-to-solution by 78.6%.  Its rolling peak is 699.95 W;
the 100 ms instantaneous field briefly reaches 714 W, which is a normal
short-timescale excursion around a 700 W rolling limit.  The power-violation
counter increases for 19.21 s of its 20.29 s marker window and the SM clock
drops from 1980 to about 1921 MHz, directly confirming power-cap throttling.
Temperature peaks at 56 C and no thermal clock event is observed.

The packed-Philox SciML path is an important middle ground: it is 2.13x faster
and raises median power to 549 W without reaching the cap.  The direct/fused
drop-in path obtains 1.57x speedup at essentially unchanged power, so it cuts
energy even though it does not make the power number larger.  Shared-memory
tiling is 1.05% slower than direct addressing and provides no power advantage.

## Pipeline utilization

All values are DCGM time-averaged percentages inside the marker window.

| Path | GRACT | SM active | SM occupancy | DRAM active | FP32 active | INT active |
|---|---:|---:|---:|---:|---:|---:|
| Production sync | 93.4 | 81.8 | 52.3 | 23.5 | 2.1 | 77.6 |
| RSwM1 async | 95.8 | 82.3 | 50.0 | 23.5 | 2.5 | 76.7 |
| Direct + fused | 95.7 | 78.0 | 53.8 | 29.0 | 3.2 | 59.7 |
| Shared + fused | 96.6 | 79.1 | 54.8 | 28.8 | 3.1 | 59.4 |
| Direct + packed Philox | 95.8 | 93.0 | 72.7 | 39.6 | 4.3 | 84.5 |
| Custom fixed + Philox | **97.9** | **94.3** | **74.4** | **68.6** | **15.2** | 47.9 |

The baseline is already running at the maximum 1980 MHz and has high graphics
and SM activity.  Its low power is explained by its instruction mix: the
integer pipeline is active 77.6%, while FP32 and DRAM activity are only 2.1%
and 23.5%.  Thus 425 W does not mean that most of the H100 is idle.  The custom
path eliminates much of the launch/broadcast/synchronization overhead and
creates denser FP32 and memory work, which is what finally reaches the 700 W
cap.

## Negative and positive controls

| Control | Work unit | Time / unit | GPU power mean / median / p95 / max (W) | GRACT / SM active / occupancy | Result |
|---|---|---:|---:|---:|---|
| cuFFT spectral drift only | one drift | 0.7549 ms | 549.6 / 566.1 / 567.8 / 568.0 | 97.5 / 92.7 / 71.0% | High power but about 8x slower than the 0.0939 ms direct drift kernel |
| FP32 GEMM 16384² | one GEMM | 0.1689 s | 688.8 / 692.7 / 696.0 / 698.4 | 99.8 / 98.8 / 12.4% | Reaches the same 700 W cap; cap counter increases 18.20 s |

The GEMM control proves the node and telemetry can reach the enforced limit.
The FFT control proves the converse: drawing more power does not imply a better
Model-B solver.  Time-to-solution and energy-to-solution must remain the primary
metrics.

## Numerical scope

The synchronization removal, RSwM1 fixed/no-rejection path, direct-index drift,
and fused divergence/scale have a separate ten-step validation showing exact
elementwise agreement with production. Explicit Philox changes the random
stream and therefore is not expected to be bitwise identical; it is judged by
noise statistics and equilibrium observables. The custom fixed solver is the
algebraic constant-diffusion specialization of fixed SOSRA and has a separate
stage/noise/conservation test. The completed `L=128`, 512-configuration HMC
stationarity comparison gives an action relative drift of `+0.000306%`
(`z=0.004`) and no significant drift in `phi2`, `phi4`, or nearest-neighbor
observables; see `hmc_validation.md`.
This HMC/stability statement is limited to `kappa=0.2705`, `lambda=0.022`,
`gamma=temperature=1`, and `dt=0.002`; other parameters require a new stability
and equilibrium validation.

## Reproduction and raw data

- `run_power_case.sh`: launches one isolated workload and both telemetry streams.
- `summarize_power.py`: marker-bounded per-run statistics.
- `summarize_power_suite.py`: cross-run table generator.
- `power_summary.csv`: complete machine-readable formal table.
- `power_results/*`: raw DCGM, `nvidia-smi`, metadata, workload, and JSON for all
  eight formal cases.
- `power_results_preliminary*`: explicitly excluded harness-development runs;
  these are retained only for audit and are not read by the suite summarizer.

The formal suite can be regenerated with the commands in
`power_methodology.md`.
