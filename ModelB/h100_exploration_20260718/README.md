# H100/GH200 Model-B optimization exploration

This directory is intentionally separate from `../ModelB.jl`.  The production
file is read-only for this study.

- Initial `ModelB.jl` SHA-256:
  `5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3`
- Hardware: NVIDIA GH200 120GB, compute capability 9.0, 132 SMs.
- Primary workload: `L=128`, `Nensemble=1024`, `Float32`, fixed SOSRA,
  `dt=0.002`, `t=2`.
- Primary metrics: wall time, site updates/s, GPU and module power,
  energy-to-solution, SM/occupancy/INT/FP32/Tensor/DRAM activity, and numerical
  agreement.

The study separates three questions:

1. Does a change preserve the lattice drift/noise and SOSRA step?
2. Does it reduce time-to-solution and energy-to-solution?
3. Does higher GPU power correspond to useful additional throughput?

The final Chinese summary and recommendation are in `FINAL_REPORT_CN.md`.

Power/utilization results are in `power_results_report.md`; the complete formal
table is `power_summary.csv`, and raw 100 ms telemetry is under `power_results/`.

## Fixed-step fused SOSRA prototype

The specialized five-operation fixed-step implementation and its reproducible
tests are in:

- `fused_fixed_sosra.jl`: packed Philox noise, dual dW/dZ kernel, and three
  fused SOSRA stage kernels;
- `validate_fused_fixed_sosra.jl`: CPU noise reference and one-step naive
  SOSRA comparison;
- `benchmark_fused_fixed_sosra.jl`: resource, kernel, short launch sweep, and
  full `t=2` benchmark modes;
- `fused_fixed_sosra_results.md`: exact validation and performance results.

Its measured median for the primary workload is 0.668668375 s for 1000 steps.
This narrow prototype does not implement adaptive rejection or noise bridges.

## Result index

- `FINAL_REPORT_CN.md`: overall decision and usage example;
- `power_results_report.md`, `power_summary.csv`, `power_results/`: formal
  power, energy, clocks, and pipeline telemetry;
- `hmc_validation.md`, `fused_vs_hmc.tsv`: L128 HMC stationarity check;
- `RNG_REPORT.md`: RNG, stream, prefetch, and full-solver comparison;
- `kernel_results.md`: direct/shared/fused/FFT kernel screening;
- `ensemble_shards_results.md`: concurrent ensemble shards;
- `maxregs_results.md`: launch/register-cap sweep;
- `trajectory_validation.md`: same-seed elementwise equivalence checks.
