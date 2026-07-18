# H100/GH200 power benchmark methodology

This benchmark is isolated from `../ModelB.jl`.  The production file's expected
SHA-256 is
`5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3`.

The solver cases use `Float32`, `L=128`, 1024 simultaneous ensembles, fixed
`SOSRA()`, `adaptive=false`, `dt=0.002`, and `t=2` per solve.  One full-size,
full-length same-path solve is completed before `POWER_WINDOW_START`, and its
large CUDA memory-pool allocations are deliberately retained.  Compilation,
cold allocation, and initial clock ramp are therefore outside the marker.
Each case then uses a fixed repetition count spanning approximately 20 seconds.

`run_power_case.sh` samples at 100 ms with both DCGM and `nvidia-smi`.  DCGM
fields are:

- GPU power (rolling and instantaneous), total-energy counter, enforced power
  limit, and cumulative power-violation time;
- SM and memory clocks;
- graphics-engine activity, SM activity and occupancy;
- Tensor, DRAM, FP32, and integer-pipeline activity.

The line-receipt time is prepended to DCGM samples, and only rows between the
two Julia markers are retained.  GPU energy-to-solution is estimated from the
marker-bounded rolling GPU-power samples, which agree with the independent
`nvidia-smi power.draw` stream and avoid 100 ms aliasing of bursty instantaneous
samples.  The DCGM total-energy counter
is retained as a cross-check; on this GH200 it includes a visible module/Grace
overhead and must not be mislabelled as GPU-only energy.  Sampled power also
provides mean, median, p95, and peak statistics.
The `nvidia-smi` stream is an independent cross-check.

Run one case at a time on an otherwise idle GPU:

```bash
bash run_power_case.sh baseline_sync 20 2.0
bash run_power_case.sh rswm1_async 20 2.0
bash run_power_case.sh direct_fused_rswm1 20 2.0
bash run_power_case.sh direct_packed_philox_rswm1 20 2.0
bash run_power_case.sh custom_fused_philox 20 2.0
bash run_power_case.sh shared_fused_rswm1 20 2.0
bash run_power_case.sh fft_drift_only 20
bash run_power_case.sh gemm_control 20
python3 summarize_power_suite.py
```

The GEMM control uses `16384 x 16384` `Float32` matrices and explicitly
synchronizes after every `mul!`; this makes the requested duration a real GPU
wall-clock duration instead of an unbounded queue of asynchronous cuBLAS
launches.
