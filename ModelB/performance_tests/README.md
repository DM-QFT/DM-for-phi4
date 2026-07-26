# Model B solver performance tests

These scripts inspect the existing solver without editing it.

```bash
julia --startup-file=no performance_tests/benchmark_fixed.jl
julia --startup-file=no performance_tests/profile_active.jl
julia --startup-file=no performance_tests/benchmark_stage_count.jl
ncu --kernel-name regex:modelb_fixed_stage --launch-skip 3 --launch-count 3 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  julia --startup-file=no performance_tests/ncu_active.jl
```

`benchmark_fixed.jl` separates first-call compilation from hot throughput and
checks whether equal element counts depend on lattice shape.
`profile_active.jl` attributes GPU time to random-number generation and the
three SOSRA stages at the `run_randn.jl` size.

`benchmark_stage_count.jl` compares the production SOSRA workload with a
benchmark-only Euler--Maruyama step for the same spatial SDE. The Euler result
is not a proposed production solver and is not an accuracy comparison; it only
measures the cost of one drift evaluation and two link-noise fields versus
SOSRA's three drift evaluations and four link-noise fields.
