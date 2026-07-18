# HMC stationarity validation of the optimized fixed SOSRA

The source is the L=128 critical HMC chain
`DM/2Dphi4/trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2`.
We select 512 evenly spaced configurations from the stored chain and do **not**
subtract their individual means. This preserves the canonical mixture of
fixed-magnetization sectors required for a Model-B stationarity test.

Each method evolves every configuration to `t=5` with `dt=0.002`, using two
independent Philox repetitions. Errors are paired with each HMC initial state;
the quoted z score uses 64 ordered blocks to retain residual HMC-chain
correlation.

| Method | Observable | Relative drift | Paired z |
|---|---|---:|---:|
| custom fused Philox | action | +0.000306% | +0.004 |
| custom fused Philox | phi2 | -0.017045% | -0.525 |
| custom fused Philox | phi4 | -0.000759% | -0.011 |
| custom fused Philox | nearest-neighbor | -0.021997% | -0.597 |
| generic SciML packed Philox | action | +0.000452% | +0.007 |
| generic SciML packed Philox | phi2 | -0.017005% | -0.523 |
| generic SciML packed Philox | phi4 | -0.000657% | -0.009 |

No drift is statistically distinguishable from zero. The maximum per-
configuration magnetization-density change was `5.97e-8` for the custom path
and `6.68e-8` for SciML. Median solve times for this `N=512,t=5` validation
were 0.87056 s and 3.17864 s, respectively; HMC file I/O and CPU observables
are outside these solve timings.

The custom implementation algebraically simplifies constant diffusion terms.
It is exactly equal to the standalone SOSRA formula when supplied the same
simplified `dW/dZ` formula, while a direct comparison with SciML differs by a
few Float32 rounding units: for 32 ensembles, maximum/RMS field differences
were `4.77e-7/8.39e-8` after one step and `1.43e-6/2.38e-7` after ten steps.
This is arithmetic reassociation, not a stochastic-method or distribution
change; the HMC test above directly checks the physical consequence.

Reproduce with:

```bash
julia --project=. --threads=16 \
  h100_exploration_20260718/validate_fused_vs_hmc.jl 512 2 \
  h100_exploration_20260718/fused_vs_hmc.tsv

julia --project=. \
  h100_exploration_20260718/validate_custom_vs_sciml.jl 32 10
```

Machine-readable results are in `fused_vs_hmc.tsv`.
