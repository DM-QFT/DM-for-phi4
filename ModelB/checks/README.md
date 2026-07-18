# Model B active configuration

The active implementation has one numerical path only:

```julia
alg = SOSRA()
adaptive = false
dt = 0.005f0
```

The state has shape `(L, L, Nensemble)`, so all ensemble members evolve in one
GPU solve while the spatial operators act only on the first two dimensions. It
targets the `DM/2Dphi4` lattice action at the L=128 critical parameters
`kappa=0.2705`, `lambda=0.022`, `T=1`, `dx=gamma=1`. The conserved noise
process always includes the auxiliary `Z` integral required by SOSRA.

## Active files

- `../ModelB.jl`: compact fixed-step SOSRA implementation.
- `test_sosra.jl`: GPU smoke test for the fixed algorithm, step count,
  finiteness, and conserved mean field.
- `l128_sosra_dt005_validation.tsv`: compact HMC validation summary.

The L=128 HMC validation gives an action drift of
`-0.1127 +/- 0.0359%`, a `phi2` drift of `-0.0222 +/- 0.0181%`, and a
common-noise field RMSE of `0.0543%`. The maximum mean-field conservation
error was `4.46e-8`.

## Run

```bash
zsh -ic 'nenv2 && julia --startup-file=no checks/test_sosra.jl'
```

Minimal use from Julia:

```julia
using Statistics
include("ModelB.jl")
u0 = CUDA.randn(Float32, 128, 128, 32)
u0 .-= mean(u0; dims=(1, 2))
sol = modelB_2d_SDE_prob(u0; tspan=(0.0f0, 1.0f0), saveat=1.0f0)
```

Historical solver comparisons and generated artifacts were removed from the
active tree and preserved for recovery at:

```text
/work/gw97/w97000/project/DM/ModelB_obsolete_20260718
```

The HMC ground-truth data under `DM/2Dphi4/trainingdata` was not moved or
modified.
