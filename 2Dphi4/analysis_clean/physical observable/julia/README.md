# Julia port — 2D φ⁴ physical observables

Julia counterpart of the Python observable scripts in the parent directory.
Computes the **physical observables** (not the figures, not the NN diagnostics).

## Files

| Julia | Python it mirrors | what it computes |
|-------|-------------------|------------------|
| `PhysicalObservables.jl` | (library) | all estimators — loading, action, global mag. observables + connected cumulants, single-site κₙ(x), U₄^site, geometric clusters, τ_int, binned-jackknife errors |
| `hmc_observables_table_2d.jl` | `hmc_observables_table_2d.py` | HMC-only table (⟨\|φ̄\|⟩, χ, ⟨S⟩/V, U₄, zero-mode κ⁽⁰⟩₂…₈, single-site κ^site, U₄^site) → CSV + txt |
| `dm_vs_hmc_observables_2d.py` → `dm_vs_hmc_observables_2d.jl` | `dm_vs_hmc_observables_2d.py` | HMC-vs-DM comparison summary at L=128 (global, site cumulants, U₄^site, κ₈(m), clusters, τ_int) → summary.txt |
| `site_cumulants_kstat.jl` | `site_cumulants_kstat.py` | single-site κ₂,₄ (unbiased k-statistics) and κ₆,₈ (jackknife-debiased) on raw cfgs |

## Run (from `2Dphi4/`)

```julia
julia "analysis_clean/physical observable/julia/hmc_observables_table_2d.jl"     # Ls=16,32,64
julia "analysis_clean/physical observable/julia/dm_vs_hmc_observables_2d.jl"      # L=128, reads DM .npy if present
julia "analysis_clean/physical observable/julia/site_cumulants_kstat.jl"         # L=128, κ=0.26,0.2705,0.28
```

Common flags: `--Ls 16 32 64`, `--kappas 0.26 0.2705 0.28`, `--n_bins 50`,
`--stride 5` (hmc_observables_table: keep every 5th cfg → matched-N HMC rows),
`--n_match 2048` (dm_vs_hmc), `--no_z2` (dm_vs_hmc).

## Conventions / notes

- **Layout**: ensembles are `(L, L, N)`, sample axis **last** — the native Julia
  layout of both the HMC `.jld2` (`jldopen(f)["cfgs"]`) and the DM `.npy`
  (`npzread`). Spatial axes are transposed vs the NumPy arrays, but every
  reported quantity is a spatial average / sum / cumulant, invariant under that
  transpose, so the scalars match Python.
- **`StatsBase.cumulant`** (Smith-1995 recursion, plug-in central moments)
  computes the connected cumulants: directly for the 1-D zero-mode magnetisation
  κ₁…₈ and per lattice site for κₙ(x). The jackknife error machinery
  uses a vectorised batch form of the same recursion (`cumulants_from_raw_moments`)
  because a per-site × per-jackknife-sample `StatsBase.cumulant` call would be
  10³–10⁶× slower — verified bitwise-identical.
- **Errors are binned jackknife only** (no bootstrap). The binning absorbs HMC
  autocorrelation; leave-one-bin-out propagates errors through the non-linear
  observables (χ, U₄, κₙ). The HMC table imposes Z₂ via even-moment formulas
  rather than augmenting the ensemble with {φ,−φ}.
- **Central values are the plug-in point estimate on the full sample.**
  Resampling only gives the error. A bootstrap mean used as the central value
  doubled the bias of κ₈. The HMC and DM rows of the `DM.tex` single-site
  cumulant table must use the same estimator (Z₂-even `persite_avg`) and the same N.
- **k-statistics** (`kstat_site`) have no Julia package, so the standard
  power-sum formulas are used (n=2,4).
- **Connected components** (clusters) use an internal flood-fill (4-connectivity,
  open BC), the stand-in for `scipy.ndimage.label`.
- **Not ported**: `beta_conf_*.py` (need the trained PyTorch NCSNpp model;
  no Julia equivalent), and all matplotlib figures (`joint_distribution_paper.py`,
  the plots inside `dm_vs_hmc_observables_2d.py`).

## Packages

`JLD2`, `NPZ`, `FFTW`, `StatsBase` (+ stdlib `Statistics`, `Random`, `Printf`,
`DelimitedFiles`). `NPZ` and `StatsBase` were added to the global v1.12 env.
