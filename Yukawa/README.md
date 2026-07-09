---
license: mit
tags:
  - lattice-field-theory
  - diffusion-models
  - physics
---

# 2D Yukawa model: HMC configurations + score-based diffusion model

Scalar-field sector of the 2D staggered Yukawa model from Albergo et al.,
[arXiv:2106.05934](https://arxiv.org/abs/2106.05934), sampled with pseudofermion
HMC, plus a score-based diffusion model (VE-SDE, NCSN++-style 2D U-Net) trained
on the g=0.1 ensemble.

## Physics setup

16x16 lattice, two-flavor staggered fermions, m_f = 0.

| set | m^2 | lambda | g | <\|M\|> (HMC) |
|---|---|---|---|---|
| g=0.1 | -4.00 | 6.0 | 0.1 | 0.07326(17) |
| g=0.3 | -1.55 | 2.4 | 0.3 | — |

## Contents

- `YukawaFermionHMC2D.jl`, `generate_samples.jl`, `force_ratio.jl` — Julia
  pseudofermion HMC (sparse Cholesky solves; paper Table II force-ratio checks)
- `samples/yukawa_g{0.1,0.3}_L16_1000000.jld2` — 100k scalar configs each
  (key `configs`, shape `(100000, 16, 16)`; 1M trajectories, save_every=10),
  plus `.npz` mirrors and per-trajectory |M| histories
- `diffusion/` — PyTorch training/sampling/analysis for the diffusion model
  (`train_yukawa.py`, `sample_yukawa.py`, `analysis_observables.py`,
  `plot_chi_hist.py`; network/SDE definitions live in the parent DM repo)
- `diffusion/runs/yukawa_L16_g0.1_ncsnpp/models/` — log-spaced checkpoints
  (epochs 1–65, sigma=50, batch 256, lr 1e-3, bf16)
- `diffusion/runs/yukawa_L16_g0.1_ncsnpp/data/` — generated samples
  (`(16, 16, N)` npy, physical field units) and comparison figures

## Diffusion-model validation (epoch 49, EM 2000 steps, log schedule, N=10240)

| observable | HMC (N=100k) | diffusion model |
|---|---|---|
| <\|M\|> | 0.07326(17) | 0.07328(59) |
| chi = V(<M^2> - <\|M\|>^2) | 0.7469(36) | 0.759(14) |
