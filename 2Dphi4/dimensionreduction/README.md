---
license: cc-by-4.0
tags:
- physics
- lattice-field-theory
- monte-carlo
- phase-transition
pretty_name: 2D phi^4 lattice field configurations
size_categories:
- 10K<n<100K
---

# 2D φ⁴ lattice field configurations

Monte Carlo configurations of the two-dimensional scalar φ⁴ theory on periodic square
lattices, generated for training and benchmarking generative models (score-based
diffusion), and for unsupervised-learning studies of the phase transition (PCA).

## Physics

Lattice action in the hopping-parameter form

$$S[\phi] = \sum_x \Big[ -2\kappa \sum_{\mu=1,2} \phi_x \phi_{x+\hat\mu} + (1-2\lambda)\phi_x^2 + \lambda \phi_x^4 \Big]$$

with quartic coupling fixed at λ = 0.022. The three hopping parameters straddle the
phase transition (κ_c ≈ 0.2708 for this λ):

| κ | phase |
|---|---|
| 0.26 | symmetric (disordered) |
| 0.2705 | near-critical |
| 0.28 | broken (ordered) |

## Files

15 `.npz` files, one per (κ, L) pair:

```
cfgs_wolff_fahmc_k=<kappa>_l=0.022_<L>^2.npz,  L ∈ {8, 16, 32, 64, 128}
```

Each file contains 10240 independent configurations. Keys:

| key | content |
|---|---|
| `cfgs` | float64 array, shape (10240, L, L) |
| `κ`, `λ` | hopping parameter, quartic coupling |
| `N`, `n_samples` | lattice size L, number of configurations |
| `acc_rate` | HMC acceptance rate |
| `ε_final` | final tuned HMC step size |

Total size ≈ 5 GB (the L=128 files are 1.3 GB each).

## Generation

Sampled with alternating Wolff cluster updates and Fourier-accelerated HMC
(`wolff_fahmc`), which keeps autocorrelations short even near criticality.
Configurations within a file are saved after thermalization and thinned so they are
effectively independent.

## Usage

```python
import numpy as np

d = np.load("cfgs_wolff_fahmc_k=0.2705_l=0.022_32^2.npz")
phi = d["cfgs"]            # (10240, 32, 32)
M = phi.mean(axis=(1, 2))  # per-configuration magnetization
```

`pca_phi4.py` reproduces the PCA analysis figures (`pca_phi4_L32.png`,
`pca_phi4_L128.png`): the PCA spectrum equals the sorted momentum-space propagator
G(k) = ⟨|φ(k)|²⟩, PC1 = L·M is the magnetization mode, and PC2–PC5 span the
lowest-momentum plane-wave quadruplet.
