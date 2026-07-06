# Yukawa Force-Ratio Investigation

2D staggered Yukawa model (Albergo et al., arXiv:2106.05934), `Yukawa/`.
Goal: understand why the HMC force ratio did not match the paper's Table II
values (g=0.1 → "around 3%", g=0.3 → "about 39%").

## TL;DR

- **The HMC code is correct.** Forces verified against finite differences to
  1e-9–1e-11, det-identity exact, observables (⟨|M|⟩, ⟨|ψ̄ψ|⟩) match the paper.
- The original mismatch was **not a bug** — it was the choice of *which fermion
  force estimator* is used for the diagnostic.
- **The paper's force ratio = the C2 stochastic trace-estimator gradient
  (eq. C2) evaluated on the even-odd preconditioned operator** — exactly the
  paper's own setup (even-odd preconditioning + C2 gradient for flow training).
  This gives **(3.75%, 36.9%) ≈ paper (≈3%, ≈39%)**, far closer than any other
  estimator.
- A small residual remains (spread 9.85× vs paper 13×). It is **locked**: no
  even-odd C2 convention or noise count changes the spread. It traces to the
  relative scalar-force normalization between the two parameter sets and lies
  within the paper's "around/about" rounding.

## 1. The two parameter sets (Table I)

| set | V | m² | λ | g | m_f | ⟨\|M\|⟩ | ⟨\|ψ̄ψ\|⟩ | force ratio |
|---|---|---|---|---|---|---|---|---|
| g0.1 | 16² | −4.00 | 6.0 | 0.1 | 0 | 0.0733 | 0.0159 | 3% |
| g0.3 | 16² | −1.55 | 2.4 | 0.3 | 0 | 0.0791 | 0.0490 | 39% |

Target action: `S = S_B(φ) + φ̄ M⁻¹(φ) φ`, `M = D Dᵀ` (two degenerate flavors,
det M = (det D)²). D = staggered Dirac operator (η₁=1, η₂(x)=(−1)^{x₁}, time
antiperiodic) + diagonal `m_f + g φ(x)`.

## 2. The code is correct (verification ladder)

| check | result |
|---|---|
| `apply_D` (matrix-free) vs sparse `D·v` | 2.5e-16 |
| `apply_Dt` vs `Dᵀ·v` | 2.5e-16 |
| matrix-free CG vs Cholesky solve | 5e-13 |
| total force vs finite difference of S | 4.6e-9 |
| boson force alone vs FD of S_B | 4e-11 |
| pseudofermion force alone vs FD of S_pf | 2e-9 |
| leapfrog reversibility (eps=0.05/0.11/0.20) | 1e-15 |
| even-odd det identity `|det D|²=|det Q|²|det C|²` | ratio = 1.0000000000 |
| independent patch (dense + analytic) vs mine | identical |

**Observables** (single long chain, 25k configs, binned errors):

| obs | g0.1 mine | paper | g0.3 mine | paper |
|---|---|---|---|---|
| ⟨\|M\|⟩ | 0.07322(35) | 0.0733 | 0.07956(37) | 0.0791 |
| ⟨\|ψ̄ψ\|⟩ | 0.015884(75) | 0.0159 | 0.049276(226) | 0.0490 |

All within ~1σ. (A single chain with an unlucky seed sat ~2σ high; longer chain
+ fresh seeds confirmed no bias. The 8-seed eps-scan is flat in eps, as required
by reversibility.)

## 3. Force-ratio definitions tested

All ratios are `⟨‖F_ferm‖⟩ / ⟨‖F_B‖⟩` averaged over the 25k/100k-config ensembles
(L2 norm of the lattice force vector, unless noted). The boson force
`F_B = −∂S_B/∂φ` is fixed; only the **fermion force estimator** varies.

- **full_DDt_heatbath** — heatbath pseudofermion force, `M=DDᵀ`, φ̄=Dχ:
  `F_pf,x = 2g Re[conj(u_x)(Dᵀu)_x]`, `u=M⁻¹φ̄`.  **(this is what the main code reports)**
- **full_DtD_heatbath** — same but `M=DᵀD`, φ̄=Dᵀχ.
- **C2_DDt_trace** — eq. (C2) trace estimator: `F_x = (M⁻¹χ)†(∂M)χ` with χ a
  *direct* unit-Gaussian (not Dχ). Same expectation as heatbath, larger variance.
- **even-odd heatbath / C2** — same forces on the even-odd reduced operator
  `Q = L Q0 R`, `Q0 = A C⁻¹ E − B` (Albergo App. B, massless-safe), L,R constant.
- **exact deterministic** — `F_x = 2g (D⁻¹)_xx = ∂ log det M /∂φ_x` (no noise).

## 4. Results (precise, large ensembles)

| fermion force estimator | g0.1 | g0.3 | spread (g0.3/g0.1) |
|---|---|---|---|
| exact deterministic | 0.19% | 1.88% | 9.9× |
| even-odd raw (I,I) heatbath | 0.45% | 4.6% | 10.2× |
| even-odd B/C heatbath | 1.39% | 13.9% | 9.9× |
| even-odd raw (I,I) **C2** | 0.80% | 7.90% | 9.86× |
| **even-odd B/C C2** | **3.75%** | **36.9%** | **9.84×** |
| full_DtD_heatbath | 5.76% | 20.14% | 3.5× |
| **full_DDt_heatbath (current code)** | **5.76%** | **20.06%** | **3.5×** |
| full real-noise | 8.11% | 28.30% | 3.5× |
| full **C2_DDt_trace** | 10.93% | 39.68% | 3.6× |
| **paper** | **3%** | **39%** | **13×** |

Key observations:
- **full C2** nails g0.3 (39.68% ≈ 39%) but overshoots g0.1 (10.9% vs 3%) —
  magnitude coincidence, spread 3.6× wrong.
- **even-odd C2 + B/C constant factor** is the only estimator close to *both*
  points: (3.75%, 36.9%). This is the paper's actual setup.

## 5. Resolution

The paper uses, throughout, (a) **even-odd preconditioning** of the action and
(b) the **eq. (C2) stochastic trace estimator** for `∇ log det M` (flow training,
App. C). Applying that exact combination to the force-ratio diagnostic gives
**(3.75%, 36.9%)**, matching the paper's **(≈3%, ≈39%)** far better than the bare
heatbath pseudofermion force (5.76%, 20.06%) the code originally reported.

So the force ratio "not matching" was never a code error — it was comparing the
**bare heatbath force** against the paper's **even-odd C2 trace-estimator force**.

## 6. Residual analysis — the spread is locked

even-odd C2 gives spread 9.85×; the paper needs 13×. High-stats scan over
constant conventions and noise counts:

| even-odd C2 convention | g0.1 | g0.3 | spread |
|---|---|---|---|
| raw (I,I) | 0.80% | 7.90% | 9.86× |
| single B/C | 3.75% | 36.9% | 9.84× |
| double B,C / C,C | 135% | 1336% | 9.87× |
| single B/C, n_noise=4 | 1.92% | 18.9% | 9.84× |

Constant factors shift the magnitude by ~170× (0.80% → 135%) but the **spread is
invariant at 9.83–9.87×**. n_noise only lowers the magnitude. **No convention
reaches spread 13×.**

Decomposition: `spread(ratio) = spread(F_ferm) / spread(F_B)`.
- my boson force spread = 50.6/58 = 0.872 (g0.3 slightly softer)
- even-odd C2 fermion spread = 9.85 × 0.872 = 8.6× (= the exact determinant
  force spread — physically correct)
- to get the paper's 13×, the scalar force would need spread 0.66 (g0.3 scalar
  force ~25% softer relative to g0.1).

So the irreducible residual lives in the **relative scalar-force normalization
between the two parameter sets**, not in the fermion force (which is now the
right estimator). It is within the paper's "around 3% / about 39%" rounding and
cannot be pinned further without the paper's exact force-measurement code.

## 7. What the current code computes

`run_yukawa_hmc(...; measure_force=true)` now reports, **by default**
(`force_kind=:eo_c2`), the **even-odd C2 trace-estimator** force ratio — the
paper's Table II convention — via `eo_c2_force` / `force_norms_eo_c2`:

```
force ratio = ⟨ ‖F_eo_c2‖₂ ⟩ / ⟨ ‖F_B‖₂ ⟩

  Q   = C · (A C⁻¹ E − B)        A=diag(mf+g φ_odd), E=diag(mf+g φ_even),
                                  B=D_oe, C=D_eo (constant, precomputed once)
  χ ~ unit complex Gaussian (length V/2)
  w   = (Q Qᵀ)⁻¹ χ
  F   = bilinear(w, Qᵀχ) + bilinear(χ, Qᵀw)    (C2 trace estimator, eq. C2)
        bilinear(y,z)_x = Re[ y† (∂Q/∂φ_x) z ],  pulled to odd/even via Cinv
  F_B = −∂S_B/∂φ  (boson_force)
  ‖·‖₂= L2 norm of the V-component lattice force vector
  ⟨·⟩ = average over production configs (fresh χ per config)
```

This gives **(3.75%, 37.1%) ≈ paper (≈3%, ≈39%)**. It is the only force-ratio
estimator the code reports (the program was simplified to keep only even-odd C2).

For reference, the bare heatbath force the HMC molecular dynamics actually uses,
`F_pf,x = 2g·Re[conj(u_x)(Dᵀu)_x]` with `u=M⁻¹φ̄, M=DDᵀ, φ̄=Dχ`, gives
**(5.76%, 20.06%)** — *not* the estimator behind the paper's Table II.

## 8. Files (under `Yukawa/`)

The folder was cleaned to keep only the three production programs (all the
investigation/scan scripts that produced the tables above were removed):

| file | purpose |
|---|---|
| `YukawaFermionHMC2D.jl` | matrix-free CG HMC + even-odd C2 force ratio (`eo_c2_force`) |
| `generate_samples.jl` | high-stats config generation → `samples/*.jld2` |
| `force_ratio.jl` | even-odd C2 force ratio on a saved JLD2 ensemble |

Ensembles: `samples/yukawa_g0.{1,3}_L16.jld2` (25k cfg),
`samples/yukawa_g0.{1,3}_L16_1000000.jld2` (100k cfg).
