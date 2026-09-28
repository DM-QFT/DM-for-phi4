# 2D SU(3) pure gauge theory (Wilson action) on an L×L periodic lattice.
# Cabibbo–Marinari heat bath (Kennedy–Pendleton SU(2) sampling) + over-relaxation,
# checkerboard-parallel over Julia threads; configurations streamed to HDF5.
#
# Action:     S = β Σ_x [1 − (1/3) Re Tr P(x)],  P(x) = U_1(x) U_2(x+1̂) U_1(x+2̂)† U_2(x)†
# One update  = 1 heat-bath sweep + n_or over-relaxation sweeps.
#
# Usage (run from this directory; -t sets the number of threads):
#   julia -t 8 su3_2d_heatbath.jl --beta 6.0 --L 16 --n_cfgs 10000 \
#         [--n_therm 500] [--n_skip 5] [--n_or 3] [--seed 1] [--out .]
# `include`-ing this file only defines the functions (SU3, plaquette, plaq_exact, ...); main() runs as a script.
#
# Output  <out>/cfgs_su3_beta=<β>_<L>^2.h5
#   cfgs : ComplexF64, Julia shape (3,3,2,L,L,n_cfgs) with cfgs[b,a,μ,x,y,n] = (U_μ(x,y))_{ab};
#          h5py therefore sees shape (n_cfgs, L, L, 2, 3, 3) with cfgs[n,y,x,mu,a,b] = (U_mu(x,y))_{ab}
#   plaq : Float64 (n_cfgs), average plaquette (1/V) Σ_x Re Tr P(x)/3 of each saved configuration
#   attrs: beta, L, n_therm, n_skip, n_or, seed, plaq_exact (infinite-volume exact value)

using LinearAlgebra, Random, Statistics, Printf, StaticArrays, HDF5, SpecialFunctions

const SU3 = SMatrix{3,3,ComplexF64,9}
const SU2 = SMatrix{2,2,ComplexF64,4}
const E = (CartesianIndex(1, 0), CartesianIndex(0, 1))

@inline limitbound(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end

@inline wrap(c, L) = CartesianIndex(limitbound.(Tuple(c), L))

# Sum of the two staples of U_μ(c): Re Tr(U_μ(c) Σ) = Σ of the two plaquettes containing the link.
function staple(U, μ, c, L)
    ν = 3 - μ
    cμ = wrap(c + E[μ], L)
    cν = wrap(c + E[ν], L)
    cν̄ = wrap(c - E[ν], L)
    cμν̄ = wrap(cμ - E[ν], L)
    return U[ν, cμ] * U[μ, cν]' * U[ν, c]' + U[ν, cμν̄]' * U[μ, cν̄]' * U[ν, cν̄]
end

# Project the (i,j) 2×2 block of W onto ℝ·SU(2):  block = k·V0 + (part with Re Tr(a·) = 0 ∀ a ∈ SU(2)).
@inline function su2_project(W, i, j)
    α = (W[i, i] + conj(W[j, j])) / 2
    β = (W[i, j] - conj(W[j, i])) / 2
    k = sqrt(abs2(α) + abs2(β))
    return k, SU2(α, -conj(β), β, conj(α)) / k
end

# SU(2) element a embedded in SU(3) at rows/columns (i,j).
@inline embed(a::SU2, i, j) =
    setindex(setindex(setindex(setindex(one(SU3), a[1, 1], i, i), a[1, 2], i, j), a[2, 1], j, i), a[2, 2], j, j)

# Kennedy–Pendleton: a0 ~ sqrt(1 − a0²) exp(α a0).
function kp_a0(rng, α)
    while true
        δ = -(log(rand(rng)) + cos(2π * rand(rng))^2 * log(rand(rng))) / α
        rand(rng)^2 <= 1 - δ / 2 && return 1 - δ
    end
end

# a0 + i a⃗·σ with |a⃗| = sqrt(1 − a0²) and a⃗ uniform on the sphere.
function su2_random(rng, a0)
    r = sqrt(1 - a0^2)
    cθ = 2rand(rng) - 1
    sθ = sqrt(1 - cθ^2)
    φ = 2π * rand(rng)
    a1 = r * sθ * cos(φ)
    a2 = r * sθ * sin(φ)
    a3 = r * cθ
    return SU2(a0 + im * a3, -a2 + im * a1, a2 + im * a1, a0 - im * a3)
end

# Gram–Schmidt on the rows; third row = conj(r1 × r2) fixes det = 1.
function reunit(V::SU3)
    r1 = SVector(V[1, 1], V[1, 2], V[1, 3])
    r1 = r1 / norm(r1)
    r2 = SVector(V[2, 1], V[2, 2], V[2, 3])
    r2 = r2 - dot(r1, r2) * r1
    r2 = r2 / norm(r2)
    r3 = conj(cross(r1, r2))
    return SU3(r1[1], r2[1], r3[1], r1[2], r2[2], r3[2], r1[3], r2[3], r3[3])
end

# One Cabibbo–Marinari step in the SU(2) subgroup (i,j): U ← A U with a ~ exp((β/3) Re Tr(a W)), W = U Σ,
# and in the subgroup Re Tr(a W) = k Re Tr(a V0) = 2k (a V0)_0.  Over-relaxation reflects about the maximum: a = (V0†)².
@inline function cm_step(V, W, i, j, β, rng, heatbath)
    k, V0 = su2_project(W, i, j)
    a = heatbath ? su2_random(rng, kp_a0(rng, 2β * k / 3)) * V0' : V0' * V0'
    A = embed(a, i, j)
    return A * V, A * W
end

# Literal subgroup indices let the compiler fold su2_project and embed; a loop over the three pairs is 1.7x slower.
function update_link!(U, μ, c, L, β, rng, heatbath)
    V = U[μ, c]
    W = V * staple(U, μ, c, L)
    V, W = cm_step(V, W, 1, 2, β, rng, heatbath)
    V, W = cm_step(V, W, 1, 3, β, rng, heatbath)
    V, W = cm_step(V, W, 2, 3, β, rng, heatbath)
    U[μ, c] = reunit(V)
end

# Checkerboard sweep: links with equal μ and equal parity of x+y share no plaquette → update in parallel.
function sweep!(U, L, β, rngs, heatbath)
    for μ in 1:2, parity in 0:1
        Threads.@threads :static for y in 1:L
            rng = rngs[Threads.threadid()]
            for x in 1:L
                (x + y) % 2 == parity || continue
                update_link!(U, μ, CartesianIndex(x, y), L, β, rng, heatbath)
            end
        end
    end
end

function plaquette(U, L)
    s = 0.0
    for c in CartesianIndices((L, L))
        P = U[1, c] * U[2, wrap(c + E[1], L)] * U[1, wrap(c + E[2], L)]' * U[2, c]'
        s += real(tr(P))
    end
    return s / (3L^2)
end

# ---------------------------------------------------------------------------------------------
# Exact result (2D, V → ∞): ⟨(1/3) Re Tr U_p⟩ = d ln Z1/dβ with the one-plaquette integral
#   Z1(β) = ∫_{SU(3)} dU exp((β/3) Re Tr U) = Σ_{q∈ℤ} det[ I_{i−j+q}(β/3) ]_{i,j=1..3}.
# Finite-L corrections are O(⟨P⟩^{L²}).
# ---------------------------------------------------------------------------------------------

# e^{-β} Z1(β) via the scaled besselix(n, x) = e^{-x} I_n(x); the q-sum converges once |q| ≫ sqrt(x).
function Z1_su3(β)
    x = β / 3
    Q = 20 + ceil(Int, 6sqrt(x))
    return sum(det(@SMatrix [besselix(i - j + q, x) for i in 1:3, j in 1:3]) for q in -Q:Q)
end

plaq_exact(β; h = 1e-3) = 1 + (log(Z1_su3(β + h)) - log(Z1_su3(β - h))) / 2h

# Mean, error and integrated autocorrelation time τ_int (window: first non-positive ρ(t)).
function mean_err_tau(x)
    n = length(x)
    m = mean(x)
    v = var(x)
    τ = 0.5
    for t in 1:min(n ÷ 2, 200)
        ρ = sum((x[i] - m) * (x[i+t] - m) for i in 1:n-t) / ((n - t) * v)
        ρ <= 0 && break
        τ += ρ
    end
    return m, sqrt(2τ * v / n), τ
end

# ---------------------------------------------------------------------------------------------

function main()
    args = Dict(lstrip(ARGS[i], '-') => ARGS[i+1] for i in 1:2:length(ARGS))
    β = parse(Float64, get(args, "beta", "6.0"))
    L = parse(Int, get(args, "L", "16"))
    n_cfgs = parse(Int, get(args, "n_cfgs", "10000"))
    n_therm = parse(Int, get(args, "n_therm", "500"))
    n_skip = parse(Int, get(args, "n_skip", "5"))
    n_or = parse(Int, get(args, "n_or", "3"))
    seed = parse(Int, get(args, "seed", "1"))
    out = get(args, "out", @__DIR__)
    mkpath(out)
    path = joinpath(out, "cfgs_su3_beta=$(β)_$(L)^2.h5")

    rngs = [Xoshiro(hash((seed, t))) for t in 1:Threads.maxthreadid()]
    U = fill(one(SU3), 2, L, L)                       # cold start
    function update!()
        sweep!(U, L, β, rngs, true)
        for _ in 1:n_or
            sweep!(U, L, β, rngs, false)
        end
    end
    P_exact = plaq_exact(β)

    println("=" ^ 64)
    println("2D SU(3) heat bath + over-relaxation")
    println("  β=$β, L=$L, n_cfgs=$n_cfgs, n_therm=$n_therm, n_skip=$n_skip, n_or=$n_or, seed=$seed")
    println("  threads: $(Threads.nthreads()),  exact ⟨P⟩ (V→∞) = $(round(P_exact, digits=6))")
    println("  output: $path  ($(round(288 * L^2 * n_cfgs / 1e9, digits=2)) GB)")
    println("=" ^ 64)

    t0 = time()
    for _ in 1:n_therm
        update!()
    end
    @printf("thermalisation done: plaq=%.6f  (%.1fs)\n", plaquette(U, L), time() - t0)

    plaq = zeros(n_cfgs)
    buf = Array{ComplexF64}(undef, 3, 3, 2, L, L, 1)
    h5open(path, "w") do f
        dset = create_dataset(f, "cfgs", ComplexF64, (3, 3, 2, L, L, n_cfgs); chunk = (3, 3, 2, L, L, 1))
        for n in 1:n_cfgs
            for _ in 1:n_skip
                update!()
            end
            plaq[n] = plaquette(U, L)
            # Transpose each 3×3 block so that row-major h5py reads cfgs[n, y, x, μ, a, b] = (U_μ(x,y))_{ab}.
            permutedims!(
                reshape(buf, 3, 3, 2, L, L),
                reshape(reinterpret(ComplexF64, U), 3, 3, 2, L, L),
                (2, 1, 3, 4, 5),
            )
            dset[:, :, :, :, :, n:n] = buf
            if n % max(1, n_cfgs ÷ 20) == 0
                @printf("  %7d/%d  plaq=%.6f  running mean=%.6f  (%.1fs)\n",
                    n, n_cfgs, plaq[n], mean(@view plaq[1:n]), time() - t0)
            end
        end
        f["plaq"] = plaq
        for (k, v) in ("beta" => β, "L" => L, "n_therm" => n_therm, "n_skip" => n_skip,
                       "n_or" => n_or, "seed" => seed, "plaq_exact" => P_exact)
            attrs(f)[k] = v
        end
    end

    m, err, τ = mean_err_tau(plaq)
    println("=" ^ 64)
    @printf("  ⟨P⟩ = %.6f ± %.6f  (τ_int = %.2f saved cfgs)\n", m, err, τ)
    @printf("  exact(V→∞) = %.6f   deviation = %.2f σ\n", P_exact, (m - P_exact) / err)
    @printf("  saved %d configurations to %s  (%.1fs)\n", n_cfgs, path, time() - t0)
    println("=" ^ 64)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
