# 2D SU(2) pure gauge theory (Wilson action) on an L×L periodic lattice.
# Kennedy–Pendleton heat bath + over-relaxation, checkerboard-parallel over Julia threads;
# configurations streamed to HDF5.
#
# Action:     S = β Σ_x [1 − (1/2) Re Tr P(x)],  P(x) = U_1(x) U_2(x+1̂) U_1(x+2̂)† U_2(x)†
# One update  = 1 heat-bath sweep + n_or over-relaxation sweeps.
#
# Usage (run from this directory; -t sets the number of threads):
#   julia -t 8 su2_2d_heatbath.jl --beta 2.0 --L 16 --n_cfgs 10000 \
#         [--n_therm 500] [--n_skip 5] [--n_or 3] [--seed 1] [--out .]
# `include`-ing this file only defines the functions (SU2, plaquette, plaq_exact, ...); main() runs as a script.
#
# Output  <out>/cfgs_su2_beta=<β>_<L>^2.h5
#   cfgs : ComplexF64, Julia shape (2,2,2,L,L,n_cfgs) with cfgs[b,a,μ,x,y,n] = (U_μ(x,y))_{ab};
#          h5py therefore sees shape (n_cfgs, L, L, 2, 2, 2) with cfgs[n,y,x,mu,a,b] = (U_mu(x,y))_{ab}
#   plaq : Float64 (n_cfgs), average plaquette (1/V) Σ_x Re Tr P(x)/2 of each saved configuration
#   attrs: beta, L, n_therm, n_skip, n_or, seed, plaq_exact (infinite-volume exact value)

using LinearAlgebra, Random, Statistics, Printf, StaticArrays, HDF5, SpecialFunctions

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

# Any real combination of SU(2) matrices is k·V0 with V0 ∈ SU(2), k = sqrt(det) ≥ 0.  The symmetrised
# (α, β) also pull a link that has drifted off the group by rounding back onto it.
@inline function su2_project(W)
    α = (W[1, 1] + conj(W[2, 2])) / 2
    β = (W[1, 2] - conj(W[2, 1])) / 2
    k = sqrt(abs2(α) + abs2(β))
    return k, SU2(α, -conj(β), β, conj(α)) / k
end

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

# With Σ = k V0 the weight is exp((β/2) Re Tr(U Σ)) = exp(β k (U V0)_0), so the heat bath draws
# U = a V0† with a ~ KP(βk), independent of the old link.  Over-relaxation reflects about the
# maximum V0†:  U ← V0† U† V0†, which leaves Re Tr(U Σ) unchanged.
function update_link!(U, μ, c, L, β, rng, heatbath)
    k, V0 = su2_project(staple(U, μ, c, L))
    if heatbath
        V = su2_random(rng, kp_a0(rng, β * k)) * V0'
    else
        V = V0' * U[μ, c]' * V0'
    end
    U[μ, c] = su2_project(V)[2]
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
    return s / (2L^2)
end

# Exact result (2D, V → ∞): ⟨(1/2) Re Tr U_p⟩ = d ln Z1/dβ with the one-plaquette integral
#   Z1(β) = ∫_{SU(2)} dU exp((β/2) Re Tr U) = (2/π) ∫_{-1}^{1} sqrt(1 − a0²) e^{β a0} da0 = 2 I_1(β)/β,
# hence ⟨P⟩ = I_2(β)/I_1(β).  Finite-L corrections are O(⟨P⟩^{L²}).
plaq_exact(β) = besselix(2, β) / besselix(1, β)

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
    β = parse(Float64, get(args, "beta", "2.0"))
    L = parse(Int, get(args, "L", "16"))
    n_cfgs = parse(Int, get(args, "n_cfgs", "10000"))
    n_therm = parse(Int, get(args, "n_therm", "500"))
    n_skip = parse(Int, get(args, "n_skip", "5"))
    n_or = parse(Int, get(args, "n_or", "3"))
    seed = parse(Int, get(args, "seed", "1"))
    out = get(args, "out", @__DIR__)
    mkpath(out)
    path = joinpath(out, "cfgs_su2_beta=$(β)_$(L)^2.h5")

    rngs = [Xoshiro(hash((seed, t))) for t in 1:Threads.maxthreadid()]
    U = fill(one(SU2), 2, L, L)                       # cold start
    function update!()
        sweep!(U, L, β, rngs, true)
        for _ in 1:n_or
            sweep!(U, L, β, rngs, false)
        end
    end
    P_exact = plaq_exact(β)

    println("=" ^ 64)
    println("2D SU(2) heat bath + over-relaxation")
    println("  β=$β, L=$L, n_cfgs=$n_cfgs, n_therm=$n_therm, n_skip=$n_skip, n_or=$n_or, seed=$seed")
    println("  threads: $(Threads.nthreads()),  exact ⟨P⟩ (V→∞) = $(round(P_exact, digits=6))")
    println("  output: $path  ($(round(128 * L^2 * n_cfgs / 1e9, digits=2)) GB)")
    println("=" ^ 64)

    t0 = time()
    for _ in 1:n_therm
        update!()
    end
    @printf("thermalisation done: plaq=%.6f  (%.1fs)\n", plaquette(U, L), time() - t0)

    plaq = zeros(n_cfgs)
    buf = Array{ComplexF64}(undef, 2, 2, 2, L, L, 1)
    h5open(path, "w") do f
        dset = create_dataset(f, "cfgs", ComplexF64, (2, 2, 2, L, L, n_cfgs); chunk = (2, 2, 2, L, L, 1))
        for n in 1:n_cfgs
            for _ in 1:n_skip
                update!()
            end
            plaq[n] = plaquette(U, L)
            # Transpose each 2×2 block so that row-major h5py reads cfgs[n, y, x, μ, a, b] = (U_μ(x,y))_{ab}.
            permutedims!(
                reshape(buf, 2, 2, 2, L, L),
                reshape(reinterpret(ComplexF64, U), 2, 2, 2, L, L),
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
