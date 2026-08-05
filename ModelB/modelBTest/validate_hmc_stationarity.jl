using CUDA, JLD2, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Stationarity of the kappa=0.2705 HMC equilibrium ensemble under ETD.
# Model B leaves exp(-S) invariant (each trajectory conserves its own M, and the
# HMC ensemble already carries the equilibrium distribution of M), so evolving
# HMC configurations must not move any observable.  Protocol copied from
# checks/fixed_sciml_results.md: evenly spaced L=128 configurations, no mean
# subtraction, several independent noise repetitions, paired z scores.

const L, Ncfg, Nrep = 128, 512, 4
const κ, λ4, γ, T = 0.2705f0, 0.022f0, 1.0f0, 1.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)
const m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ4) - 8.0f0 * κ

function load_hmc()
    path = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                             "cfgs_wolff_fahmc_k=$(κ)_l=$(λ4)_$(L)^2.jld2"))
    jldopen(path, "r") do f
        n = Int(f["n_samples"])
        @assert f["N"] == L && isapprox(f["κ"], κ) && isapprox(f["λ"], λ4)
        data = JLD2.readmmap(JLD2.get_dataset(f, "cfgs"))
        idx = round.(Int, range(n / Ncfg, n; length=Ncfg))
        u = Array{Float32}(undef, L, L, Ncfg)
        for (i, j) in enumerate(idx); copyto!(@view(u[:, :, i]), @view(data[:, :, j])); end
        CuArray(u)
    end
end

# Per-configuration observables.  shifts are periodic, so circshift is exact.
function observables(u)
    nn = u .* (circshift(u, (-1, 0, 0)) .+ circshift(u, (0, -1, 0)))
    lap = circshift(u, (1, 0, 0)) .+ circshift(u, (-1, 0, 0)) .+
          circshift(u, (0, 1, 0)) .+ circshift(u, (0, -1, 0)) .- 4 .* u
    dS = (-2.0f0κ) .* lap .+ m02 .* u .+ (4.0f0λ4) .* u .^ 3
    d(x) = vec(Array(mean(x; dims=(1, 2))))
    uk = rfft(u, (1, 2))
    S(i, j) = vec(real.(Array(abs2.(uk[i, j, :])))) ./ Float32(L^2)
    return (action = d(@. -2.0f0κ * nn + u^2 + λ4 * (u^2 - 1.0f0)^2),
            phi2 = d(u .^ 2), phi4 = d(u .^ 4), nn = d(nn),
            dS2 = d(dS .^ 2),
            Sk_min = S(2, 1), Sk_quarter = S(L ÷ 4 + 1, 1),
            M = d(u))
end

function evolve_sosra(u, dt, seed; tmax=5.0f0)
    b = modelb_fixed_buffers(u; seed=UInt64(seed))
    for _ in 1:round(Int, tmax / dt); modelb_fixed_step!(u, b, dt, coef...); end
    CUDA.synchronize(); u
end

function evolve_etd(u, dt, seed; rk2=false, tmax=5.0f0)
    plan = etd_plan(similar(u))
    uk = etd_spectrum(plan, u)
    (rk2 ? etdrk2_evolve! : etd_evolve!)(
        uk, plan, round(Int, tmax / dt), (UInt32(seed), UInt32(0)), dt; γ, T, κ, λ=λ4)
    CUDA.synchronize()
    copy(etd_state(plan, uk))
end

const u0 = load_hmc()
const o0 = observables(u0)

# paired drift: average the per-configuration change over repetitions, then test
# whether its ensemble mean differs from zero
function drift(evolve, tmax)
    acc = Dict(k => zeros(Float64, Ncfg) for k in keys(o0))
    for rep in 1:Nrep
        ot = observables(evolve(copy(u0), 20260726 + rep, tmax))
        for k in keys(o0); acc[k] .+= Float64.(ot[k] .- o0[k]); end
    end
    out = Pair{Symbol, NTuple{2, Float64}}[]
    for k in keys(o0)
        d = acc[k] ./ Nrep
        z = mean(d) / (std(d) / sqrt(Ncfg))
        push!(out, k => (100 * mean(d) / mean(Float64.(o0[k])), z))
    end
    return out
end

function report(name, evolve, tmax)
    @printf("%-26s", name)
    for (k, (rel, z)) in drift(evolve, tmax)
        k === :M && continue
        @printf(" %8.4f%%(%+5.2f)", rel, z)
    end
    println()
end

names = [k for k in keys(o0) if k !== :M]
println("kappa=$κ HMC ensemble, L=$L, $Ncfg cfgs, $Nrep reps, relative drift %(paired z)")
@printf("%-26s%s\n", "", join((@sprintf(" %17s", string(k)) for k in names)))

sosra(dt) = (u, s, t) -> evolve_sosra(u, dt, s; tmax=t)
etd1(dt) = (u, s, t) -> evolve_etd(u, dt, s; tmax=t)
rk2(dt) = (u, s, t) -> evolve_etd(u, dt, s; rk2=true, tmax=t)

println("\n--- evolve to t=5 ---")
for dt in (0.002f0, 0.01f0, 0.02f0)
    report("SOSRA  dt=$dt", sosra(dt), 5.0f0)
end
for dt in (0.002f0, 0.01f0, 0.05f0)
    report("ETD1   dt=$dt", etd1(dt), 5.0f0)
end
for dt in (0.01f0, 0.05f0, 0.1f0)
    report("ETDRK2 dt=$dt", rk2(dt), 5.0f0)
end

println("\n--- evolve to t=100 (a wrong invariant measure saturates here) ---")
report("SOSRA  dt=0.002", sosra(0.002f0), 100.0f0)
report("SOSRA  dt=0.02", sosra(0.02f0), 100.0f0)
report("ETD1   dt=0.05", etd1(0.05f0), 100.0f0)
report("ETDRK2 dt=0.1", rk2(0.1f0), 100.0f0)

# magnetisation must not move at all
uk = etd_spectrum(etd_plan(similar(u0)), copy(u0))
plan = etd_plan(similar(u0))
uk = etd_spectrum(plan, copy(u0))
etd_evolve!(uk, plan, 2000, (UInt32(1), UInt32(0)), 0.05f0; γ, T, κ, λ=λ4)
@printf("\nETD1 t=100 dt=0.05: max|ΔM| = %.3e\n",
        maximum(abs, vec(Array(mean(etd_state(plan, uk); dims=(1, 2)))) .- o0.M))
