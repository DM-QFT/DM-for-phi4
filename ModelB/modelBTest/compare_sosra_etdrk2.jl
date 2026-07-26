using CUDA, JLD2, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Head-to-head at a fixed dt where both schemes should be essentially exact.
# Both start from the same 512 HMC equilibrium configurations and run to t=2000.
# Model B preserves exp(-S), so the t=0 HMC ensemble is itself the exact static
# reference: all three columns must agree.
#
# UV observables (phi2, phi4, action, |dS/dphi|^2, (lap phi)^2, S(k) top shells)
# decorrelate fast, so SOSRA and ETDRK2 are compared unpaired.
# IR observables (S(k) lowest shells) still remember the shared initial
# condition at t=2000, so the paired difference is far more sensitive there.

const L, Ncfg = 128, 512
const λ4, γ, T = 0.022f0, 1.0f0, 1.0f0
const dt = 0.002f0
const checkpoints = Float32[100, 500, 1000, 2000]

function load_hmc(κ)
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

const khat2 = [4.0f0 * (sinpi(i / L)^2 + sinpi(j / L)^2)
               for i in 0:(L ÷ 2), j in 0:(L - 1)]
# The lowest few k̂² take only 2-4 modes each, so they stay exact shells; the UV
# is averaged over wide bands, otherwise a single-mode shell is far too noisy.
const shells = let r = round.(khat2; digits=5), v = sort(unique(round.(khat2; digits=5)))
    vcat([("IR k̂²=$(round(x; digits=4))", r .== x) for x in v[2:5]],
         [("band $lo-$hi", (khat2 .> lo) .& (khat2 .<= hi))
          for (lo, hi) in ((0.05f0, 0.5f0), (1.0f0, 2.0f0), (4.0f0, 6.0f0),
                           (7.0f0, 8.0f0))])
end
const shell_masks = [m for (_, m) in shells]

function observables(u, κ)
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ4) - 8.0f0 * κ
    nn = u .* (circshift(u, (-1, 0, 0)) .+ circshift(u, (0, -1, 0)))
    lap = circshift(u, (1, 0, 0)) .+ circshift(u, (-1, 0, 0)) .+
          circshift(u, (0, 1, 0)) .+ circshift(u, (0, -1, 0)) .- 4 .* u
    dS = (-2.0f0κ) .* lap .+ m02 .* u .+ (4.0f0λ4) .* u .^ 3
    d(x) = Float64.(vec(Array(mean(x; dims=(1, 2)))))
    S = real.(Array(abs2.(rfft(u, (1, 2))))) ./ Float32(L^2)
    out = Dict{String, Vector{Float64}}(
        "M" => d(u),
        "phi2" => d(u .^ 2), "phi4" => d(u .^ 4),
        "action" => d(@. -2.0f0κ * nn + u^2 + λ4 * (u^2 - 1.0f0)^2),
        "nn" => d(nn), "dS2" => d(dS .^ 2), "lap2" => d(lap .^ 2))
    for ((name, _), mask) in zip(shells, shell_masks)
        out[name] = Float64.(vec(mean(S[mask, :]; dims=1)))
    end
    return out
end

function run_sosra(u, κ, seed)
    coef = modelb_coefficients(; γ, T, κ, λ=λ4)
    b = modelb_fixed_buffers(u; seed=UInt64(seed))
    out = Dict{Float32, Any}()
    t = 0.0f0
    for tc in checkpoints
        for _ in 1:round(Int, (tc - t) / dt); modelb_fixed_step!(u, b, dt, coef...); end
        CUDA.synchronize(); t = tc
        out[tc] = observables(u, κ)
        @printf("  SOSRA   t=%.0f done (seed %d)\n", tc, seed); flush(stdout)
    end
    return out, Array(u)
end

function run_etdrk2(u0, κ, seed)
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, copy(u0))
    key = (UInt32(seed), UInt32(0))
    out = Dict{Float32, Any}()
    t = 0.0f0; s = 0
    local u
    for tc in checkpoints
        n = round(Int, (tc - t) / dt)
        etdrk2_evolve!(uk, plan, n, key, dt; γ, T, κ, λ=λ4, step0=s)
        s += n; t = tc
        CUDA.synchronize()
        u = etd_state(plan, uk)
        out[tc] = observables(u, κ)
        @printf("  ETDRK2  t=%.0f done (seed %d)\n", tc, seed); flush(stdout)
    end
    return out, Array(u)
end

const keys_ordered = vcat(["phi2", "phi4", "action", "nn", "dS2", "lap2"],
                          [n for (n, _) in shells])
const seeds = (20260727, 815551)

se(x) = std(x) / sqrt(length(x))
zpair(a, b) = (d = b .- a; mean(d) / se(d))

# Two independent noise seeds per scheme: the seed-to-seed spread of one scheme
# is the yardstick for whether an ETDRK2-vs-SOSRA difference means anything.
function report(κ, o0, r)
    S1, S2 = r[(:sosra, 1)][2000.0f0], r[(:sosra, 2)][2000.0f0]
    E1, E2 = r[(:etdrk2, 1)][2000.0f0], r[(:etdrk2, 2)][2000.0f0]
    @printf("\nkappa=%.4f, L=%d, %d cfgs, dt=%.3f, t=2000, 2 seeds per scheme\n",
            κ, L, Ncfg, dt)
    @printf("%-18s %12s %12s %12s %18s %18s\n", "observable", "HMC t=0",
            "SOSRA", "ETDRK2", "seed-seed (SOSRA)", "ETDRK2 - SOSRA")
    for k in keys_ordered
        ms = (mean(S1[k]) + mean(S2[k])) / 2
        me = (mean(E1[k]) + mean(E2[k])) / 2
        @printf("%-18s %12.6f %12.6f %12.6f  %+7.3f%%(z%+5.1f) %+7.3f%%(z%+5.1f)\n",
                k, mean(o0[k]), ms, me,
                100 * (mean(S2[k]) - mean(S1[k])) / ms, zpair(S1[k], S2[k]),
                100 * (me - ms) / ms, zpair(vcat(S1[k], S2[k]), vcat(E1[k], E2[k])))
    end
    @printf("\ndrift vs the exact HMC reference (both seeds averaged)\n%-8s", "t")
    for k in ("phi2", "phi4", "action", "dS2", "IR k̂²=0.0024")
        @printf(" %20s", k)
    end
    println()
    for tc in checkpoints
        @printf("%-8.0f", tc)
        for k in ("phi2", "phi4", "action", "dS2", "IR k̂²=0.0024")
            f(a, b) = 100 * ((mean(a[tc][k]) + mean(b[tc][k])) / 2 - mean(o0[k])) /
                      mean(o0[k])
            @printf("  S%+6.3f%% E%+6.3f%%", f(r[(:sosra, 1)], r[(:sosra, 2)]),
                    f(r[(:etdrk2, 1)], r[(:etdrk2, 2)]))
        end
        println()
    end
    @printf("\nmax|ΔM| : SOSRA %.3e   ETDRK2 %.3e\n",
            maximum(abs, S1["M"] .- o0["M"]), maximum(abs, E1["M"] .- o0["M"]))
end

results, fields = Dict(), Dict()
for κ in (0.2705f0, 0.28f0)
    println("=== kappa=$κ ==="); flush(stdout)
    u0 = load_hmc(κ)
    o0 = observables(u0, κ)
    r = Dict()
    for (i, seed) in enumerate(seeds)
        r[(:sosra, i)], fields[(κ, :sosra, i)] = run_sosra(copy(u0), κ, seed)
        r[(:etdrk2, i)], fields[(κ, :etdrk2, i)] = run_etdrk2(u0, κ, seed)
    end
    report(κ, o0, r)
    results[κ] = (hmc=o0, runs=r)
    flush(stdout)
end

jldsave(joinpath(@__DIR__, "compare_sosra_etdrk2_t2000.jld2");
        results, fields, L, Ncfg, dt, checkpoints, lambda=λ4, seeds)
println("\nsaved compare_sosra_etdrk2_t2000.jld2")
