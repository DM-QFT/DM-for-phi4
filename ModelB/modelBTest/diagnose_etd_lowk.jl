using CUDA, JLD2, Statistics, Printf

include(joinpath(@__DIR__, "modelb_etd.jl"))

# ETDRK2's S(k) at the four lowest momenta drifts away from the exact HMC
# reference at kappa=0.2705 while SOSRA tracks it.  Two candidates:
#   (a) an O(dt) splitting error -- the bare linear rate of those modes is
#       negative (spinodal) while the true relaxation rate is positive, so the
#       exactly-integrated linear part fights a lagged phi^3, or
#   (b) an implementation bug, which would be dt-independent.
# Scan dt over 8x, and compare ETD1 against ETDRK2 to see whether the corrector
# is involved.

const L, Ncfg = 128, 512
const κ, λ4, γ, T = 0.2705f0, 0.022f0, 1.0f0, 1.0f0
const tmax = 2000.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)

function load_hmc()
    path = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                             "cfgs_wolff_fahmc_k=$(κ)_l=$(λ4)_$(L)^2.jld2"))
    jldopen(path, "r") do f
        n = Int(f["n_samples"])
        data = JLD2.readmmap(JLD2.get_dataset(f, "cfgs"))
        idx = round.(Int, range(n / Ncfg, n; length=Ncfg))
        u = Array{Float32}(undef, L, L, Ncfg)
        for (i, j) in enumerate(idx); copyto!(@view(u[:, :, i]), @view(data[:, :, j])); end
        CuArray(u)
    end
end

const khat2 = [4.0f0 * (sinpi(i / L)^2 + sinpi(j / L)^2)
               for i in 0:(L ÷ 2), j in 0:(L - 1)]
const rk = round.(khat2; digits=5)
const shells = let v = sort(unique(rk))
    vcat([("k̂²=$(round(x; digits=4))", rk .== x) for x in v[2:5]],
         [("band 1-2", (khat2 .> 1) .& (khat2 .<= 2)),
          ("band 7-8", (khat2 .> 7) .& (khat2 .<= 8))])
end

function measure(u)
    S = real.(Array(abs2.(rfft(u, (1, 2))))) ./ Float32(L^2)
    out = Dict{String, Vector{Float64}}("phi2" => Float64.(vec(Array(mean(u .^ 2; dims=(1, 2))))))
    for (name, mask) in shells
        out[name] = Float64.(vec(mean(S[mask, :]; dims=1)))
    end
    return out
end

run_sosra(u0, dt) = begin
    u = copy(u0); b = modelb_fixed_buffers(u; seed=UInt64(20260727))
    for _ in 1:round(Int, tmax / dt); modelb_fixed_step!(u, b, dt, coef...); end
    CUDA.synchronize(); measure(u)
end

function run_etd(u0, dt; rk2)
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, copy(u0))
    (rk2 ? etdrk2_evolve! : etd_evolve!)(
        uk, plan, round(Int, tmax / dt), (UInt32(20260727), UInt32(0)), dt;
        γ, T, κ, λ=λ4)
    CUDA.synchronize(); measure(etd_state(plan, uk))
end

const u0 = load_hmc()
const ref = measure(u0)
const names = vcat("phi2", [n for (n, _) in shells])

@printf("kappa=%.4f, L=%d, %d cfgs, t=%.0f; drift of each quantity from the\n", κ, L, Ncfg, tmax)
@printf("exact HMC t=0 ensemble (statistical error on one column ~4%% for the\n")
@printf("lowest shells, 0.17%% for phi2)\n\n")
@printf("%-22s", "scheme")
for n in names; @printf(" %12s", n); end
println()

function row(label, o)
    @printf("%-22s", label)
    for n in names
        @printf(" %11.3f%%", 100 * (mean(o[n]) - mean(ref[n])) / mean(ref[n]))
    end
    println(); flush(stdout)
end

row("SOSRA   dt=0.002", run_sosra(u0, 0.002f0))
row("ETD1    dt=0.002", run_etd(u0, 0.002f0; rk2=false))
row("ETDRK2  dt=0.008", run_etd(u0, 0.008f0; rk2=true))
row("ETDRK2  dt=0.002", run_etd(u0, 0.002f0; rk2=true))
