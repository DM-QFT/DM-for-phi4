# Extend the L=256 ensembles 5120 -> 10240 by continuing each chain from its
# last stored configuration. Short warmup only re-tunes the FA metric and ε.
using Printf
include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

const OUT = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata"))
const L = 256
const λ2 = 0.022
const NS = 5120
const THIN = 5

κs = [0.26, 0.2705, 0.28]

tasks = map(enumerate(κs)) do (i, κ)
    Threads.@spawn begin
        path = joinpath(OUT, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ2)_$(L)^2.jld2")
        old = jldopen(path, "r") do f
            @assert f["n_samples"] == NS
            read(f, "cfgs")
        end
        cfgs, acc, ε_f, _, clf, abs_mag = wolff_fahmc(
            D=2, N_lat=L, κ=κ, λ=λ2,
            n_samples=NS, n_warmup=1500, n_windows=5,
            n_steps=100, n_wolff=5, n_thin=THIN,
            ε0=0.05, m2_eff=0.01, δ_target=0.75,
            seed=20260803 + 137i, store_cfgs=true, metric=:fit,
            phi0=old[:, :, end])
        (old, cfgs, acc, ε_f, clf, abs_mag)
    end
end

for (κ, t) in zip(κs, tasks)
    old, new, acc, ε_f, clf, abs_mag = fetch(t)
    phi2_old = sum(abs2, old) / length(old)
    phi2_new = sum(abs2, new) / length(new)
    τ_int, _ = integrated_autocorrelation_time(autocorrelation(abs_mag; max_lag=500))
    full = cat(old, new; dims=3)
    path = joinpath(OUT, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ2)_$(L)^2.jld2")
    tmp = path * ".tmp"
    jldsave(tmp; cfgs=full, N=L, n_samples=2NS, κ=κ, λ=λ2,
            acc_rate=acc, ε_final=ε_f)
    mv(tmp, path; force=true)
    @printf("κ=%.4f: acc=%.2f cluster=%.1f%%V | phi2 old/new %.5f/%.5f | τ_int(|m|)=%.2f stored cfgs\n",
            κ, acc, 100clf, phi2_old, phi2_new, τ_int / THIN)
    println("extended -> $path (n=10240)")
end
