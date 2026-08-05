# L=256 Wolff+FA-HMC ensembles (3 kappas, 5120 cfgs) for beta_conf term scaling.
# Schema matches the existing trainingdata files: cfgs (L,L,n), N, n_samples,
# acc_rate, ε_final, κ, λ.
using Printf
include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

const OUT = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata"))
const L = 256
const λ2 = 0.022
const NS = 5120

κs = [0.26, 0.2705, 0.28]
# near-critical run gets twice the warmup; metric refits sit in the middle third
warmups = Dict(0.26 => 2500, 0.2705 => 5000, 0.28 => 2500)

const THIN = 5

tasks = map(enumerate(κs)) do (i, κ)
    Threads.@spawn wolff_fahmc(
        D=2, N_lat=L, κ=κ, λ=λ2,
        n_samples=NS, n_warmup=warmups[κ],
        n_windows=8, n_steps=100, n_wolff=5, n_thin=THIN,
        ε0=0.05, m2_eff=0.01, δ_target=0.75,
        seed=20260728 + 137i, store_cfgs=true, metric=:fit)
end

for (κ, t) in zip(κs, tasks)
    cfgs, acc, ε_f, _, clf, abs_mag = fetch(t)
    h1, h2 = mean(abs_mag[1:end÷2]), mean(abs_mag[end÷2+1:end])
    se = std(abs_mag) / sqrt(length(abs_mag) / 2)
    τ_int, _ = integrated_autocorrelation_time(autocorrelation(abs_mag; max_lag=500))
    path = joinpath(OUT, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ2)_$(L)^2.jld2")
    jldsave(path; cfgs=cfgs, N=L, n_samples=NS, κ=κ, λ=λ2,
            acc_rate=acc, ε_final=ε_f)
    @printf("κ=%.4f: acc=%.2f cluster=%.1f%%V | <|m|> halves %.5f/%.5f (Δ=%.1fσ) | τ_int(|m|)=%.1f iters -> %.2f stored cfgs\n",
            κ, acc, 100clf, h1, h2, abs(h1 - h2) / se, τ_int, τ_int / THIN)
    println("saved -> $path")
end
