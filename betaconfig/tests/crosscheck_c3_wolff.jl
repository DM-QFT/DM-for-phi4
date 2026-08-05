# Independent Wolff+FAHMC cross-check of the C3 MALA ensembles.
# exp(-beta S_{kappa,lambda}(x)) with x = c*phi maps to the standard action with
# kappa' = beta kappa c^2, lambda' = beta lambda c^4, c^2 root of
# 2 beta lambda u^2 + beta(1-2lambda) u - 1 = 0.
using Printf
include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

const L = 64
const κ0, λ0 = 0.28, 0.022
βs = parse.(Float64, readlines(joinpath(@__DIR__, "..", "results", "verification", "c3_betas.txt")))

outdir = joinpath(@__DIR__, "..", "results", "verification", "crosscheck_wolff")
mkpath(outdir)

tasks = map(enumerate(βs)) do (i, β)
    Threads.@spawn begin
        a = β * (1 - 2λ0)
        u = (-a + sqrt(a^2 + 8β * λ0)) / (4β * λ0)
        c = sqrt(u)
        κp, λp = β * κ0 * u, β * λ0 * u^2
        cfgs, acc, _, _, _, _ = wolff_fahmc(
            D=2, N_lat=L, κ=κp, λ=λp,
            n_samples=4096, n_warmup=2000, n_windows=5,
            n_steps=50, n_wolff=5, n_thin=2,
            ε0=0.05, m2_eff=0.01, seed=90000 + i, store_cfgs=true, metric=:fit)
        (cfgs, c, κp, λp, acc)
    end
end

for (β, t) in zip(βs, tasks)
    cfgs, c, κp, λp, acc = fetch(t)
    jldsave(joinpath(outdir, @sprintf("wolff_beta%.4f.jld2", β));
            cfgs=cfgs, c=c, beta=β, kappa_p=κp, lambda_p=λp)
    @printf("beta=%.4f: kappa'=%.5f lambda'=%.6f c=%.4f acc=%.2f\n",
            β, κp, λp, c, acc)
end
