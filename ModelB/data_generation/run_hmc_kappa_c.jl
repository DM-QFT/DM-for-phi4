using JLD2, Statistics

include(joinpath(@__DIR__, "..", "..", "julia", "WolffFAHMC_ND.jl"))

# κc gives U4=0.6103 at L=128, near the 2D Ising value 0.6106.
κ, λ = 0.27088, 0.022
n_samples = isempty(ARGS) ? 10240 : parse(Int, ARGS[1])
Ls = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [8, 16, 32, 64]
n_warmup = 2000
dir = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata"))
mkpath(dir)

for L in Ls
    seed = 20260901 + L
    seconds = @elapsed cfgs, accept_rate, ε_final, _, _, _ = wolff_fahmc(;
        D=2, N_lat=L, κ, λ, n_samples, n_warmup, store_cfgs=true, seed,
    )
    M = vec(mean(cfgs; dims=(1, 2)))
    U4 = 1 - mean(M .^ 4) / (3 * mean(M .^ 2)^2)
    jldsave(joinpath(dir, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^2.jld2");
            cfgs, N=L, κ, λ, n_samples, acc_rate=accept_rate, ε_final, n_warmup,
            seed, binder_u4=U4, generation_seconds=seconds)
    println("L=$L U4=$(round(U4; digits=4)) accept=$(round(accept_rate; digits=3)) " *
            "time=$(round(seconds; digits=1)) s"); flush(stdout)
end
