# L=128 的 3D 临界点组态，κ=0.1923、λ=0.9，共 2560 个。单链每次迭代约 3 s，抽稀 10 出 2560 个要一天，
# 改为 n_chains 条独立链并行各出 n_per 个，seed 不同，最后沿第 4 维拼成一个文件。julia -t 10。
using Statistics, JLD2
include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

κ, λ = 0.1923, 0.9
L = 128
n_chains = 10
n_per = 256
n_warmup = 2000
n_thin = 10
n_wolff = 200
path = normpath(joinpath(@__DIR__, "..", "trainingdata", "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^3.jld2"))

tasks = map(1:n_chains) do c
    Threads.@spawn begin
        seed = 20261245 + c
        seconds = @elapsed cfgs, acc_rate, ε_final, _, cluster_frac, abs_mag = wolff_fahmc(;
            D=3, N_lat=L, κ, λ, n_samples=n_per, n_warmup, n_thin, n_wolff, store_cfgs=true, seed,
        )
        τ_int, _ = integrated_autocorrelation_time(autocorrelation(abs_mag; max_lag=200))
        println("chain $c  seed=$seed  acc=$(round(acc_rate; digits=3))  ε=$(round(ε_final; digits=4))  " *
                "cluster=$(round(100cluster_frac; digits=2))%V  τ_int(|M|)=$(round(τ_int; digits=2)) iters  " *
                "time=$(round(seconds; digits=0)) s")
        flush(stdout)
        (cfgs, acc_rate, ε_final, cluster_frac, τ_int, seed, seconds)
    end
end

res = fetch.(tasks)
cfgs = cat((r[1] for r in res)...; dims=4)
jldsave(path;
        cfgs, N=L, κ, λ, n_samples=n_chains * n_per, n_chains, n_per,
        acc_rate=mean(r[2] for r in res), ε_final=mean(r[3] for r in res), n_warmup, n_thin, n_wolff,
        seeds=[r[6] for r in res], cluster_frac=mean(r[4] for r in res),
        tau_int_chains=[r[5] for r in res], generation_seconds=maximum(r[7] for r in res))
println("saved $(size(cfgs)) -> $path")
