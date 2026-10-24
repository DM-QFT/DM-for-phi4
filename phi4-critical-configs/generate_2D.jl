# 用 WolffFAHMC_ND.jl 重新生成 HF 数据集的 2D 组态，κ=0.2705、λ=0.022、L=16..128。
# 不抽稀时 |M| 和 S 的 τ_int 约 2 次迭代，每 5 次迭代存一个组态使相邻组态独立。
# 存成与 2Dphi4/trainingdata 相同 schema 的 jld2，再由 make_npy.py 转 npy。四个 L 并行，julia -t 4。
using Statistics, JLD2
include(joinpath(@__DIR__, "..", "tools", "julia", "WolffFAHMC_ND.jl"))

κ, λ = 0.2705, 0.022
n_samples = 10240
n_warmup = 2000
n_thin = 5
Ls = [16, 32, 64, 128]
dir = joinpath(@__DIR__, "jld2")
mkpath(dir)

tasks = map(Ls) do L
    Threads.@spawn begin
        seed = 20261017 + L
        seconds = @elapsed cfgs, acc_rate, ε_final, _, cluster_frac, abs_mag = wolff_fahmc(;
            D=2, N_lat=L, κ, λ, n_samples, n_warmup, n_thin, store_cfgs=true, seed,
        )
        τ_int, _ = integrated_autocorrelation_time(autocorrelation(abs_mag; max_lag=200))
        jldsave(joinpath(dir, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^2.jld2");
                cfgs, N=L, κ, λ, n_samples, acc_rate, ε_final, n_warmup, n_thin, seed, cluster_frac,
                generation_seconds=seconds)
        (acc_rate, ε_final, cluster_frac, τ_int, seconds)
    end
end

for (L, t) in zip(Ls, tasks)
    acc_rate, ε_final, cluster_frac, τ_int, seconds = fetch(t)
    println("L=$L  acc=$(round(acc_rate; digits=3))  ε=$(round(ε_final; digits=4))  " *
            "cluster=$(round(100cluster_frac; digits=1))%V  " *
            "τ_int(|M|)=$(round(τ_int; digits=2)) iters  " *
            "time=$(round(seconds; digits=0)) s")
    flush(stdout)
end
