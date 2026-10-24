# 用 tools/julia/WolffFAHMC_ND.jl 生成 3D φ⁴ 临界点组态，κ=0.1923、λ=0.9、L=16,32,64，各 2560 个。
# 3D 临界点的 Wolff 团簇只占体积的百分之几，默认 n_wolff=5 时 |M| 的 τ_int 在 L=32 约 7 次迭代、
# L=64 超过 12 次。实测 L=32 上 n_wolff=50 把 τ_int 压到 3、n_wolff=200 压到 2.4，而每次团簇翻转只花
# O(V) 的几纳秒每格点，远比一条 HMC 轨迹便宜，所以取 n_wolff=100 并每 10 次迭代存一个组态。
# 三个 L 并行，julia -t 3。目标文件已存在时加 _thin 后缀，不覆盖已有训练数据。
using Statistics, JLD2
include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

κ, λ = 0.1923, 0.9
n_samples = 2560
n_warmup = 2000
n_thin = 10
n_wolff = 100
Ls = [16, 32, 64]
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))

tasks = map(Ls) do L
    Threads.@spawn begin
        seed = 20261117 + L
        seconds = @elapsed cfgs, acc_rate, ε_final, _, cluster_frac, abs_mag = wolff_fahmc(;
            D=3, N_lat=L, κ, λ, n_samples, n_warmup, n_thin, n_wolff, store_cfgs=true, seed,
        )
        τ_int, _ = integrated_autocorrelation_time(autocorrelation(abs_mag; max_lag=200))
        path = joinpath(dir, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^3.jld2")
        if isfile(path)
            path = replace(path, ".jld2" => "_thin$(n_thin).jld2")
        end
        jldsave(path; cfgs, N=L, κ, λ, n_samples, acc_rate, ε_final, n_warmup, n_thin, n_wolff, seed,
                cluster_frac, generation_seconds=seconds)
        (path, acc_rate, ε_final, cluster_frac, τ_int, seconds)
    end
end

for (L, t) in zip(Ls, tasks)
    path, acc_rate, ε_final, cluster_frac, τ_int, seconds = fetch(t)
    println("L=$L  acc=$(round(acc_rate; digits=3))  ε=$(round(ε_final; digits=4))  " *
            "cluster=$(round(100cluster_frac; digits=1))%V  τ_int(|M|)=$(round(τ_int; digits=2)) iters  " *
            "time=$(round(seconds; digits=0)) s  -> $path")
    flush(stdout)
end
