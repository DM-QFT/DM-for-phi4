using JLD2, Statistics, Printf

include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

# Model B 逐轨迹守恒 M，canonical 协议必须在初态采样其分布。
@assert length(ARGS) in (6, 7)
L = parse(Int, ARGS[1])
n = parse(Int, ARGS[2])
kappa = parse(Float64, ARGS[3])
lambda = parse(Float64, ARGS[4])
seed = parse(Int, ARGS[5])
output = abspath(ARGS[6])
n_thin = length(ARGS) == 7 ? parse(Int, ARGS[7]) : 1
@assert !ispath(output)

seconds = @elapsed cfgs64, acceptance, epsilon, _, _, _ = wolff_fahmc(;
    D=2, N_lat=L, κ=kappa, λ=lambda, n_samples=n, n_warmup=2000,
    n_thin, store_cfgs=true, seed,
)
cfgs = Float32.(cfgs64)
@assert size(cfgs) == (L, L, n)
magnetization = vec(mean(cfgs; dims=(1, 2)))
U4 = 1 - mean(magnetization .^ 4) / (3 * mean(magnetization .^ 2)^2)
mkpath(dirname(output))
jldsave(
    output;
    cfgs, L, kappa, lambda, acceptance, epsilon, seed, n_thin, seconds,
    magnetization, binder_u4=U4,
)
@printf(
    "saved %s  acceptance=%.3f  U4=%.4f  std(M)=%.4f  %.1fs\n",
    output, acceptance, U4, std(magnetization), seconds,
)
