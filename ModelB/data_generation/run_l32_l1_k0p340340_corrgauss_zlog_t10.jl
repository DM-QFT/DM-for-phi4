using CUDA, FFTW, JLD2, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L, Nensemble = 32, 51200
κ, λ, γ, T, dt, horizon = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
# Z 对数均匀采样：均匀测度会把质量堆在大 Z 弱涨落端（zwide 族实测 G1
# 包络反而缩水、HMC 覆盖 97%→77%），log 测度每个数量级等权，
# P(Z<0.2)≈26%，G1 包络到 ~40+，盖住 HMC 红外尾（~35）。
# 小 Z 大振幅初态在 dt=0.002 下爆（作业 2511138 实测 NaN），target 演化的
# 前 1 个时间单位用 dt=1e-4（Z 到 0.05 时 |φ| 尾部 ~9，裕量 20 倍），
# 之后回到 dt=0.002；条件仍是 t=0 的原始高斯构型。
warmup, warmup_dt = 1.0f0, 0.0001f0
initial_seed, noise_seed = UInt64(2026081001), UInt64(2026081002)
Z_range, m2_range, M_range = (0.05f0, 10.0f0), (0.0f0, 0.8f0), (-0.9f0, 0.9f0)
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))
condition = joinpath(
    dir, "modelB_L32_Ne51200_corrgausszlog_l1_kappa_0.340340_t0_seed2026081001.jld2",
)
target = joinpath(
    dir,
    "modelB_L32_Ne51200_corrgausszlog_l1_kappa_0.340340_t10_dt0.002_seed2026081002.jld2",
)
@assert !ispath(condition) && !ispath(target)
mkpath(dir)

rng = Xoshiro(initial_seed)
Z = Z_range[1] .* (Z_range[2] / Z_range[1]) .^ rand(rng, Float32, Nensemble)
m2 = m2_range[1] .+ (m2_range[2] - m2_range[1]) .* rand(rng, Float32, Nensemble)
M = M_range[1] .+ (M_range[2] - M_range[1]) .* rand(rng, Float32, Nensemble)
q = 4.0f0 .* sinpi.(Float32.(0:L-1) ./ L) .^ 2
q2 = reshape(q, L, 1) .+ reshape(q, 1, L)
# q²(0)=∞ removes the zero mode before adding the independently sampled M.
q2[1, 1] = Inf32
CUDA.seed!(initial_seed)
F = fft(CUDA.randn(Float32, L, L, Nensemble), (1, 2))
F ./= sqrt.(
    reshape(CuArray(q2), L, L, 1) .* reshape(CuArray(Z), 1, 1, :) .+
    reshape(CuArray(m2), 1, 1, :)
)
u = real.(ifft(F, (1, 2)))
u .+= reshape(CuArray(M), 1, 1, :)
cfgs = Array(u)
M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
@assert maximum(abs, M0 - M) < 1.0e-5
jldsave(
    condition;
    cfgs, L, Nensemble, kappa=κ, lambda=λ,
    propagator_Z=Z, propagator_m2=m2, constant_mean=M,
    Z_range, m2_range, M_range, initial_seed, warmup, warmup_dt,
)

buffers = modelb_fixed_buffers(u; seed=noise_seed)
seconds = @elapsed begin
    modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt=warmup_dt, tspan=(0.0f0, warmup))
    modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(warmup, horizon))
    CUDA.synchronize()
end
@assert all(isfinite, u)
cfgs = Array(u)
drift = vec(mean(Float64, cfgs; dims=(1, 2))) - M0
jldsave(
    target;
    cfgs, L, Nensemble, time=horizon, kappa=κ, lambda=λ, gamma=γ,
    temperature=T, dt, noise_seed, source_path=condition,
    max_abs_conservation_drift=maximum(abs, drift),
)
println("time=$(round(seconds; digits=1)) s, max|ΔM|=$(maximum(abs, drift))")
