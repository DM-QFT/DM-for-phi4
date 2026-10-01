using CUDA, FFTW, JLD2, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L, Nensemble = 32, 51200
κ, λ, γ, T, dt, horizon = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.0001f0, 0.5f0
# zlog 的 Δt=0.5 核迭代时最软模功率流失（t=20 时 G_1 只剩 SOSRA 的 0.855）。
# 检验是否训练集覆盖问题：对 51200 个 HMC 平衡构型逐个做 |φ_q|²/V ~ Exp(1/(Z q̂²+m²))
# 的 MLE，有效 Z 落在 0.68–1.10（0.5–99.5%，极值 0.55–1.26，中位数 0.88），|M| 最大 0.85，
# 系综平均谱 Z̄=0.91、m̄²=0.11；zlog 集 Z∈[0.05,10] 只有 8.9% 落在 HMC 的 Z 范围。
# 这里把区间收到 HMC 周围：Z 均匀 [0.45,1.3]（中位数 0.875，49% 落在 HMC 带内，3 倍跨度
# 内对数测度没有意义），m² 均匀 [0,0.3]，M 均匀 ±0.85，对应格点 std ≈0.5–1.15、
# ⟨G_1⟩ 2.5–29，盖住 HMC 的 std 0.45–0.94、G_1 0.15–38（指数尾）。
# Z≥0.45 振幅温和，dt=1e-4 全程只是为了与 zlog 的 t0.5 数据集数值一致。
initial_seed, noise_seed = UInt64(2026091701), UInt64(2026091702)
Z_range, m2_range, M_range = (0.45f0, 1.3f0), (0.0f0, 0.3f0), (-0.85f0, 0.85f0)
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))
condition = joinpath(
    dir, "modelB_L32_Ne51200_corrgausszhmc_l1_kappa_0.340340_t0_seed2026091701.jld2",
)
target = joinpath(
    dir,
    "modelB_L32_Ne51200_corrgausszhmc_l1_kappa_0.340340_t0p5_dt0.0001_seed2026091702.jld2",
)
@assert !ispath(condition) && !ispath(target)
mkpath(dir)

rng = Xoshiro(initial_seed)
Z = Z_range[1] .+ (Z_range[2] - Z_range[1]) .* rand(rng, Float32, Nensemble)
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
    Z_range, m2_range, M_range, initial_seed,
)

buffers = modelb_fixed_buffers(u; seed=noise_seed)
seconds = @elapsed begin
    modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, horizon))
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
