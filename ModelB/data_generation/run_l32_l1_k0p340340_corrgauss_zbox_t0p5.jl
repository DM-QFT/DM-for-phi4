using CUDA, FFTW, JLD2, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L, Nensemble = 32, 102400
κ, λ, γ, T, dt, horizon = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.0001f0, 0.5f0
# 同时盖住 HMC 平衡态和 Fig.2 非平衡链（iid 高斯 + HMC 的 M）整条 SOSRA 轨迹。逐构型 MLE：
# HMC 有效 Z 0.73–1.04（极值到 1.26）、系综 m²≈0.1–0.2；非平衡态 t=0 是平谱 (Z=0, m²≈1)，
# t=0.5 时紫外已弛豫到 Z≈0.65、m²≈0.66，再沿 (0.75,0.39)@t=5、(0.8,0.27)@t=20 走到 HMC。
# 并集取矩形 Z∈[0,1.3]、m²∈[0,1.2]，只切掉 Z、m² 同时趋零的发散角：要求 ⟨G_1⟩=1/(Z q̂₁²+m²)≤30
# （zhmc 盒的角），即 m² ≥ 1/30 − Z q̂₁²，只占矩形面积 0.5%。数据量加倍到 102400，
# 训练用 5000 epoch 保持与 51200×10000 相同的 8M 步。M 两个系综同分布，沿用 ±0.85。
initial_seed, noise_seed = UInt64(2026091801), UInt64(2026091802)
Z_range, m2_max, M_range, G1_cap = (0.0f0, 1.3f0), 1.2f0, (-0.85f0, 0.85f0), 30.0f0
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))
condition = joinpath(
    dir, "modelB_L32_Ne102400_corrgausszbox_l1_kappa_0.340340_t0_seed2026091801.jld2",
)
target = joinpath(
    dir,
    "modelB_L32_Ne102400_corrgausszbox_l1_kappa_0.340340_t0p5_dt0.0001_seed2026091802.jld2",
)
@assert !ispath(condition) && !ispath(target)
mkpath(dir)

q = 4.0f0 .* sinpi.(Float32.(0:L-1) ./ L) .^ 2
q2 = reshape(q, L, 1) .+ reshape(q, 1, L)
q2_1 = q2[2, 2]
rng = Xoshiro(initial_seed)
Z = Z_range[1] .+ (Z_range[2] - Z_range[1]) .* rand(rng, Float32, Nensemble)
m2_lo = max.(0.0f0, 1.0f0 / G1_cap .- Z .* q2_1)
m2 = m2_lo .+ (m2_max .- m2_lo) .* rand(rng, Float32, Nensemble)
M = M_range[1] .+ (M_range[2] - M_range[1]) .* rand(rng, Float32, Nensemble)
CUDA.seed!(initial_seed)
F = fft(CUDA.randn(Float32, L, L, Nensemble), (1, 2))
F ./= sqrt.(
    reshape(CuArray(q2), L, L, 1) .* reshape(CuArray(Z), 1, 1, :) .+
    reshape(CuArray(m2), 1, 1, :)
)
# 零模单独置零再加独立采样的 M；不能像 zlog 脚本那样用 q²(0)=∞，Z=0 时 ∞·0=NaN
F[1, 1, :] .= 0
u = real.(ifft(F, (1, 2)))
u .+= reshape(CuArray(M), 1, 1, :)
cfgs = Array(u)
M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
@assert maximum(abs, M0 - M) < 1.0e-5
jldsave(
    condition;
    cfgs, L, Nensemble, kappa=κ, lambda=λ,
    propagator_Z=Z, propagator_m2=m2, constant_mean=M,
    Z_range, m2_max, G1_cap, M_range, initial_seed,
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
