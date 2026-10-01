using CUDA, FFTW, JLD2, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L, Nensemble = 32, 51200
κ, λ, γ, T, dt, horizon = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.0001f0, 0.5f0
# 盒 A（analysis/param_clouds.py 的测量）：HMC 有效 Z 0.875±11%（估计噪声），非平衡 t≥0.5 后 Z 0.42–0.95，
# 取 Z 均匀 [0.45,1.15]；HMC 的 m² 随 |M| 从 0（|M|<0.3）升到 0.9（|M|≈0.7），zhmc 的 m²≤0.3 只盖住
# 67% 的 HMC 构型，取 m² 均匀 [0,0.8] 盖住 95%；M 均匀 ±0.8 盖住 99.6%。
initial_seed, noise_seed = UInt64(2026091901), UInt64(2026091902)
Z_range, m2_range, M_range = (0.45f0, 1.15f0), (0.0f0, 0.8f0), (-0.8f0, 0.8f0)
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))
condition = joinpath(
    dir, "modelB_L32_Ne51200_corrgaussboxa_l1_kappa_0.340340_t0_seed2026091901.jld2",
)
target = joinpath(
    dir,
    "modelB_L32_Ne51200_corrgaussboxa_l1_kappa_0.340340_t0p5_dt0.0001_seed2026091902.jld2",
)
@assert !ispath(condition) && !ispath(target)
mkpath(dir)

q = 4.0f0 .* sinpi.(Float32.(0:L-1) ./ L) .^ 2
q2 = reshape(q, L, 1) .+ reshape(q, 1, L)
rng = Xoshiro(initial_seed)
Z = Z_range[1] .+ (Z_range[2] - Z_range[1]) .* rand(rng, Float32, Nensemble)
m2 = m2_range[1] .+ (m2_range[2] - m2_range[1]) .* rand(rng, Float32, Nensemble)
M = M_range[1] .+ (M_range[2] - M_range[1]) .* rand(rng, Float32, Nensemble)
CUDA.seed!(initial_seed)
F = fft(CUDA.randn(Float32, L, L, Nensemble), (1, 2))
F ./= sqrt.(
    reshape(CuArray(q2), L, L, 1) .* reshape(CuArray(Z), 1, 1, :) .+
    reshape(CuArray(m2), 1, 1, :)
)
# 零模置零再加独立采样的 M
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
