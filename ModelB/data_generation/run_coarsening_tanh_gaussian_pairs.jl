# 硬化关联高斯合成条件 → SOSRA 演化 Δt=10 的粗化训练对。
# 条件 = A·tanh((ℓ/w)·ĝ_ℓ) + η₀·randn（逐样本 M=0），1/8 为纯噪声条件覆盖 t=0 起步。
# 谱用高斯型 exp(-q²ℓ²/4)：2D OU 谱每倍频程功率相同，UV 斑点会淹没大 ℓ 的畴结构；
# 高斯谱 IR 主导，水平集是干净大畴。壁宽 w 直接随机化，硬化度 a=ℓ/w 保证任意 ℓ 下
# 壁宽都是 O(1) 格距。外推声明：网络从未见过任何真实 Model B 轨迹态。
#   julia data_generation/run_coarsening_tanh_gaussian_pairs.jl
using CUDA, FFTW, JLD2, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

κ, λ, γ, T, dt, horizon = 0.85f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
Nensemble = 51200
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))

# (L, initial_seed, noise_seed)
runs = (
    (8, UInt64(2026081308), UInt64(2026081309)),
    (16, UInt64(2026081316), UInt64(2026081317)),
    (32, UInt64(2026081332), UInt64(2026081333)),
    (64, UInt64(2026081364), UInt64(2026081365)),
)

for (L, initial_seed, noise_seed) in runs
    condition = joinpath(
        dir, "modelB_L$(L)_Ne$(Nensemble)_tanhgauss_l1_kappa_0.85_t0_seed$(initial_seed).jld2",
    )
    target = joinpath(
        dir, "modelB_L$(L)_Ne$(Nensemble)_tanhgauss_l1_kappa_0.85_t10_dt0.002_seed$(noise_seed).jld2",
    )
    @assert !ispath(condition) && !ispath(target)

    rng = Xoshiro(initial_seed)
    logu(lo, hi) = exp.(log(lo) .+ (log(hi) - log(lo)) .* rand(rng, Float32, Nensemble))
    ℓ = logu(2.0f0, Float32(L) / 3.0f0)
    w = logu(0.7f0, 3.0f0)
    a = ℓ ./ w
    A = 0.8f0 .+ 1.0f0 .* rand(rng, Float32, Nensemble)
    η0 = 0.05f0 .+ 0.3f0 .* rand(rng, Float32, Nensemble)
    pure_noise = rand(rng, Float32, Nensemble) .< 0.125f0
    A[pure_noise] .= 0.0f0
    η0[pure_noise] .= 0.05f0 .+ 0.15f0 .* rand(rng, Float32, sum(pure_noise))

    q = 4.0f0 .* sinpi.(Float32.(0:L-1) ./ L) .^ 2
    q2 = CuArray(reshape(q, L, 1) .+ reshape(q, 1, L))
    CUDA.seed!(initial_seed)
    F = fft(CUDA.randn(Float32, L, L, Nensemble), (1, 2))
    F .*= exp.(.-q2 .* reshape(CuArray(ℓ .^ 2), 1, 1, :) ./ 4.0f0)
    g = real.(ifft(F, (1, 2)))
    g .-= mean(g; dims=(1, 2))
    g ./= sqrt.(mean(abs2.(g); dims=(1, 2)))
    u = reshape(CuArray(A), 1, 1, :) .* tanh.(reshape(CuArray(a), 1, 1, :) .* g) .+
        reshape(CuArray(η0), 1, 1, :) .* CUDA.randn(Float32, L, L, Nensemble)
    u .-= mean(u; dims=(1, 2))

    cfgs = Array(u)
    jldsave(condition; cfgs, L, Nensemble, kappa=κ, lambda=λ,
        corr_length=ℓ, wall_width=w, hardness=a, amplitude=A, noise_amp=η0,
        pure_noise_mask=Vector{Bool}(pure_noise), initial_seed)
    M0 = vec(mean(Float64, cfgs; dims=(1, 2)))

    buffers = modelb_fixed_buffers(u; seed=noise_seed)
    seconds = @elapsed begin
        modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, horizon))
        CUDA.synchronize()
    end
    @assert all(isfinite, u)
    cfgs = Array(u)
    drift = vec(mean(Float64, cfgs; dims=(1, 2))) - M0
    jldsave(target;
        cfgs, L, Nensemble, time=horizon, kappa=κ, lambda=λ, gamma=γ,
        temperature=T, dt, noise_seed, source_path=condition,
        max_abs_conservation_drift=maximum(abs, drift))
    println("L=$L: $(round(seconds; digits=1)) s, " *
            "noise-only $(sum(pure_noise)), max|ΔM|=$(maximum(abs, drift))")
end
