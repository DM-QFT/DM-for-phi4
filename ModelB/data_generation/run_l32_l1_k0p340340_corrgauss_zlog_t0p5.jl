using CUDA, JLD2, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

# Δt=0.5 版：条件复用 t10 数据集的 t=0 zlog 高斯构型，只演化 target。
# t10 版前 1 个时间单位用 dt=1e-4 防小 Z 大振幅爆掉，这里整个 horizon 都在
# 该区间内，所以全程 dt=1e-4。
κ, λ, γ, T, dt, horizon = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.0001f0, 0.5f0
noise_seed = UInt64(2026081002)
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))
condition = joinpath(
    dir, "modelB_L32_Ne51200_corrgausszlog_l1_kappa_0.340340_t0_seed2026081001.jld2",
)
target = joinpath(
    dir,
    "modelB_L32_Ne51200_corrgausszlog_l1_kappa_0.340340_t0p5_dt0.0001_seed2026081002.jld2",
)
@assert ispath(condition) && !ispath(target)

cfgs = load(condition, "cfgs")
L, Nensemble = size(cfgs, 1), size(cfgs, 3)
M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
u = CuArray(cfgs)

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
