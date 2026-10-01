using CUDA, JLD2, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

# 盒 A 的 Δt=10 版：条件复用 boxa 的 t=0 构型，target 按 zlog t10 的配方演化，
# 前 1 个时间单位 dt=1e-4，之后 dt=0.002 到 t=10。
κ, λ, γ, T, dt, horizon = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
warmup, warmup_dt = 1.0f0, 0.0001f0
noise_seed = UInt64(2026091903)
dir = normpath(joinpath(@__DIR__, "..", "trainingdata"))
condition = joinpath(
    dir, "modelB_L32_Ne51200_corrgaussboxa_l1_kappa_0.340340_t0_seed2026091901.jld2",
)
target = joinpath(
    dir,
    "modelB_L32_Ne51200_corrgaussboxa_l1_kappa_0.340340_t10_dt0.002_seed2026091903.jld2",
)
@assert ispath(condition) && !ispath(target)

cfgs = load(condition, "cfgs")
L, Nensemble = size(cfgs, 1), size(cfgs, 3)
M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
u = CuArray(cfgs)

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
    temperature=T, dt, warmup, warmup_dt, noise_seed, source_path=condition,
    max_abs_conservation_drift=maximum(abs, drift),
)
println("time=$(round(seconds; digits=1)) s, max|ΔM|=$(maximum(abs, drift))")
