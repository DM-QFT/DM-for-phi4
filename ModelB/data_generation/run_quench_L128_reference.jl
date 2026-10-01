# 用 DM 淬火链的 L=128 初态跑 SOSRA 参照到 t=1000, 供 ML4PS Fig.3(a) 同初态对比。
# 参数与 run_coarsening_tanh_gaussian_pairs.jl 的训练数据一致, 输出布局同 run_modelb_references.jl。
#   julia data_generation/run_quench_L128_reference.jl

using CUDA, JLD2, NPZ, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const κ, λ, γ, T, dt, Δt = 0.85f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
const TARGET = 1000.0f0

source = joinpath(ROOT, "runs", "modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss",
                  "samples", "quench_t0_L128_N32.npz")
output = joinpath(ROOT, "results", "modelb_reference", "quench_tanhgauss_quench_L128_N32.h5")
seed = UInt64(20260913)

magnetization(c) = vec(mean(Float64, c; dims=(1, 2)))

# npz 是 [ensemble, x, y]；(3,2,1) 让写盘后 h5py 读出仍是 [ensemble, x, y]
cfgs = permutedims(npzread(source)["phi_gen"], (3, 2, 1))
L, Nensemble = size(cfgs, 1), size(cfgs, 3)
M0 = magnetization(cfgs)

nsteps = round(Int, TARGET / Δt)
all_cfgs = Array{Float32}(undef, L, L, Nensemble, nsteps)
times = Vector{Float32}(undef, nsteps)
drifts = Vector{Float64}(undef, nsteps)

u = CuArray(cfgs)
buffers = modelb_fixed_buffers(u; seed)
println("=== quench L=$L N=$Nensemble κ=$κ  t=0→$(TARGET) ===")
flush(stdout)
for i in 1:nsteps
    seconds = @elapsed begin
        modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, Δt))
        CUDA.synchronize()
    end
    @assert all(isfinite, u)
    snapshot = Array(u)
    all_cfgs[:, :, :, i] = snapshot
    times[i] = Float32(i * Δt)
    drifts[i] = maximum(abs, magnetization(snapshot) - M0)
    if i % 10 == 0
        println("  t=$(Int(times[i])): $(round(seconds; digits=3)) s, max|ΔM|=$(drifts[i])")
        flush(stdout)
    end
end

jldsave(output;
    cfgs=all_cfgs, times, max_abs_conservation_drift=drifts,
    L, Nensemble, kappa=κ, lambda=λ, dt, gamma=γ, temperature=T,
    noise_seed=seed, source_path=source, layout="cfgs[time, ensemble, x, y]")
println("saved $output")
