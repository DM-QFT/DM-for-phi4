# 为每条 DM 链生成 Model B 物理参照，从各链自己的 t=0 初态出发演化到 t=1000。
#
# 每条参照写成 results/modelb_reference/ 下的一个合并文件（时间是最外维），布局遵循
# docs/REPO_LAYOUT.md：h5py 读出即 cfgs[time, ensemble, x, y]。Julia 是列
# 主序，所以内存里数组按 [y, x, ensemble, time] 组织。
#
#   julia data_generation/run_modelb_references.jl

using CUDA, JLD2, NPZ, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const κ, λ, γ, T, dt, Δt = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
const TARGET = 2000.0f0

hmc_noskip = joinpath(ROOT, "runs",
    "modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip", "samples")
corrgauss = joinpath(ROOT, "runs",
    "modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip", "samples")

# (初态 npz, 输出文件名, 噪声 seed)；全部写进 results/modelb_reference/
references = (
    (joinpath(hmc_noskip, "hmc_L64_N512_seed2026080201",
              "hmc_ep10000_L64_t0_N512.npz"),
     "hmc_noskip_L64_N512.h5", UInt64(2026080401)),
    (joinpath(hmc_noskip, "hmc_chain_ep10000_N1024_seed20260901",
              "hmc_ep10000_t0_N1024.npz"),
     "hmc_noskip_hmc_L32_N1024.h5", UInt64(20261101)),
    (joinpath(corrgauss, "hmc_chain_ep10000_N1024_seed20260901",
              "hmc_ep10000_t0_N1024.npz"),
     "corrgauss_hmc_L32_N1024.h5", UInt64(20261101)),
    (joinpath(corrgauss, "gaussian_chain_ep10000_N1024_seed20261001",
              "gaussian_ep10000_t0_N1024_seed20260721.npz"),
     "corrgauss_gaussian_L32_N1024.h5", UInt64(20261201)),
    # 同一组 Gaussian 场，逐轨道把 M 换成配对 HMC 构型的 M（守恒荷分布与 Eq 一致）
    (joinpath(corrgauss, "gaussianM_chain_ep10000_N1024",
              "gaussianM_ep10000_t0_N1024.npz"),
     "corrgauss_gaussianM_L32_N1024.h5", UInt64(20261301)),
)

magnetization(c) = vec(mean(Float64, c; dims=(1, 2)))

const REFDIR = joinpath(ROOT, "results", "modelb_reference")
mkpath(REFDIR)
for (source, name, seed) in references
    output = joinpath(REFDIR, name)
    if ispath(output)
        println("跳过（已存在）: $output"); flush(stdout); continue
    end
    data = npzread(source)
    # npz 是 [ensemble, x, y]；(3,2,1) 让写盘后 h5py 读出仍是 [ensemble, x, y]
    cfgs = permutedims(data["phi_gen"], (3, 2, 1))
    L, Nensemble = size(cfgs, 1), size(cfgs, 3)
    indices = Vector{Int64}(data["cond_indices"])
    M0 = magnetization(cfgs)

    nsteps = round(Int, TARGET / Δt)
    all_cfgs = Array{Float32}(undef, L, L, Nensemble, nsteps)
    times = Vector{Float32}(undef, nsteps)
    drifts = Vector{Float64}(undef, nsteps)

    u = CuArray(cfgs)
    buffers = modelb_fixed_buffers(u; seed)
    println("=== $name  L=$L N=$Nensemble  t=0→$(TARGET) ===")
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
            println("  t=$(Int(times[i])): $(round(seconds; digits=3)) s, " *
                    "max|ΔM|=$(drifts[i])")
            flush(stdout)
        end
    end

    jldsave(output;
        cfgs=all_cfgs, times, cond_indices_0based=indices,
        max_abs_conservation_drift=drifts,
        L, Nensemble, kappa=κ, lambda=λ, dt, gamma=γ, temperature=T,
        noise_seed=seed, source_path=source,
        layout="cfgs[time, ensemble, x, y]")
    println("saved $output"); flush(stdout)
end
