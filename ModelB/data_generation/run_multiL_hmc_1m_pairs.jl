using CUDA, JLD2, Statistics

include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))
include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

# Fig. 3 每个时间间隔的三个尺寸共 1024000 对。两个作业使用独立种子和文件。
# 各尺寸的样本数是 batch 256 的整数倍，L=32 多一个 batch。
const HMC_SIZES = ((8, 341248), (16, 341248), (32, 341504))
const HMC_DIR = normpath(joinpath(@__DIR__, "..", "trainingdata"))

function generate_hmc_pairs(L, Nensemble, initial_seed, noise_seed, horizon, dir)
    κ, λ = 0.34034, 1.0
    γ, T, dt = 1.0f0, 1.0f0, 0.002f0
    n_warmup, n_thin, n_steps, n_wolff = 5000, 10, 50, 5
    discarded_post_warmup = 5000
    generated_samples = Nensemble + discarded_post_warmup
    stem = "modelB_L$(L)_Ne$(Nensemble)_hmc_l1_kappa_0.340340"
    condition = joinpath(dir, "$(stem)_t0_seed$(initial_seed).jld2")
    tag = horizon == 0.5f0 ? "0p5" : "10"
    target = joinpath(dir, "$(stem)_t$(tag)_dt0.002_seed$(noise_seed).jld2")
    if isfile(condition) && isfile(target)
        println("L=$L 的训练对已存在，直接用于续训")
        return
    end
    @assert !ispath(condition) && !ispath(target)
    mkpath(dir)

    seconds = @elapsed cfgs64, acceptance, epsilon, _, cluster_fraction, _ = wolff_fahmc(;
        D=2, N_lat=L, κ, λ, n_samples=generated_samples, n_warmup, n_thin,
        n_steps, n_wolff, store_cfgs=true, seed=initial_seed,
    )
    cfgs = Float32.(@view cfgs64[:, :, discarded_post_warmup+1:end])
    cfgs64 = nothing
    @assert all(isfinite, cfgs)
    M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
    binder_u4 = 1 - mean(M0 .^ 4) / (3 * mean(M0 .^ 2)^2)
    jldsave(condition;
        cfgs, L, Nensemble, kappa=κ, lambda=λ, initial_seed,
        n_warmup, n_thin, n_steps, n_wolff, acceptance, epsilon,
        discarded_post_warmup, generated_samples,
        cluster_fraction, magnetization=M0, binder_u4, hmc_generation_seconds=seconds,
    )
    println("saved $condition  acceptance=$acceptance  U4=$binder_u4")
    flush(stdout)

    u = CuArray(cfgs)
    buffers = modelb_fixed_buffers(u; seed=UInt64(noise_seed))
    seconds = @elapsed begin
        modelB_2d_fixed!(u, buffers; κ, λ, γ, T, dt, tspan=(0.0f0, horizon))
        CUDA.synchronize()
    end
    @assert all(isfinite, u)
    copyto!(cfgs, u)
    drift = maximum(abs, vec(mean(Float64, cfgs; dims=(1, 2))) - M0)
    @assert drift < 1.0e-5
    jldsave(target;
        cfgs, L, Nensemble, time=horizon, kappa=κ, lambda=λ,
        gamma=γ, temperature=T, dt, noise_seed, source_path=condition,
        max_abs_conservation_drift=drift,
    )
    println("saved $target  integration=$(round(seconds; digits=1)) s  max|ΔM|=$drift")
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    horizon = parse(Float32, only(ARGS))
    @assert horizon in (0.5f0, 10.0f0)
    seed_offset = horizon == 0.5f0 ? 0 : 200
    for (L, Nensemble) in HMC_SIZES
        generate_hmc_pairs(L, Nensemble, 2026092200 + seed_offset + L,
                           2026092300 + seed_offset + L, horizon, HMC_DIR)
        GC.gc()
        CUDA.reclaim()
    end
end
