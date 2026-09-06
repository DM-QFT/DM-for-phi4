using CUDA, JLD2, NPZ, Printf, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

root = normpath(joinpath(@__DIR__, ".."))
run = joinpath(root, "results", "multiL")
Nensemble = 512
κ, λ, γ, T, dt = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0
times = vcat(collect(0.5:0.5:50.0), collect(60.0:10.0:1000.0))
sources = (
    (
        32,
        # 现名 hmc_L32_N2048_t0.npz；参考轨迹只用了前 512 个初态（原 N512 文件）
        joinpath(run, "hmc_L32_N2048_t0.npz"),
        UInt64(2026085101),
    ),
    (
        64,
        joinpath(
            root, "runs",
            "modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip",
            "samples", "hmc_L64_N512_seed2026080201",
            "hmc_ep10000_L64_t0_N512.npz",
        ),
        UInt64(2026085201),
    ),
    (
        128,
        joinpath(run, "hmc_L128_N512_t0.npz"),
        UInt64(2026085301),
    ),
)

function time_tag(time)
    text = @sprintf("%.1f", time)
    return endswith(text, ".0") ? text[1:end-2] : replace(text, "." => "p")
end

for (L, source, seed) in sources
    !isempty(ARGS) && L ∉ parse.(Int, ARGS) && continue
    output = joinpath(root, "results", "modelb_reference", "multiL_L$L.h5")
    if ispath(output)
        println("跳过（已存在）: $output")
        continue
    end
    data = npzread(source)
    initial = permutedims(data["phi_gen"], (3, 2, 1))   # h5py 读出即 [ensemble,x,y]
    @assert size(initial) == (L, L, Nensemble)
    M0 = vec(mean(Float64, initial; dims=(1, 2)))
    u = CuArray(initial)
    buffers = modelb_fixed_buffers(u; seed)
    all_cfgs = Array{Float32}(undef, L, L, Nensemble, length(times))
    generation_seconds = Vector{Float64}(undef, length(times))
    max_abs_conservation_drift = Vector{Float64}(undef, length(times))
    previous = 0.0
    for (i, time) in enumerate(times)
        seconds = @elapsed begin
            modelB_2d_fixed!(
                u, buffers; γ, T, κ, λ, dt,
                tspan=(0.0f0, Float32(time - previous)),
            )
            CUDA.synchronize()
        end
        cfgs = Array(u)
        drift = maximum(abs, vec(mean(Float64, cfgs; dims=(1, 2))) - M0)
        all_cfgs[:, :, :, i] = cfgs
        generation_seconds[i] = seconds
        max_abs_conservation_drift[i] = drift
        println(
            "L=$L t=$(time_tag(time)): $(round(seconds; digits=3)) s, " *
            "max|ΔM|=$drift",
        )
        flush(stdout)
        previous = time
    end
    mkpath(dirname(output))
    jldsave(
        output; cfgs=all_cfgs, times=Float64.(times), L, Nensemble,
        kappa=κ, lambda=λ, gamma=γ, temperature=T, dt,
        noise_seed=seed, source_path=source, generation_seconds,
        max_abs_conservation_drift,
        layout="cfgs[time, ensemble, x, y]",
    )
    println("saved $output")
    flush(stdout)
end
