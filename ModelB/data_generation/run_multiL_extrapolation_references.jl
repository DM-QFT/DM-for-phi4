using CUDA, JLD2, NPZ, Printf, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

root = normpath(joinpath(@__DIR__, ".."))
run = joinpath(root, "runs", "multiL_ep10000_extrapolation")
Nensemble = 512
κ, λ, γ, T, dt = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0
times = vcat(collect(0.5:0.5:50.0), collect(60.0:10.0:1000.0))
sources = (
    (
        32,
        joinpath(run, "hmc_L32_N512_t0.npz"),
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
)

function time_tag(time)
    text = @sprintf("%.1f", time)
    return endswith(text, ".0") ? text[1:end-2] : replace(text, "." => "p")
end

for (L, source, seed) in sources
    data = npzread(source)
    initial = permutedims(data["phi_gen"], (2, 3, 1))
    @assert size(initial) == (L, L, Nensemble)
    M0 = vec(mean(Float64, initial; dims=(1, 2)))
    u = CuArray(initial)
    buffers = modelb_fixed_buffers(u; seed)
    directory = joinpath(run, "modelb_reference", "L$L")
    mkpath(directory)
    previous = 0.0
    for time in times
        seconds = @elapsed begin
            modelB_2d_fixed!(
                u, buffers; γ, T, κ, λ, dt,
                tspan=(0.0f0, Float32(time - previous)),
            )
            CUDA.synchronize()
        end
        cfgs = Array(u)
        drift = maximum(abs, vec(mean(Float64, cfgs; dims=(1, 2))) - M0)
        output = joinpath(
            directory,
            "modelB_L$(L)_Ne512_hmc_t$(time_tag(time))_chain_" *
            "dt0.002_seed$(seed).jld2",
        )
        @assert !ispath(output)
        jldsave(
            output; cfgs, L, Nensemble, time=Float32(time),
            kappa=κ, lambda=λ, gamma=γ, temperature=T, dt,
            noise_seed=seed, source_path=source,
            generation_seconds=seconds,
            max_abs_conservation_drift=drift,
        )
        println(
            "L=$L t=$(time_tag(time)): $(round(seconds; digits=3)) s, " *
            "max|ΔM|=$drift",
        )
        flush(stdout)
        previous = time
    end
end
