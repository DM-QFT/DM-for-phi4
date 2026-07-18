using CUDA
using Statistics

include(joinpath(@__DIR__, "fused_fixed_sosra.jl"))
using .FusedFixedSOSRA128

const FUSED_POWER_TEND = 2.0f0

function make_fused_u0(ensembles; seed=20260718)
    CUDA.seed!(seed)
    u = CUDA.randn(Float32, 128, 128, ensembles)
    u .-= mean(u; dims=(1, 2))
    return u
end

function synchronized_seconds(f)
    CUDA.synchronize()
    start_ns = time_ns()
    f()
    CUDA.synchronize()
    return Float64(time_ns() - start_ns) * 1.0e-9
end

function marker(label)
    println(label, " epoch=", time())
    flush(stdout)
end

function main()
    requested_window_seconds = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 20.0
    @assert requested_window_seconds == 20.0
    requested_t_end = length(ARGS) >= 2 ? parse(Float32, ARGS[2]) : FUSED_POWER_TEND
    @assert requested_t_end == FUSED_POWER_TEND
    threads = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1024
    blocks = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 264
    noise_threads = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 512
    noise_blocks = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 528

    u0 = make_fused_u0(1024)
    u = similar(u0)
    workspace = allocate_workspace(u0; rng_kind=:library_philox, seed=314159)
    params = ModelBParameters()
    copyto!(u, u0)
    solve_fixed_sosra!(
        u, workspace, params;
        t_end=FUSED_POWER_TEND,
        dt=0.002f0, threads, blocks, noise_threads, noise_blocks,
    )
    CUDA.synchronize()
    @assert all(isfinite, u)
    maximum(abs, u)
    CUDA.synchronize()
    marker("POWER_WINDOW_START")

    repetitions = 30
    timings = Float64[]
    for repetition in 1:repetitions
        copyto!(u, u0)
        seed_packed_rng!(workspace, 314159 + repetition)
        elapsed = synchronized_seconds() do
            solve_fixed_sosra!(
                u, workspace, params;
                dt=0.002f0, t_end=FUSED_POWER_TEND,
                threads, blocks, noise_threads, noise_blocks,
            )
        end
        @assert all(isfinite, u)
        push!(timings, elapsed)
        println(
            "POWER_SOLVE mode=custom_fused_philox",
            " iteration=", repetition,
            " seconds=", elapsed,
            " naccept=", round(Int, FUSED_POWER_TEND / 0.002f0),
            " maxabs=", maximum(abs, u),
        )
        flush(stdout)
    end
    CUDA.synchronize()
    marker("POWER_WINDOW_END")
    println(
        "POWER_SUMMARY mode=custom_fused_philox",
        " iterations=", length(timings),
        " measured_solve_seconds=", sum(timings),
        " solve_t_end=", FUSED_POWER_TEND,
        " median_solve_seconds=", median(timings),
        " threads=", threads,
        " blocks=", blocks,
        " noise_threads=", noise_threads,
        " noise_blocks=", noise_blocks,
    )
end

main()
