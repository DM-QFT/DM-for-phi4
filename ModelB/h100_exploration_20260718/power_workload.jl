include(joinpath(@__DIR__, "solver_kernel_variants.jl"))
include(joinpath(@__DIR__, "rng_variants.jl"))
using .H100RNGVariants

const POWER_RSWM1 = DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1)
const POWER_SOLVE_TEND = 2.0f0

"""Select a solver path without changing the production ModelB.jl file."""
function select_power_variant(mode)
    if mode == "baseline_sync"
        return launch_modelb_drift!, modelb_noise_process, true
    elseif mode == "rswm1_async"
        return launch_modelb_drift!, rswm1_async_noise_process, false
    elseif mode == "direct_fused_rswm1"
        return drift_linear128_best!, rswm1_linear_fused_noise_process, false
    elseif mode == "shared_fused_rswm1"
        return drift_shared128_best!, rswm1_linear_fused_noise_process, false
    elseif mode == "direct_packed_philox_rswm1"
        packed_philox = (t, u) -> H100RNGVariants.packed_noise_process128(
            t,
            u;
            rng_kind=:library_philox,
            seed=314159,
            synchronize_callbacks=false,
            rswm=POWER_RSWM1,
        )
        return drift_linear128_best!, packed_philox, false
    end
    error("unknown mode: $mode")
end

function epoch_seconds()
    return time()
end

function marker(label)
    println(label, " epoch=", epoch_seconds())
    flush(stdout)
end

function power_synchronized_seconds(f)
    CUDA.synchronize()
    start_ns = time_ns()
    value = f()
    CUDA.synchronize()
    return Float64(time_ns() - start_ns) * 1.0e-9, value
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "baseline_sync"
    requested_window_seconds = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 20.0
    @assert requested_window_seconds == 20.0
    requested_t_end = length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : POWER_SOLVE_TEND
    @assert requested_t_end == POWER_SOLVE_TEND
    drift_launch!, noise_builder, synchronize_callbacks = select_power_variant(mode)

    # Complete one full-size/full-length solve before the marker.  Besides JIT
    # compilation this populates CUDA's large-array memory pool and reaches a
    # steady clock/power state.  Do not CUDA.reclaim() afterward.
    u0 = make_u0()
    warm_solution = solve_sosra(
        u0;
        drift_launch!, noise_builder, synchronize_callbacks,
        dt=DT_DEFAULT, t_end=POWER_SOLVE_TEND,
    )
    CUDA.synchronize()
    @assert string(warm_solution.retcode) == "Success"
    warm_final = warm_solution.u[end]
    @assert all(isfinite, warm_final)
    maximum(abs, warm_final)
    CUDA.synchronize()
    warm_solution = nothing
    warm_final = nothing
    GC.gc(false)
    CUDA.synchronize()
    marker("POWER_WINDOW_START")

    repetitions = Dict(
        "baseline_sync" => 4,
        "rswm1_async" => 5,
        "direct_fused_rswm1" => 7,
        "shared_fused_rswm1" => 7,
        "direct_packed_philox_rswm1" => 9,
    )
    repetitions = repetitions[mode]
    solve_seconds = Float64[]
    for repetition in 1:repetitions
        elapsed, solution = power_synchronized_seconds() do
            solve_sosra(
                u0;
                drift_launch!, noise_builder, synchronize_callbacks,
                dt=DT_DEFAULT, t_end=POWER_SOLVE_TEND,
            )
        end
        final = solution.u[end]
        @assert string(solution.retcode) == "Success"
        @assert all(isfinite, final)
        push!(solve_seconds, elapsed)
        println(
            "POWER_SOLVE mode=", mode,
            " iteration=", repetition,
            " seconds=", elapsed,
            " naccept=", solution.stats.naccept,
            " maxabs=", maximum(abs, final),
        )
        flush(stdout)
        solution = nothing
    end
    CUDA.synchronize()
    marker("POWER_WINDOW_END")
    println(
        "POWER_SUMMARY mode=", mode,
        " iterations=", length(solve_seconds),
        " measured_solve_seconds=", sum(solve_seconds),
        " solve_t_end=", POWER_SOLVE_TEND,
        " median_solve_seconds=", median(solve_seconds),
        " total_lattice_steps=", length(solve_seconds) * round(Int, POWER_SOLVE_TEND / DT_DEFAULT),
    )
end

main()
