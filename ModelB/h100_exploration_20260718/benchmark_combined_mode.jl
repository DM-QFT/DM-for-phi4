include(joinpath(@__DIR__, "solver_kernel_variants.jl"))

function select_variant(mode)
    if mode == "linear_rswm1_async"
        return drift_linear128_best!, rswm1_linear_fused_noise_process, false
    elseif mode == "linear_rswm3_async"
        return drift_linear128_best!, linear_fused_noise_process, false
    elseif mode == "shared_rswm1_async"
        return drift_shared128_best!, rswm1_linear_fused_noise_process, false
    else
        error("unknown combined mode: $mode")
    end
end

function main()
    mode = ARGS[1]
    t_end = length(ARGS) >= 2 ? parse(Float32, ARGS[2]) : TEND_DEFAULT
    repetitions = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
    drift_launch!, noise_builder, synchronize_callbacks = select_variant(mode)

    warm = make_u0(L=128, ensembles=2, seed=11)
    solve_sosra(
        warm; drift_launch!, noise_builder, synchronize_callbacks,
        dt=DT_DEFAULT, t_end=DT_DEFAULT,
    )
    CUDA.synchronize()

    u0 = make_u0()
    times = Float64[]
    for repetition in 1:repetitions
        CUDA.seed!(314159 + repetition)
        elapsed, solution = synchronized_seconds() do
            solve_sosra(
                u0; drift_launch!, noise_builder, synchronize_callbacks,
                dt=DT_DEFAULT, t_end,
            )
        end
        final = solution.u[end]
        println(mode, " repetition=", repetition,
                " seconds=", elapsed,
                " retcode=", solution.retcode,
                " naccept=", solution.stats.naccept,
                " finite=", all(isfinite, final),
                " maxabs=", maximum(abs, final),
                " max_mass=", maximum(abs.(sum(final; dims=(1, 2)))))
        push!(times, elapsed)
        solution = nothing
        final = nothing
        GC.gc(false)
    end
    println(mode, " median=", median(times),
            " warm_median=", repetitions > 1 ? median(times[2:end]) : times[1])
end

main()
