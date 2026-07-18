include(joinpath(@__DIR__, "solver_variants.jl"))

function run_variant(name, u0; synchronize_callbacks, noise_builder, t_end)
    CUDA.seed!(314159)
    elapsed, solution = synchronized_seconds() do
        solve_sosra(
            u0;
            synchronize_callbacks,
            noise_builder,
            dt=DT_DEFAULT,
            t_end,
        )
    end
    final = solution.u[end]
    println(name, " seconds=", elapsed,
            " finite=", all(isfinite, final),
            " maxabs=", maximum(abs, final),
            " max_mass=", maximum(abs.(sum(final; dims=(1, 2)))))
    return elapsed, copy(final)
end

function main()
    t_end = length(ARGS) >= 1 ? parse(Float32, ARGS[1]) : 0.2f0
    u0 = make_u0()

    # Compile both callback paths on a small state first.
    warm = make_u0(L=8, ensembles=2, seed=11)
    solve_sosra(warm; t_end=DT_DEFAULT)
    solve_sosra(
        warm; synchronize_callbacks=false,
        noise_builder=async_noise_process, t_end=DT_DEFAULT,
    )
    CUDA.synchronize()

    baseline_time, baseline = run_variant(
        "baseline_sync", u0;
        synchronize_callbacks=true,
        noise_builder=modelb_noise_process,
        t_end,
    )
    async_time, async = run_variant(
        "async_same_stream", u0;
        synchronize_callbacks=false,
        noise_builder=async_noise_process,
        t_end,
    )
    println("trajectory_max_abs_difference=", max_abs_difference(baseline, async))
    println("speedup=", baseline_time / async_time)
end

main()
