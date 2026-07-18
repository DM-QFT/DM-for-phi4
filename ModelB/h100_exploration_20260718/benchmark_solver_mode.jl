include(joinpath(@__DIR__, "solver_variants.jl"))

function main()
    mode = ARGS[1]
    t_end = length(ARGS) >= 2 ? parse(Float32, ARGS[2]) : TEND_DEFAULT
    repetitions = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
    is_async = mode in ("async", "rswm1_async")
    noise_builder = if mode == "sync"
        modelb_noise_process
    elseif mode == "async"
        async_noise_process
    elseif mode == "rswm1_sync"
        rswm1_noise_process
    elseif mode == "rswm1_async"
        rswm1_async_noise_process
    else
        error("unknown mode: $mode")
    end
    synchronize_callbacks = !is_async

    warm = make_u0(L=8, ensembles=2, seed=11)
    solve_sosra(
        warm; synchronize_callbacks, noise_builder,
        dt=DT_DEFAULT, t_end=DT_DEFAULT,
    )
    CUDA.synchronize()

    u0 = make_u0()
    times = Float64[]
    for repetition in 1:repetitions
        CUDA.seed!(314159 + repetition)
        elapsed, solution = synchronized_seconds() do
            solve_sosra(
                u0; synchronize_callbacks, noise_builder,
                dt=DT_DEFAULT, t_end,
            )
        end
        final = solution.u[end]
        println(mode, " repetition=", repetition,
                " seconds=", elapsed,
                " retcode=", solution.retcode,
                " tfinal=", solution.t[end],
                " naccept=", solution.stats.naccept,
                " nreject=", solution.stats.nreject,
                " finite=", all(isfinite, final),
                " maxabs=", maximum(abs, final))
        push!(times, elapsed)
        solution = nothing
        final = nothing
        GC.gc(false)
    end
    println(mode, " median=", median(times),
            " warm_median=", repetitions > 1 ? median(times[2:end]) : times[1])
end

main()
