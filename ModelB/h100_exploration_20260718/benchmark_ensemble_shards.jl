include(joinpath(@__DIR__, "solver_kernel_variants.jl"))
include(joinpath(@__DIR__, "rng_variants.jl"))

using .H100RNGVariants
using Printf

const SHARD_RSWM1 = DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1)

function solve_shard(shard, seed, t_end, rng_mode)
    CUDA.seed!(seed)
    noise_builder = if rng_mode == "default"
        rswm1_linear_fused_noise_process
    elseif rng_mode == "philox"
        (t, u) -> packed_noise_process128(
            t, u;
            rng_kind=:library_philox,
            seed,
            synchronize_callbacks=false,
            rswm=SHARD_RSWM1,
        )
    else
        error("rng_mode must be default or philox")
    end
    return solve_sosra(
        shard;
        drift_launch! = drift_linear128_best!,
        noise_builder,
        synchronize_callbacks = false,
        dt = DT_DEFAULT,
        t_end,
    )
end

function run_shards(u0, nshards, t_end, base_seed, rng_mode)
    ranges = ensemble_ranges(size(u0, 3), nshards)
    # Prepare slices outside the timed region.  The ordinary single-solve path
    # also copies u0 inside make_modelb_problem, so each solve still has one
    # like-for-like initial-state copy during timing.
    shards = [copy(@view u0[:, :, range]) for range in ranges]
    solutions = Vector{Any}(undef, nshards)
    streams = [CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING) for _ in 1:nshards]

    CUDA.device_synchronize()
    elapsed = @elapsed begin
        @sync for shard_id in 1:nshards
            Threads.@spawn begin
                CUDA.stream!(streams[shard_id]) do
                    solutions[shard_id] = solve_shard(
                        shards[shard_id], base_seed + shard_id - 1, t_end, rng_mode
                    )
                    CUDA.synchronize(streams[shard_id])
                end
            end
        end
        CUDA.device_synchronize()
    end
    return elapsed, solutions
end

function benchmark_shard_count(u0, nshards, t_end, repetitions, rng_mode)
    warm = make_u0(L=128, ensembles=8 * nshards, seed=19)
    run_shards(warm, nshards, DT_DEFAULT, 111, rng_mode)

    times = Float64[]
    for repetition in 1:repetitions
        elapsed, solutions = run_shards(
            u0, nshards, t_end, 314159 + 100 * repetition, rng_mode
        )
        finals = [solution.u[end] for solution in solutions]
        finite = all(all(isfinite, final) for final in finals)
        maxabs = maximum(maximum(abs, final) for final in finals)
        max_mass = maximum(
            maximum(abs.(sum(final; dims=(1, 2)))) for final in finals
        )
        accepts = sum(solution.stats.naccept for solution in solutions)
        @printf(
            "rng=%s shards=%d repetition=%d seconds=%.6f finite=%s maxabs=%.9g max_mass=%.9g total_accepts=%d\n",
            rng_mode,
            nshards,
            repetition,
            elapsed,
            string(finite),
            maxabs,
            max_mass,
            accepts,
        )
        push!(times, elapsed)
        solutions = nothing
        finals = nothing
        GC.gc(false)
    end
    warm_times = repetitions > 1 ? times[2:end] : times
    @printf(
        "rng=%s shards=%d median=%.6f warm_median=%.6f ensemble_steps_per_second=%.3f\n",
        rng_mode,
        nshards,
        median(times),
        median(warm_times),
        ENSEMBLES_DEFAULT * round(Int, t_end / DT_DEFAULT) / median(warm_times),
    )
    return median(warm_times)
end

function main()
    shard_argument = length(ARGS) >= 1 ? ARGS[1] : "all"
    t_end = length(ARGS) >= 2 ? parse(Float32, ARGS[2]) : TEND_DEFAULT
    repetitions = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    rng_argument = length(ARGS) >= 4 ? ARGS[4] : "both"
    shard_counts = if shard_argument == "all"
        (1, 2, 4, 8)
    else
        nshards = parse(Int, shard_argument)
        nshards in (1, 2, 4, 8) || error("nshards must be 1, 2, 4, 8, or all")
        (nshards,)
    end
    rng_modes = rng_argument == "both" ? ("default", "philox") : (rng_argument,)
    all(mode -> mode in ("default", "philox"), rng_modes) ||
        error("rng mode must be default, philox, or both")
    u0 = make_u0()
    for rng_mode in rng_modes, nshards in shard_counts
        benchmark_shard_count(u0, nshards, t_end, repetitions, rng_mode)
    end
end

main()
