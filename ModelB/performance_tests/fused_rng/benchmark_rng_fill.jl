using CUDA, Printf, Random, Statistics

include(joinpath(@__DIR__, "fused_philox.jl"))

function benchmark_fill!(fill!, seed!; repetitions=7, calls=1000)
    times = Float64[]
    fill!()
    CUDA.synchronize()
    for repetition in 1:repetitions
        seed!(UInt64(20260726 + repetition))
        CUDA.synchronize()
        seconds = CUDA.@elapsed begin
            for _ in 1:calls
                fill!()
            end
        end
        push!(times, seconds / calls)
    end
    return median(times)
end

function main()
    L, Nensemble = 32, 5120
    links = CUDA.zeros(Float32, L, L, Nensemble, 4)
    padded_links =
        CUDA.zeros(Float32, nextpow(2, length(links)))
    seed = UInt64(20260726)
    library_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    gpuarrays_rng = CUDA.RNG(seed)
    direct_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    padded_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )

    rows = (
        (
            method="library_wrapper",
            generated=nextpow(2, length(links)),
            seconds=benchmark_fill!(
                () -> Random.randn!(library_rng, links),
                seed_value -> Random.seed!(library_rng, seed_value),
            ),
        ),
        (
            method="library_preallocated_pow2",
            generated=length(padded_links),
            seconds=benchmark_fill!(
                () -> Random.randn!(padded_rng, padded_links),
                seed_value -> Random.seed!(padded_rng, seed_value),
            ),
        ),
        (
            method="gpuarrays_public",
            generated=length(links),
            seconds=benchmark_fill!(
                () -> Random.randn!(gpuarrays_rng, links),
                seed_value -> Random.seed!(gpuarrays_rng, seed_value),
            ),
        ),
        (
            method="curand_direct_even",
            generated=length(links),
            seconds=benchmark_fill!(
                () -> direct_curand_fill!(direct_rng, links),
                seed_value -> Random.seed!(direct_rng, seed_value),
            ),
        ),
    )

    output = joinpath(@__DIR__, "output")
    mkpath(output)
    open(joinpath(output, "rng_fill_timings.tsv"), "w") do io
        println(io, "method\trequested\tgenerated\tms")
        for row in rows
            @printf(
                io,
                "%s\t%d\t%d\t%.8g\n",
                row.method,
                length(links),
                row.generated,
                1.0e3 * row.seconds,
            )
        end
    end
    for row in rows
        @printf(
            "%s\trequested=%d\tgenerated=%d\t%.6f ms\n",
            row.method,
            length(links),
            row.generated,
            1.0e3 * row.seconds,
        )
    end
end

main()
