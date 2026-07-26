using CUDA, Printf, Statistics

include(joinpath(@__DIR__, "fused_philox.jl"))

function benchmark_shape(L, Nensemble)
    steps = 1000
    repetitions = 5
    dt = 0.002f0
    parameters = (
        0.56f0,
        -0.328f0,
        0.088f0,
        1.0f0,
        sqrt(2.0f0),
    )
    seed = UInt64(20260726)
    u0 = CUDA.randn(Float32, L, L, Nensemble)

    standard_u = copy(u0)
    standard_buffers = modelb_fixed_buffers(standard_u; seed)
    modelb_fixed_step!(
        standard_u, standard_buffers, dt, parameters...,
    )

    padded_u = copy(u0)
    padded_buffers = padded_curand_buffers(padded_u; seed)
    modelb_fixed_step!(
        padded_u, padded_buffers, dt, parameters...,
    )

    gpuarrays_u = copy(u0)
    gpuarrays_workspace = gpuarrays_buffers(gpuarrays_u; seed)
    modelb_fixed_step!(
        gpuarrays_u, gpuarrays_workspace, dt, parameters...,
    )

    direct_u = copy(u0)
    direct_buffers = modelb_fixed_buffers(direct_u; seed)
    direct_curand_step!(
        direct_u, direct_buffers, dt, parameters...,
    )

    tiled_u = copy(u0)
    tiled_buffers = fused_philox_buffers(tiled_u; seed)
    tiled_philox_step!(
        tiled_u, tiled_buffers, dt, parameters...,
    )
    CUDA.synchronize()

    standard_times = Float64[]
    padded_times = Float64[]
    gpuarrays_times = Float64[]
    direct_times = Float64[]
    tiled_times = Float64[]
    for repetition in 1:repetitions
        repetition_seed = seed + UInt64(repetition)
        copyto!(standard_u, u0)
        seed_modelb_fixed!(standard_buffers, repetition_seed)
        CUDA.synchronize()
        standard_seconds = CUDA.@elapsed begin
            for _ in 1:steps
                modelb_fixed_step!(
                    standard_u,
                    standard_buffers,
                    dt,
                    parameters...,
                )
            end
        end
        push!(standard_times, standard_seconds)

        copyto!(padded_u, u0)
        seed_modelb_fixed!(padded_buffers, repetition_seed)
        CUDA.synchronize()
        padded_seconds = CUDA.@elapsed begin
            for _ in 1:steps
                modelb_fixed_step!(
                    padded_u,
                    padded_buffers,
                    dt,
                    parameters...,
                )
            end
        end
        push!(padded_times, padded_seconds)

        copyto!(gpuarrays_u, u0)
        seed_gpuarrays!(gpuarrays_workspace, repetition_seed)
        CUDA.synchronize()
        gpuarrays_seconds = CUDA.@elapsed begin
            for _ in 1:steps
                modelb_fixed_step!(
                    gpuarrays_u,
                    gpuarrays_workspace,
                    dt,
                    parameters...,
                )
            end
        end
        push!(gpuarrays_times, gpuarrays_seconds)

        copyto!(direct_u, u0)
        seed_modelb_fixed!(direct_buffers, repetition_seed)
        CUDA.synchronize()
        direct_seconds = CUDA.@elapsed begin
            for _ in 1:steps
                direct_curand_step!(
                    direct_u,
                    direct_buffers,
                    dt,
                    parameters...,
                )
            end
        end
        push!(direct_times, direct_seconds)

        copyto!(tiled_u, u0)
        seed_fused_philox!(tiled_buffers, repetition_seed)
        CUDA.synchronize()
        tiled_seconds = CUDA.@elapsed tiled_philox_steps!(
            tiled_u,
            tiled_buffers,
            steps,
            dt,
            parameters...,
        )
        push!(tiled_times, tiled_seconds)
    end
    standard = median(standard_times) / steps
    padded = median(padded_times) / steps
    gpuarrays = median(gpuarrays_times) / steps
    direct = median(direct_times) / steps
    tiled = median(tiled_times) / steps
    @printf(
        "%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.4f\t%.4f\t%.4f\t%.4f\n",
        L,
        Nensemble,
        length(u0),
        1.0e3 * standard,
        1.0e3 * padded,
        1.0e3 * gpuarrays,
        1.0e3 * direct,
        1.0e3 * tiled,
        standard / padded,
        standard / gpuarrays,
        standard / direct,
        standard / tiled,
    )
    return (
        L=L,
        Nensemble=Nensemble,
        elements=length(u0),
        standard_ms=1.0e3 * standard,
        padded_ms=1.0e3 * padded,
        gpuarrays_ms=1.0e3 * gpuarrays,
        direct_ms=1.0e3 * direct,
        tiled_ms=1.0e3 * tiled,
        padded_speedup=standard / padded,
        gpuarrays_speedup=standard / gpuarrays,
        direct_speedup=standard / direct,
        tiled_speedup=standard / tiled,
    )
end

function main()
    rows = NamedTuple[]
    for (L, Nensemble) in (
        (32, 5120),
        (32, 8192),
        (128, 512),
        (256, 128),
    )
        push!(rows, benchmark_shape(L, Nensemble))
    end
    output = joinpath(@__DIR__, "output")
    mkpath(output)
    open(joinpath(output, "shape_timings.tsv"), "w") do io
        println(
            io,
            "L\tNensemble\telements\tstandard_ms\t",
            "padded_ms\tgpuarrays_ms\tdirect_ms\ttiled_ms\t",
            "padded_speedup\tgpuarrays_speedup\t",
            "direct_speedup\ttiled_speedup",
        )
        for row in rows
            @printf(
                io,
                "%d\t%d\t%d\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n",
                row.L,
                row.Nensemble,
                row.elements,
                row.standard_ms,
                row.padded_ms,
                row.gpuarrays_ms,
                row.direct_ms,
                row.tiled_ms,
                row.padded_speedup,
                row.gpuarrays_speedup,
                row.direct_speedup,
                row.tiled_speedup,
            )
        end
    end
end

main()
