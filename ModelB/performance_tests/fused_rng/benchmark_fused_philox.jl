using CUDA, Printf, Random, Statistics

include(joinpath(@__DIR__, "fused_philox.jl"))

function gpu_seconds(f)
    CUDA.synchronize()
    return CUDA.@elapsed f()
end

function main()
    L, Nensemble = 256, 128
    steps = 1000
    repetitions = 7
    dt = 0.002f0
    two_kappa = 0.56f0
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * 0.022f0) - 8.0f0 * 0.28f0
    c3 = 0.088f0
    inv_gamma = 1.0f0
    noise_amplitude = sqrt(2.0f0)
    seed = UInt64(20260726)

    CUDA.seed!(20260725)
    u0 = CUDA.randn(Float32, L, L, Nensemble)

    u_curand = copy(u0)
    curand_buffers = modelb_fixed_buffers(u_curand; seed)
    modelb_fixed_step!(
        u_curand, curand_buffers, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
    )

    u_gpuarrays = copy(u0)
    gpuarrays_buffers = (
        h01=similar(u_gpuarrays),
        h02=similar(u_gpuarrays),
        k1=similar(u_gpuarrays),
        q=similar(u_gpuarrays),
        links=similar(
            u_gpuarrays,
            Float32,
            (size(u_gpuarrays)..., 4),
        ),
        rng=CUDA.RNG(seed),
        lattice=Val(L),
    )
    modelb_fixed_step!(
        u_gpuarrays, gpuarrays_buffers, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
    )

    u_fused = copy(u0)
    fused_buffers = fused_philox_buffers(u_fused; seed)
    fused_philox_step!(
        u_fused, fused_buffers, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
    )

    u_tiled = copy(u0)
    tiled_buffers = fused_philox_buffers(u_tiled; seed)
    tiled_philox_step!(
        u_tiled, tiled_buffers, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
    )

    u_overlapped = copy(u0)
    overlapped_buffers =
        overlapped_curand_buffers(u_overlapped; seed)
    overlapped_curand_steps!(
        u_overlapped, overlapped_buffers, 1, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
    )
    CUDA.synchronize()

    curand_times = Float64[]
    gpuarrays_times = Float64[]
    fused_times = Float64[]
    tiled_times = Float64[]
    overlapped_times = Float64[]
    for repetition in 1:repetitions
        repetition_seed = seed + UInt64(repetition)

        copyto!(u_curand, u0)
        seed_modelb_fixed!(curand_buffers, repetition_seed)
        push!(
            curand_times,
            gpu_seconds(
                () -> begin
                    for _ in 1:steps
                        modelb_fixed_step!(
                            u_curand, curand_buffers, dt,
                            two_kappa, m02, c3, inv_gamma,
                            noise_amplitude,
                        )
                    end
                end,
            ),
        )

        copyto!(u_gpuarrays, u0)
        Random.seed!(gpuarrays_buffers.rng, repetition_seed)
        push!(
            gpuarrays_times,
            gpu_seconds(
                () -> begin
                    for _ in 1:steps
                        modelb_fixed_step!(
                            u_gpuarrays, gpuarrays_buffers, dt,
                            two_kappa, m02, c3, inv_gamma,
                            noise_amplitude,
                        )
                    end
                end,
            ),
        )

        copyto!(u_fused, u0)
        seed_fused_philox!(fused_buffers, repetition_seed)
        push!(
            fused_times,
            gpu_seconds(
                () -> fused_philox_steps!(
                    u_fused, fused_buffers, steps, dt,
                    two_kappa, m02, c3, inv_gamma, noise_amplitude,
                ),
            ),
        )

        copyto!(u_tiled, u0)
        seed_fused_philox!(tiled_buffers, repetition_seed)
        push!(
            tiled_times,
            gpu_seconds(
                () -> tiled_philox_steps!(
                    u_tiled, tiled_buffers, steps, dt,
                    two_kappa, m02, c3, inv_gamma, noise_amplitude,
                ),
            ),
        )

        copyto!(u_overlapped, u0)
        seed_overlapped_curand!(
            overlapped_buffers,
            repetition_seed,
        )
        push!(
            overlapped_times,
            gpu_seconds(
                () -> overlapped_curand_steps!(
                    u_overlapped, overlapped_buffers, steps, dt,
                    two_kappa, m02, c3, inv_gamma, noise_amplitude,
                ),
            ),
        )
    end

    n = Int32(length(u_fused))
    fused_kernel = @cuda launch=false fused_philox_stage1_kernel!(
        fused_buffers.h01,
        fused_buffers.h02,
        fused_buffers.k1,
        fused_buffers.q,
        u_fused,
        sqrt(dt),
        dt,
        two_kappa,
        m02,
        c3,
        inv_gamma,
        noise_amplitude,
        seed,
        UInt64(0),
        Val(L),
        n,
    )
    curand_kernel = @cuda launch=false modelb_fixed_stage1_kernel!(
        curand_buffers.h01,
        curand_buffers.h02,
        curand_buffers.k1,
        curand_buffers.q,
        u_curand,
        curand_buffers.links,
        sqrt(dt),
        dt,
        two_kappa,
        m02,
        c3,
        inv_gamma,
        noise_amplitude,
        Val(L),
        n,
    )
    tiled_kernel = @cuda launch=false tiled_philox_stage1_kernel!(
        tiled_buffers.h01,
        tiled_buffers.h02,
        tiled_buffers.k1,
        tiled_buffers.q,
        u_tiled,
        sqrt(dt),
        dt,
        two_kappa,
        m02,
        c3,
        inv_gamma,
        noise_amplitude,
        seed,
        UInt64(0),
        Val(L),
        Val(8),
    )

    curand_seconds = median(curand_times)
    gpuarrays_seconds = median(gpuarrays_times)
    fused_seconds = median(fused_times)
    tiled_seconds = median(tiled_times)
    overlapped_seconds = median(overlapped_times)
    curand_drift = maximum(
        abs,
        mean(u_curand; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    gpuarrays_drift = maximum(
        abs,
        mean(u_gpuarrays; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    fused_drift = maximum(
        abs,
        mean(u_fused; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    tiled_drift = maximum(
        abs,
        mean(u_tiled; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    overlapped_drift = maximum(
        abs,
        mean(u_overlapped; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )

    @printf("elements\t%d\n", length(u0))
    @printf("steps\t%d\n", steps)
    @printf(
        "curand_ms_per_step\t%.6f\n",
        1.0e3 * curand_seconds / steps,
    )
    @printf(
        "gpuarrays_ms_per_step\t%.6f\n",
        1.0e3 * gpuarrays_seconds / steps,
    )
    @printf(
        "fused_ms_per_step\t%.6f\n",
        1.0e3 * fused_seconds / steps,
    )
    @printf(
        "tiled_ms_per_step\t%.6f\n",
        1.0e3 * tiled_seconds / steps,
    )
    @printf(
        "overlapped_ms_per_step\t%.6f\n",
        1.0e3 * overlapped_seconds / steps,
    )
    @printf("curand_over_fused\t%.4f\n", curand_seconds / fused_seconds)
    @printf("curand_over_tiled\t%.4f\n", curand_seconds / tiled_seconds)
    @printf(
        "curand_over_overlapped\t%.4f\n",
        curand_seconds / overlapped_seconds,
    )
    @printf(
        "curand_over_gpuarrays\t%.4f\n",
        curand_seconds / gpuarrays_seconds,
    )
    @printf("curand_stage1_registers\t%d\n", CUDA.registers(curand_kernel))
    @printf("fused_stage1_registers\t%d\n", CUDA.registers(fused_kernel))
    @printf("tiled_stage1_registers\t%d\n", CUDA.registers(tiled_kernel))
    @printf("curand_max_abs_mean_drift\t%.8e\n", curand_drift)
    @printf("gpuarrays_max_abs_mean_drift\t%.8e\n", gpuarrays_drift)
    @printf("fused_max_abs_mean_drift\t%.8e\n", fused_drift)
    @printf("tiled_max_abs_mean_drift\t%.8e\n", tiled_drift)
    @printf(
        "overlapped_max_abs_mean_drift\t%.8e\n",
        overlapped_drift,
    )

    output = joinpath(@__DIR__, "output")
    mkpath(output)
    open(joinpath(output, "variant_timings.tsv"), "w") do io
        println(
            io,
            "method\tms_per_step\tspeedup\tstage1_registers\t",
            "link_workspace_mib\tmax_M_drift",
        )
        rows = (
            (
                "standard_curand",
                curand_seconds,
                CUDA.registers(curand_kernel),
                4,
                curand_drift,
            ),
            (
                "separate_gpuarrays",
                gpuarrays_seconds,
                CUDA.registers(curand_kernel),
                4,
                gpuarrays_drift,
            ),
            (
                "direct_stateless",
                fused_seconds,
                CUDA.registers(fused_kernel),
                0,
                fused_drift,
            ),
            (
                "tiled_stateless",
                tiled_seconds,
                CUDA.registers(tiled_kernel),
                0,
                tiled_drift,
            ),
            (
                "overlapped_curand",
                overlapped_seconds,
                CUDA.registers(curand_kernel),
                8,
                overlapped_drift,
            ),
        )
        for (method, seconds, registers, link_fields, drift) in rows
            @printf(
                io,
                "%s\t%.8g\t%.8g\t%d\t%.8g\t%.8g\n",
                method,
                1.0e3 * seconds / steps,
                curand_seconds / seconds,
                registers,
                link_fields * sizeof(Float32) * length(u0) / 2.0^20,
                drift,
            )
        end
    end
end

main()
