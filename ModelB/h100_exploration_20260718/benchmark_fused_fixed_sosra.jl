using CUDA
using Statistics
using Printf

include(joinpath(@__DIR__, "fused_fixed_sosra.jl"))
using .FusedFixedSOSRA128

function make_u0(ensembles; seed=20260718)
    CUDA.seed!(seed)
    u = CUDA.randn(Float32, 128, 128, ensembles)
    u .-= mean(u; dims=(1, 2))
    return u
end

function synchronized_seconds(f)
    CUDA.synchronize()
    return Float64(CUDA.@elapsed f())
end

function median_kernel_seconds(f; warmup=3, samples=12)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    values = Float64[]
    for _ in 1:samples
        push!(values, synchronized_seconds(f))
    end
    return median(values), minimum(values), maximum(values)
end

function print_resource_summary(u, workspace, params)
    n = Int32(length(u))
    noise = @cuda launch=false FusedFixedSOSRA128.packed_conserved_increments_kernel!(
        workspace.dW, workspace.dZ, workspace.links, sqrt(0.002f0), n
    )
    stage1 = @cuda launch=false FusedFixedSOSRA128.sosra_stage1_kernel!(
        workspace.h01, workspace.k1, u, workspace.dW, workspace.dZ,
        0.002f0, params, n,
    )
    stage2 = @cuda launch=false FusedFixedSOSRA128.sosra_stage2_kernel!(
        workspace.h02, workspace.k2, u, workspace.h01, workspace.k1,
        workspace.dW, workspace.dZ, 0.002f0, params, n,
    )
    stage3 = @cuda launch=false FusedFixedSOSRA128.sosra_stage3_kernel!(
        u, workspace.h02, workspace.k1, workspace.k2, workspace.dW,
        0.002f0, params, n,
    )
    for (name, kernel) in (
        ("packed_noise", noise), ("stage1", stage1),
        ("stage2", stage2), ("stage3", stage3),
    )
        println(
            "RESOURCE\t", name,
            "\tregisters=", CUDA.registers(kernel),
            "\tmaxthreads=", CUDA.maxthreads(kernel),
            "\tlocal=", CUDA.memory(kernel).local,
            "\tshared=", CUDA.memory(kernel).shared,
            "\tlaunch=", CUDA.launch_configuration(kernel.fun),
        )
    end
    return (
        noise_maxthreads=CUDA.maxthreads(noise),
        stage_maxthreads=minimum(CUDA.maxthreads.((stage1, stage2, stage3))),
    )
end

function kernel_breakdown(u, workspace, params, limits)
    fill_packed_normals!(workspace)
    launch_packed_increments!(workspace, 0.002f0)
    launch_stage1!(u, workspace, 0.002f0, params)
    launch_stage2!(u, workspace, 0.002f0, params)
    CUDA.synchronize()

    println("KERNEL\tname\tthreads\tblocks\tmedian_ms\tmin_ms\tmax_ms")
    rng_t = median_kernel_seconds(() -> fill_packed_normals!(workspace))
    println("KERNEL\tpacked_curand\tNA\tNA\t", join(1.0e3 .* collect(rng_t), '\t'))

    for threads in (256, 512, 768, 1024)
        threads > limits.noise_maxthreads && continue
        for blocks in (264, 528, 1056)
            t = median_kernel_seconds(
                () -> launch_packed_increments!(workspace, 0.002f0; threads, blocks);
                samples=8,
            )
            println(
                "KERNEL\tpacked_noise\t", threads, '\t', blocks, '\t',
                join(1.0e3 .* collect(t), '\t'),
            )
        end
    end

    for threads in (256, 512, 768, 1024)
        threads > limits.stage_maxthreads && continue
        for blocks in (264, 528, 1056)
            t1 = median_kernel_seconds(
                () -> launch_stage1!(u, workspace, 0.002f0, params; threads, blocks);
                samples=6,
            )
            t2 = median_kernel_seconds(
                () -> launch_stage2!(u, workspace, 0.002f0, params; threads, blocks);
                samples=6,
            )
            t3 = median_kernel_seconds(
                () -> launch_stage3!(u, workspace, 0.002f0, params; threads, blocks);
                samples=6,
            )
            println(
                "KERNEL\tstages_sum\t", threads, '\t', blocks, '\t',
                1.0e3 * sum(first.((t1, t2, t3))), "\tNA\tNA",
            )
        end
    end
end

function short_sweep(u0, workspace, params, limits; steps=50)
    u = similar(u0)
    println("SHORT\tthreads\tblocks\tseconds\tstep_us\tfinite\tmaxabs")
    for threads in (256, 512, 768, 1024)
        threads > limits.stage_maxthreads && continue
        for blocks in (264, 528, 1056)
            copyto!(u, u0)
            seed_packed_rng!(workspace, 10000 + threads + blocks)
            elapsed = synchronized_seconds() do
                solve_fixed_sosra!(
                    u, workspace, params;
                    dt=0.002f0, t_end=steps * 0.002f0, steps,
                    threads, blocks, noise_threads=512, noise_blocks=528,
                )
            end
            @printf(
                "SHORT\t%d\t%d\t%.9f\t%.6f\t%s\t%.8g\n",
                threads, blocks, Float64(elapsed), 1.0e6 * elapsed / steps,
                string(all(isfinite, u)), Float64(maximum(abs, u)),
            )
        end
    end
end

function full_benchmark(;
    repetitions=3,
    threads=1024,
    blocks=264,
    noise_threads=512,
    noise_blocks=528,
    rng_kind=:library_philox,
)
    u0 = make_u0(1024)
    u = similar(u0)
    workspace = allocate_workspace(u0; rng_kind)
    params = ModelBParameters()

    # Warm all kernels and the cuRAND path before measurements.
    copyto!(u, u0)
    fixed_sosra_step!(
        u, workspace, params;
        threads, blocks, noise_threads, noise_blocks,
    )
    CUDA.synchronize()
    # Exercise one complete trajectory before formal timing.  Besides reaching
    # steady clocks, this avoids a Julia 1.12/CUDA.jl fresh-process anomaly in
    # which the first host timer surrounding this long GPU loop can be returned
    # as an invalid subnormal even though the GPU result is valid.
    copyto!(u, u0)
    seed_packed_rng!(workspace, 313000)
    solve_fixed_sosra!(
        u, workspace, params;
        dt=0.002f0, t_end=2.0f0,
        threads, blocks, noise_threads, noise_blocks,
    )
    CUDA.synchronize()

    timings = Float64[]
    for repetition in 1:repetitions
        function timed_repetition!()
            copyto!(u, u0)
            seed_packed_rng!(workspace, 314159 + repetition)
            return synchronized_seconds() do
                solve_fixed_sosra!(
                    u, workspace, params;
                    dt=0.002f0, t_end=2.0f0,
                    threads, blocks, noise_threads, noise_blocks,
                )
            end
        end
        elapsed = timed_repetition!()
        # Retain a defensive retry for any invalid timing result. CUDA-event
        # timing avoids the corrupt host loop-local timer observed with Julia
        # 1.12 in this workload.
        if !(elapsed > 0.1)
            elapsed = timed_repetition!()
        end
        finite = all(isfinite, u)
        maxabs = maximum(abs, u)
        max_mass = maximum(abs.(sum(u; dims=(1, 2))))
        @printf(
            "FULL\trepetition=%d\tseconds=%.9f\tstep_us=%.6f\tfinite=%s\tmaxabs=%.8g\tmax_mass=%.8g\tthreads=%d\tblocks=%d\tnoise_threads=%d\tnoise_blocks=%d\trng_kind=%s\n",
            repetition, elapsed, 1.0e3 * elapsed, string(finite),
            Float64(maxabs), Float64(max_mass), threads, blocks,
            noise_threads, noise_blocks, string(rng_kind),
        )
        push!(timings, elapsed)
    end
    println(
        "FULL\tmedian=", median(timings),
        "\tminimum=", minimum(timings),
        "\tmaximum=", maximum(timings),
    )
end

function main()
    mode = isempty(ARGS) ? "short" : ARGS[1]
    if mode == "full"
        repetitions = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
        threads = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1024
        blocks = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 264
        noise_threads = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 512
        noise_blocks = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 528
        rng_kind = length(ARGS) >= 7 ? Symbol(ARGS[7]) : :library_philox
        full_benchmark(;
            repetitions, threads, blocks, noise_threads, noise_blocks, rng_kind,
        )
        return
    end

    u0 = make_u0(1024)
    u = copy(u0)
    workspace = allocate_workspace(u)
    params = ModelBParameters()
    limits = print_resource_summary(u, workspace, params)
    if mode == "resources"
        return
    elseif mode == "kernels"
        kernel_breakdown(u, workspace, params, limits)
    elseif mode == "short"
        short_sweep(u0, workspace, params, limits)
    else
        error("mode must be resources, kernels, short, or full")
    end
end

main()
