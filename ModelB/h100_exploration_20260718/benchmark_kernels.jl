include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "kernels_variants.jl"))
using .H100KernelVariants

function print_time(kind, name, threads, blocks, timing)
    median_s, min_s, max_s = timing
    println(join((kind, name, threads, blocks,
                  1.0e3 * median_s, 1.0e3 * min_s, 1.0e3 * max_s), '\t'))
end

function main()
    errors = validate_against_modelb!(launch_modelb_drift!, launch_conserved_noise!)
    println("VALIDATION\t", errors)

    u = make_u0()
    out = similar(u)
    xi_x = CUDA.randn(Float32, size(u))
    xi_y = CUDA.randn(Float32, size(u))
    noise_work = similar(u)
    two_kappa, m02, c3, inv_gamma = 0.541f0, -0.252f0, 0.088f0, 1.0f0
    scale = sqrt(DT_DEFAULT)

    linear_kernel = @cuda launch=false H100KernelVariants.modelb_drift_linear128_kernel!(
        out, u, two_kappa, m02, c3, inv_gamma, Int32(length(u))
    )
    noise_kernel = @cuda launch=false H100KernelVariants.conserved_noise_scaled128_kernel!(
        out, xi_x, xi_y, scale, Int32(length(u))
    )
    println("RESOURCE\tdrift_linear\tregisters=", CUDA.registers(linear_kernel),
            "\tmaxthreads=", CUDA.maxthreads(linear_kernel),
            "\tmemory=", CUDA.memory(linear_kernel),
            "\tlaunch=", CUDA.launch_configuration(linear_kernel.fun))
    println("RESOURCE\tnoise_scaled\tregisters=", CUDA.registers(noise_kernel),
            "\tmaxthreads=", CUDA.maxthreads(noise_kernel),
            "\tmemory=", CUDA.memory(noise_kernel),
            "\tlaunch=", CUDA.launch_configuration(noise_kernel.fun))

    println("kind\tname\tthreads\tblocks\tmedian_ms\tmin_ms\tmax_ms")
    print_time("drift", "production", 896, 132,
        kernel_median_seconds(
            () -> launch_modelb_drift!(out, u, two_kappa, m02, c3, inv_gamma);
            samples=12,
        ))

    maxthreads = CUDA.maxthreads(linear_kernel)
    for threads in (128, 256, 384, 512, 640, 768, 896, 1024)
        threads > maxthreads && continue
        for blocks in (132, 264, 396, 528, 1056)
            timing = kernel_median_seconds(
                () -> launch_drift_linear128!(
                    out, u, two_kappa, m02, c3, inv_gamma;
                    threads, blocks,
                );
                warmup=2, samples=8,
            )
            print_time("drift", "linear128", threads, blocks, timing)
        end
    end

    print_time("drift", "shared32x8", "32x8", "4x16x1024",
        kernel_median_seconds(
            () -> launch_drift_shared128_32x8!(
                out, u, two_kappa, m02, c3, inv_gamma
            ); samples=12,
        ))
    print_time("drift", "shared16x16", "16x16", "8x8x1024",
        kernel_median_seconds(
            () -> launch_drift_shared128_16x16!(
                out, u, two_kappa, m02, c3, inv_gamma
            ); samples=12,
        ))

    print_time("noise", "production_divergence", 640, 264,
        kernel_median_seconds(
            () -> launch_conserved_noise!(out, xi_x, xi_y); samples=12,
        ))
    print_time("noise", "production_divergence_plus_scale", 640, 264,
        kernel_median_seconds(() -> begin
            launch_conserved_noise!(noise_work, xi_x, xi_y)
            @. out = scale * noise_work
        end; samples=12))

    noise_maxthreads = CUDA.maxthreads(noise_kernel)
    for threads in (128, 256, 384, 512, 640, 768, 896, 1024)
        threads > noise_maxthreads && continue
        for blocks in (132, 264, 396, 528, 1056)
            timing = kernel_median_seconds(
                () -> launch_noise_scaled128!(
                    out, xi_x, xi_y, scale; threads, blocks
                );
                warmup=2, samples=8,
            )
            print_time("noise", "linear128_fused_scale", threads, blocks, timing)
        end
    end
end

main()
