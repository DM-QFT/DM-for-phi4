include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "kernels_variants.jl"))

using .H100KernelVariants
using Printf

function launch_production_capped!(
    du, u, two_kappa, m02, c3, inv_gamma, threads, blocks, maxregs
)
    if isnothing(maxregs)
        @cuda threads=threads blocks=blocks modelb_drift_kernel!(
            du, u, two_kappa, m02, c3, inv_gamma
        )
    else
        @cuda threads=threads blocks=blocks maxregs=maxregs modelb_drift_kernel!(
            du, u, two_kappa, m02, c3, inv_gamma
        )
    end
    return nothing
end

function launch_direct_capped!(
    du, u, two_kappa, m02, c3, inv_gamma, threads, blocks, maxregs
)
    n = Int32(length(u))
    if isnothing(maxregs)
        @cuda threads=threads blocks=blocks H100KernelVariants.modelb_drift_linear128_kernel!(
            du, u, two_kappa, m02, c3, inv_gamma, n
        )
    else
        @cuda threads=threads blocks=blocks maxregs=maxregs H100KernelVariants.modelb_drift_linear128_kernel!(
            du, u, two_kappa, m02, c3, inv_gamma, n
        )
    end
    return nothing
end

function benchmark_family(name, launch!, du, reference, u, coefficients, configurations)
    two_kappa, m02, c3, inv_gamma = coefficients
    for (threads, blocks, maxregs) in configurations
        try
            fill!(du, 0.0f0)
            launch!(du, u, two_kappa, m02, c3, inv_gamma, threads, blocks, maxregs)
            CUDA.synchronize()
            # A second launch also avoids treating the production kernel's
            # observed cold-first anomaly as a launch-configuration result.
            launch!(du, u, two_kappa, m02, c3, inv_gamma, threads, blocks, maxregs)
            CUDA.synchronize()
            error = maximum(abs.(du .- reference))
            median_s, min_s, max_s = kernel_median_seconds(; warmup=2, samples=15) do
                launch!(du, u, two_kappa, m02, c3, inv_gamma, threads, blocks, maxregs)
            end
            @printf(
                "family=%s threads=%d blocks=%d maxregs=%s median_ms=%.6f min_ms=%.6f max_ms=%.6f max_error=%.9g\n",
                name,
                threads,
                blocks,
                isnothing(maxregs) ? "none" : string(maxregs),
                1e3 * median_s,
                1e3 * min_s,
                1e3 * max_s,
                error,
            )
        catch exception
            @printf(
                "family=%s threads=%d blocks=%d maxregs=%s ERROR=%s\n",
                name,
                threads,
                blocks,
                isnothing(maxregs) ? "none" : string(maxregs),
                sprint(showerror, exception),
            )
        end
    end
end

function main()
    u = make_u0()
    du = similar(u)
    reference = similar(u)
    coefficients = (0.541f0, -0.344f0, 0.088f0, 1.0f0)
    launch_drift_linear128!(reference, u, coefficients...; threads=1024, blocks=528)
    CUDA.synchronize()

    production_configs = Tuple{Int, Int, Union{Nothing, Int}}[]
    for maxregs in (nothing, 64, 56, 48, 40, 32)
        for (threads, blocks) in ((512, 264), (640, 264), (768, 264), (896, 132), (896, 264), (1024, 264))
            push!(production_configs, (threads, blocks, maxregs))
        end
    end
    benchmark_family(
        "production", launch_production_capped!, du, reference, u,
        coefficients, production_configs,
    )

    direct_configs = Tuple{Int, Int, Union{Nothing, Int}}[]
    for maxregs in (nothing, 40, 32, 28, 24)
        for (threads, blocks) in ((512, 528), (768, 528), (1024, 264), (1024, 528))
            push!(direct_configs, (threads, blocks, maxregs))
        end
    end
    benchmark_family(
        "direct128", launch_direct_capped!, du, reference, u,
        coefficients, direct_configs,
    )
end

main()
