using CUDA, Printf, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

function euler_kernel!(
    v, u, links, sqrt_dt, dt, two_kappa, m02, c3, inv_gamma,
    noise_amplitude, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    mask, shift, _ = modelb_fixed_lattice(lattice)

    @inbounds for linear in id:stride:n
        x0, y0, base, row = modelb_fixed_geometry(linear, lattice)
        xm1 = (x0 - Int32(1)) & mask
        ym1 = (y0 - Int32(1)) & mask
        x_previous = row + xm1 + Int32(1)
        y_previous = base + (ym1 << shift) + x0 + Int32(1)
        div_xi =
            links[linear] - links[x_previous] +
            links[n + linear] - links[n + y_previous]
        drift = modelb_fixed_drift(
            u, linear, two_kappa, m02, c3, inv_gamma, lattice,
        )
        v[linear] =
            u[linear] + dt * drift +
            sqrt_dt * noise_amplitude * div_xi
    end
    return nothing
end

function euler_steps!(
    u, v, links, rng, steps, dt, two_kappa, m02, c3, inv_gamma,
    noise_amplitude, lattice,
)
    n = Int32(length(u))
    threads = 768
    blocks = min(cld(length(u), threads), 264)
    for _ in 1:steps
        Random.randn!(rng, links)
        @cuda threads=threads blocks=blocks euler_kernel!(
            v, u, links, sqrt(dt), dt, two_kappa, m02, c3, inv_gamma,
            noise_amplitude, lattice, n,
        )
        u, v = v, u
    end
    return u, v
end

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
    seed = UInt64(20260720)

    CUDA.seed!(20260719)
    u0 = CUDA.randn(Float32, L, L, Nensemble)

    u_sosra = copy(u0)
    sosra_buffers = modelb_fixed_buffers(u_sosra; seed)
    modelb_fixed_step!(
        u_sosra, sosra_buffers, dt, two_kappa, m02, c3, inv_gamma,
        noise_amplitude,
    )

    u_euler = copy(u0)
    v_euler = similar(u0)
    links_euler = similar(u0, Float32, (size(u0)..., 2))
    euler_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    Random.seed!(euler_rng, seed)
    u_euler, v_euler = euler_steps!(
        u_euler, v_euler, links_euler, euler_rng, 1, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude, Val(L),
    )
    CUDA.synchronize()

    sosra_times = Float64[]
    euler_times = Float64[]
    for repetition in 1:repetitions
        copyto!(u_sosra, u0)
        seed_modelb_fixed!(sosra_buffers, seed + UInt64(repetition))
        CUDA.synchronize()
        push!(
            sosra_times,
            gpu_seconds(
                () -> modelB_2d_fixed!(
                    u_sosra, sosra_buffers;
                    γ=1.0f0, T=1.0f0, κ=0.28f0, λ=0.022f0,
                    tspan=(0.0f0, steps * dt), dt,
                ),
            ),
        )

        copyto!(u_euler, u0)
        Random.seed!(euler_rng, seed + UInt64(repetition))
        CUDA.synchronize()
        euler_seconds = gpu_seconds(
            () -> begin
                u_euler, v_euler = euler_steps!(
                    u_euler, v_euler, links_euler, euler_rng, steps, dt,
                    two_kappa, m02, c3, inv_gamma, noise_amplitude, Val(L),
                )
            end,
        )
        push!(euler_times, euler_seconds)
    end

    sosra_seconds = median(sosra_times)
    euler_seconds = median(euler_times)
    sosra_drift = maximum(
        abs, mean(u_sosra; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    euler_drift = maximum(
        abs, mean(u_euler; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )

    @printf("elements\t%d\n", length(u0))
    @printf("steps\t%d\n", steps)
    @printf("sosra_ms_per_step\t%.6f\n", 1.0e3 * sosra_seconds / steps)
    @printf("euler_ms_per_step\t%.6f\n", 1.0e3 * euler_seconds / steps)
    @printf("sosra_over_euler\t%.3f\n", sosra_seconds / euler_seconds)
    @printf("sosra_max_abs_mean_drift\t%.8e\n", sosra_drift)
    @printf("euler_max_abs_mean_drift\t%.8e\n", euler_drift)
end

main()
