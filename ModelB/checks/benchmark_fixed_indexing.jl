using CUDA, Printf, Random, Statistics

include(joinpath(@__DIR__, "..", "ModelB_fixed.jl"))

@inline function cartesian_limitbound2(a, n)
    if a == n + 1
        return 1
    elseif a == 0
        return n
    elseif a == n + 2
        return 2
    elseif a == -1
        return n - 1
    end
    return a
end

@inline function modelb_cartesian_drift(
    u, linear, two_kappa, m02, c3, inv_gamma,
)
    n = size(u, 1)
    x, y, ensemble = Tuple(CartesianIndices(u)[linear])

    xp1 = cartesian_limitbound2(x + 1, n)
    xm1 = cartesian_limitbound2(x - 1, n)
    yp1 = cartesian_limitbound2(y + 1, n)
    ym1 = cartesian_limitbound2(y - 1, n)
    xp2 = cartesian_limitbound2(x + 2, n)
    xm2 = cartesian_limitbound2(x - 2, n)
    yp2 = cartesian_limitbound2(y + 2, n)
    ym2 = cartesian_limitbound2(y - 2, n)

    center = @inbounds u[x, y, ensemble]
    left = @inbounds u[xm1, y, ensemble]
    right = @inbounds u[xp1, y, ensemble]
    down = @inbounds u[x, ym1, ensemble]
    up = @inbounds u[x, yp1, ensemble]

    nearest_sum = left + right + down + up
    diagonal_sum = @inbounds (
        u[xm1, ym1, ensemble] + u[xm1, yp1, ensemble] +
        u[xp1, ym1, ensemble] + u[xp1, yp1, ensemble]
    )
    second_neighbor_sum = @inbounds (
        u[xm2, y, ensemble] + u[xp2, y, ensemble] +
        u[x, ym2, ensemble] + u[x, yp2, ensemble]
    )

    minus_biharmonic =
        -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
    laplacian_phi = nearest_sum - 4 * center
    laplacian_phi3 =
        fixed_cube(left) + fixed_cube(right) + fixed_cube(down) +
        fixed_cube(up) - 4 * fixed_cube(center)

    return inv_gamma * (
        two_kappa * minus_biharmonic +
        m02 * laplacian_phi +
        c3 * laplacian_phi3
    )
end

function cartesian_noise_stage1_kernel!(
    h01, k1, dW, q, u, links, sqrt_dt, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
    lattice, n,
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

        wx = links[linear]
        wy = links[n + linear]
        zx = links[(n << 1) + linear]
        zy = links[3 * n + linear]
        dw = sqrt_dt * (
            wx - links[x_previous] + wy - links[n + y_previous]
        )
        dz = sqrt_dt * (
            zx - links[(n << 1) + x_previous] +
            zy - links[3 * n + y_previous]
        )
        noise = 0.5f0 * noise_amplitude * (
            dw + SOSRA_INV_SQRT3 * dz
        )
        drift = modelb_cartesian_drift(u, linear, two_kappa, m02, c3, inv_gamma)

        dW[linear] = dw
        q[linear] = noise
        k1[linear] = drift
        h01[linear] =
            u[linear] + dt * SOSRA_A21 * drift + SOSRA_B21 * noise
    end
    return nothing
end

function cartesian_stage2_kernel!(
    h02, k2, u, h01, k1, q, dt,
    two_kappa, m02, c3, inv_gamma, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        drift = modelb_cartesian_drift(h01, linear, two_kappa, m02, c3, inv_gamma)
        noise = q[linear]
        k2[linear] = drift
        h02[linear] = u[linear] + dt * (
            SOSRA_A31 * k1[linear] + SOSRA_A32 * drift
        ) + SOSRA_B31_B32 * noise
    end
    return nothing
end

function cartesian_stage3_kernel!(
    u, h02, k1, k2, dW, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        k3 = modelb_cartesian_drift(h02, linear, two_kappa, m02, c3, inv_gamma)
        u[linear] = u[linear] + dt * (
            SOSRA_ALPHA1 * k1[linear] +
            SOSRA_ALPHA2 * k2[linear] +
            SOSRA_ALPHA3 * k3
        ) + noise_amplitude * dW[linear]
    end
    return nothing
end

function cartesian_step!(
    u, buffers, dt, parameters, threads, blocks,
)
    two_kappa, m02, c3, inv_gamma, noise_amplitude = parameters
    Random.randn!(buffers.rng, buffers.links)
    n = Int32(length(u))
    @cuda threads=threads blocks=blocks cartesian_noise_stage1_kernel!(
        buffers.h01, buffers.k1, buffers.dW, buffers.q,
        u, buffers.links, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
        buffers.lattice, n,
    )
    @cuda threads=threads blocks=blocks cartesian_stage2_kernel!(
        buffers.h02, buffers.k2, u, buffers.h01, buffers.k1,
        buffers.q, dt,
        two_kappa, m02, c3, inv_gamma, n,
    )
    @cuda threads=threads blocks=blocks cartesian_stage3_kernel!(
        u, buffers.h02, buffers.k1, buffers.k2, buffers.dW, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude, n,
    )
    return nothing
end

function cartesian_solve!(u, buffers, steps, threads, blocks)
    check_modelb_fixed_state(u)
    κ = 0.2705f0
    λ = 0.022f0
    parameters = (
        2.0f0 * κ,
        2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ,
        4.0f0 * λ,
        1.0f0,
        sqrt(2.0f0),
    )
    for _ in 1:steps
        cartesian_step!(
            u, buffers, 0.002f0, parameters, threads, blocks,
        )
    end
    return u
end

function elapsed_seconds(f)
    CUDA.synchronize()
    start = time_ns()
    value = f()
    CUDA.synchronize()
    return (time_ns() - start) / 1.0e9, value
end

function main()
    CUDA.seed!(20260719)
    u0 = CUDA.randn(Float32, 128, 128, 1024)
    u0 .-= mean(u0; dims=(1, 2))
    u = copy(u0)
    buffers = modelb_fixed_buffers(u; seed=UInt64(1))
    κ = 0.2705f0
    λ = 0.022f0
    parameters = (
        2.0f0 * κ,
        2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ,
        4.0f0 * λ,
        1.0f0,
        sqrt(2.0f0),
    )
    drift_parameters = parameters[1:4]
    n = Int32(length(u))

    k1 = @cuda launch=false cartesian_noise_stage1_kernel!(
        buffers.h01, buffers.k1, buffers.dW, buffers.q,
        u, buffers.links, sqrt(0.002f0), 0.002f0,
        parameters..., buffers.lattice, n,
    )
    k2 = @cuda launch=false cartesian_stage2_kernel!(
        buffers.h02, buffers.k2, u, buffers.h01, buffers.k1,
        buffers.q, 0.002f0, drift_parameters..., n,
    )
    k3 = @cuda launch=false cartesian_stage3_kernel!(
        u, buffers.h02, buffers.k1, buffers.k2, buffers.dW, 0.002f0,
        parameters..., n,
    )
    maxthreads = minimum(CUDA.maxthreads.((k1, k2, k3)))
    println(
        "resources cartesian registers=",
        CUDA.registers.((k1, k2, k3)),
        " maxthreads=", maxthreads,
    )

    best = (Inf, 0, 0)
    for threads in (256, 512, 768, 896, 1024)
        threads > maxthreads && continue
        for blocks in (132, 264, 528)
            copyto!(u, u0)
            seed_modelb_fixed!(buffers, UInt64(1000 + threads + blocks))
            seconds, _ = elapsed_seconds(
                () -> cartesian_solve!(u, buffers, 20, threads, blocks),
            )
            println(
                "tune threads=", threads, " blocks=", blocks,
                " step_ms=", 1000 * seconds / 20,
            )
            seconds < best[1] && (best = (seconds, threads, blocks))
        end
    end
    threads, blocks = best[2], best[3]
    println("selected threads=", threads, " blocks=", blocks)

    # Full-path warmup and exact same-seed comparison.
    copyto!(u, u0)
    seed_modelb_fixed!(buffers, UInt64(314159))
    cartesian_solve!(u, buffers, 1000, threads, blocks)
    CUDA.synchronize()
    u_cartesian = copy(u)

    copyto!(u, u0)
    seed_modelb_fixed!(buffers, UInt64(314159))
    modelB_2d_fixed!(u, buffers; tspan=(0.0f0, 2.0f0), dt=0.002f0)
    CUDA.synchronize()
    println("warmup max_abs_difference=", maximum(abs, u .- u_cartesian))

    direct_times = Float64[]
    cartesian_times = Float64[]
    ratios = Float64[]
    println("repetition\tdirect_s\tcartesian_s\tratio\tmax_error")
    for repetition in 1:8
        seed = UInt64(400000 + repetition)
        run_direct = function ()
            copyto!(u, u0)
            seed_modelb_fixed!(buffers, seed)
            modelB_2d_fixed!(u, buffers; tspan=(0.0f0, 2.0f0), dt=0.002f0)
            return copy(u)
        end
        run_cartesian = function ()
            copyto!(u, u0)
            seed_modelb_fixed!(buffers, seed)
            cartesian_solve!(u, buffers, 1000, threads, blocks)
            return copy(u)
        end

        if isodd(repetition)
            direct_time, direct = elapsed_seconds(run_direct)
            cartesian_time, cartesian = elapsed_seconds(run_cartesian)
        else
            cartesian_time, cartesian = elapsed_seconds(run_cartesian)
            direct_time, direct = elapsed_seconds(run_direct)
        end
        error = maximum(abs, direct .- cartesian)
        ratio = cartesian_time / direct_time
        push!(direct_times, direct_time)
        push!(cartesian_times, cartesian_time)
        push!(ratios, ratio)
        @printf(
            "%d\t%.9f\t%.9f\t%.6f\t%.8g\n",
            repetition, direct_time, cartesian_time, ratio, error,
        )
    end

    @printf(
        "median\t%.9f\t%.9f\t%.6f\n",
        median(direct_times), median(cartesian_times), median(ratios),
    )
end

main()
