using CUDA, Random

# Fixed SOSRA specialization for power-of-two, Float32 Model-B lattices.
# It evolves all ensembles in the third array dimension and returns the final
# CuArray directly; it does not construct a SciML solution object or save
# intermediate states.

# SOSRA coefficients from StochasticDiffEq, specialized to additive noise.
const SOSRA_ALPHA1 = 0.2889875f0
const SOSRA_ALPHA2 = 0.68598807f0
const SOSRA_ALPHA3 = 0.025024459f0
const SOSRA_A21 = 0.6923962f0
const SOSRA_A31 = -3.1609142f0
const SOSRA_A32 = 4.1609144f0
const SOSRA_B21 = 1.3371633f0
const SOSRA_B31_B32 = 1.442371f0 + 1.8632741f0
const SOSRA_INV_SQRT3 = inv(sqrt(3.0f0))

@inline fixed_cube(x) = x * x * x

function check_modelb_fixed_state(u)
    @assert u isa CuArray "u must be a CuArray"
    @assert eltype(u) == Float32 "u must use Float32"
    lattice_size = size(u, 1)
    valid_shape =
        ndims(u) == 3 && size(u, 2) == lattice_size &&
        lattice_size >= 4 && ispow2(lattice_size)
    @assert valid_shape "u must have shape (L, L, Nensemble), with power-of-two L"
    @assert size(u, 3) > 0 "Nensemble must be positive"
    @assert length(u) <= typemax(Int32) ÷ 4 "too many ensembles"
    return nothing
end

function modelb_fixed_buffers(u; seed=rand(UInt64))
    check_modelb_fixed_state(u)
    lattice = Val(size(u, 1))
    # Keep the random state with the buffers so other CUDA random calls do not
    # change this solver's trajectory.
    rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    Random.seed!(rng, seed)
    return (
        h01=similar(u),
        h02=similar(u),
        k1=similar(u),
        k2=similar(u),
        dW=similar(u),
        q=similar(u),
        links=similar(u, Float32, (size(u)..., 4)),
        rng=rng,
        lattice=lattice,
    )
end

function seed_modelb_fixed!(buffers, seed)
    Random.seed!(buffers.rng, seed)
    return nothing
end

function check_modelb_fixed_buffers(u, buffers)
    state_arrays = (
        buffers.h01, buffers.h02, buffers.k1, buffers.k2,
        buffers.dW, buffers.q,
    )
    matching_state = all(size(array) == size(u) for array in state_arrays)
    @assert matching_state "buffers do not match u"
    @assert size(buffers.links) == (size(u)..., 4) "buffers do not match u"
    @assert buffers.lattice == Val(size(u, 1)) "buffers do not match u"
    return nothing
end

function modelb_fixed_step_count(tspan, dt)
    dt = Float32(dt)
    dt > 0.0f0 || throw(ArgumentError("dt must be positive"))
    duration = Float32(tspan[2] - tspan[1])
    duration >= 0.0f0 || throw(ArgumentError("tspan must run forward"))
    steps = round(Int, duration / dt)
    tolerance = 8.0f0 * eps(Float32) * max(duration, 1.0f0)
    abs(steps * dt - duration) <= tolerance || throw(ArgumentError(
        "tspan must contain an integer number of fixed dt steps",
    ))
    return steps, dt
end

@inline function modelb_fixed_lattice(::Val{L}) where L
    # L is part of the buffer type, so these become constants in each kernel.
    shift = Int32(trailing_zeros(L))
    return Int32(L - 1), shift, shift + shift
end

@inline function modelb_fixed_geometry(linear, lattice)
    mask, shift, plane_shift = modelb_fixed_lattice(lattice)
    linear0 = linear - Int32(1)
    x0 = linear0 & mask
    y0 = (linear0 >> shift) & mask
    ensemble0 = linear0 >> plane_shift
    base = ensemble0 << plane_shift
    row = base + (y0 << shift)
    return x0, y0, base, row
end

@inline function modelb_fixed_drift(
    u, linear, two_kappa, m02, c3, inv_gamma,
    lattice,
)
    # This returns one Float32 register value; it does not allocate an array.
    mask, shift, _ = modelb_fixed_lattice(lattice)
    x0, y0, base, row = modelb_fixed_geometry(linear, lattice)

    xm1 = (x0 - Int32(1)) & mask
    xp1 = (x0 + Int32(1)) & mask
    xm2 = (x0 - Int32(2)) & mask
    xp2 = (x0 + Int32(2)) & mask
    ym1 = (y0 - Int32(1)) & mask
    yp1 = (y0 + Int32(1)) & mask
    ym2 = (y0 - Int32(2)) & mask
    yp2 = (y0 + Int32(2)) & mask

    row_m1 = base + (ym1 << shift)
    row_p1 = base + (yp1 << shift)
    row_m2 = base + (ym2 << shift)
    row_p2 = base + (yp2 << shift)

    center = @inbounds u[row + x0 + Int32(1)]
    left = @inbounds u[row + xm1 + Int32(1)]
    right = @inbounds u[row + xp1 + Int32(1)]
    down = @inbounds u[row_m1 + x0 + Int32(1)]
    up = @inbounds u[row_p1 + x0 + Int32(1)]

    nearest_sum = left + right + down + up
    diagonal_sum = @inbounds (
        u[row_m1 + xm1 + Int32(1)] + u[row_p1 + xm1 + Int32(1)] +
        u[row_m1 + xp1 + Int32(1)] + u[row_p1 + xp1 + Int32(1)]
    )
    second_neighbor_sum = @inbounds (
        u[row + xm2 + Int32(1)] + u[row + xp2 + Int32(1)] +
        u[row_m2 + x0 + Int32(1)] + u[row_p2 + x0 + Int32(1)]
    )

    # drift = Delta(dS/dphi)/gamma.  The two Laplacians are fused here so no
    # intermediate dS/dphi array is needed.
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

function modelb_fixed_noise_stage1_kernel!(
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
        drift = modelb_fixed_drift(
            u, linear, two_kappa, m02, c3, inv_gamma,
            lattice,
        )

        dW[linear] = dw
        # SOSRA stage noise: sigma * I_(1,0) / dt.
        q[linear] = noise
        k1[linear] = drift
        h01[linear] =
            u[linear] + dt * SOSRA_A21 * drift + SOSRA_B21 * noise
    end
    return nothing
end

function modelb_fixed_stage2_kernel!(
    h02, k2, u, h01, k1, q, dt,
    two_kappa, m02, c3, inv_gamma,
    lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for linear in id:stride:n
        drift = modelb_fixed_drift(
            h01, linear, two_kappa, m02, c3, inv_gamma,
            lattice,
        )
        noise = q[linear]
        k2[linear] = drift
        h02[linear] = u[linear] + dt * (
            SOSRA_A31 * k1[linear] + SOSRA_A32 * drift
        ) + SOSRA_B31_B32 * noise
    end
    return nothing
end

function modelb_fixed_stage3_kernel!(
    u, h02, k1, k2, dW, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
    lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for linear in id:stride:n
        k3 = modelb_fixed_drift(
            h02, linear, two_kappa, m02, c3, inv_gamma,
            lattice,
        )
        u[linear] = u[linear] + dt * (
            SOSRA_ALPHA1 * k1[linear] +
            SOSRA_ALPHA2 * k2[linear] +
            SOSRA_ALPHA3 * k3
        ) + noise_amplitude * dW[linear]
    end
    return nothing
end

function modelb_fixed_step!(
    u, buffers, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    Random.randn!(buffers.rng, buffers.links)
    lattice = buffers.lattice
    n = Int32(length(u))
    # Tuned on the 132-SM GH200 for L=128 with 1024 ensembles.
    stage1_threads = 768
    stage1_blocks = min(cld(length(u), stage1_threads), 264)
    stage_threads = 1024
    stage_blocks = min(cld(length(u), stage_threads), 264)

    @cuda threads=stage1_threads blocks=stage1_blocks modelb_fixed_noise_stage1_kernel!(
        buffers.h01, buffers.k1, buffers.dW, buffers.q,
        u, buffers.links, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
        lattice, n,
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage2_kernel!(
        buffers.h02, buffers.k2, u, buffers.h01, buffers.k1,
        buffers.q, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, n,
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage3_kernel!(
        u, buffers.h02, buffers.k1, buffers.k2, buffers.dW, dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
        lattice, n,
    )
    return nothing
end

function modelB_2d_fixed!(
    u,
    buffers;
    γ=1.0f0,
    tspan=(0.0f0, 15.0f0),
    T=1.0f0,
    κ=0.2705f0,
    λ=0.022f0,
    dt=0.002f0,
)
    # This in-place path stays asynchronous. Synchronize before timing,
    # reading the result on the CPU, changing stream, or reseeding buffers.
    check_modelb_fixed_state(u)
    check_modelb_fixed_buffers(u, buffers)
    γ, T, κ, λ, dt = Float32.((γ, T, κ, λ, dt))
    tspan = Float32.(tspan)

    @assert γ > 0.0f0
    @assert T >= 0.0f0
    steps, dt = modelb_fixed_step_count(tspan, dt)

    inv_gamma = 1.0f0 / γ
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    noise_amplitude = sqrt(2.0f0 * T * inv_gamma)

    for _ in 1:steps
        modelb_fixed_step!(
            u, buffers, dt,
            two_kappa, m02, c3, inv_gamma, noise_amplitude,
        )
    end
    return u
end

function modelB_2d_fixed(
    u0;
    γ=1.0f0,
    tspan=(0.0f0, 15.0f0),
    T=1.0f0,
    κ=0.2705f0,
    λ=0.022f0,
    dt=0.002f0,
    seed=rand(UInt64),
)
    # Pass seed explicitly when the exact random trajectory must be reproducible.
    u = CuArray{Float32}(u0)
    buffers = modelb_fixed_buffers(u; seed)
    modelB_2d_fixed!(u, buffers; γ, tspan, T, κ, λ, dt)
    CUDA.synchronize()
    return u
end
