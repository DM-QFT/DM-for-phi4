using CUDA, Random

include(joinpath(@__DIR__, "..", "..", "solver", "ModelB_fixed.jl"))

function realspace_parameters(; γ=1.0f0, T=1.0f0, κ=0.28f0, λ=0.022f0)
    γ, T, κ, λ = Float32.((γ, T, κ, λ))
    @assert γ > 0.0f0 && T >= 0.0f0
    return (
        γ=γ,
        T=T,
        κ=κ,
        λ=λ,
        two_kappa=2.0f0 * κ,
        m02=2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ,
        c3=4.0f0 * λ,
        inv_gamma=1.0f0 / γ,
        noise_amplitude=sqrt(2.0f0 * T / γ),
    )
end

function philox_rng(seed)
    rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    Random.seed!(rng, seed)
    return rng
end

function euler_buffers(u; seed)
    check_modelb_fixed_state(u)
    return (
        links=similar(u, Float32, (size(u)..., 2)),
        rng=philox_rng(seed),
        lattice=Val(size(u, 1)),
    )
end

function heun_buffers(u; seed)
    check_modelb_fixed_state(u)
    return (
        v=similar(u),
        k1=similar(u),
        q=similar(u),
        links=similar(u, Float32, (size(u)..., 2)),
        rng=philox_rng(seed),
        lattice=Val(size(u, 1)),
    )
end

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
        dW = sqrt_dt * (
            links[linear] - links[x_previous] +
            links[n + linear] - links[n + y_previous]
        )
        drift = modelb_fixed_drift(
            u, linear, two_kappa, m02, c3, inv_gamma, lattice,
        )
        v[linear] =
            u[linear] + dt * drift + noise_amplitude * dW
    end
    return nothing
end

function heun_stage1_kernel!(
    v, k1, q, u, links, sqrt_dt, dt, two_kappa, m02, c3,
    inv_gamma, noise_amplitude, lattice, n,
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
        dW = sqrt_dt * (
            links[linear] - links[x_previous] +
            links[n + linear] - links[n + y_previous]
        )
        drift = modelb_fixed_drift(
            u, linear, two_kappa, m02, c3, inv_gamma, lattice,
        )
        noise = noise_amplitude * dW
        k1[linear] = drift
        q[linear] = noise
        v[linear] = u[linear] + dt * drift + noise
    end
    return nothing
end

function heun_stage2_kernel!(
    u, v, k1, q, dt, two_kappa, m02, c3, inv_gamma, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for linear in id:stride:n
        k2 = modelb_fixed_drift(
            v, linear, two_kappa, m02, c3, inv_gamma, lattice,
        )
        u[linear] =
            u[linear] + 0.5f0 * dt * (k1[linear] + k2) + q[linear]
    end
    return nothing
end

function euler_steps!(u, v, buffers, steps, dt, p)
    dt = Float32(dt)
    n = Int32(length(u))
    threads = 768
    blocks = min(cld(length(u), threads), 264)
    sqrt_dt = sqrt(dt)
    for _ in 1:steps
        Random.randn!(buffers.rng, buffers.links)
        @cuda threads=threads blocks=blocks euler_kernel!(
            v, u, buffers.links, sqrt_dt, dt,
            p.two_kappa, p.m02, p.c3, p.inv_gamma,
            p.noise_amplitude, buffers.lattice, n,
        )
        u, v = v, u
    end
    return u, v
end

function heun_steps!(u, buffers, steps, dt, p)
    dt = Float32(dt)
    n = Int32(length(u))
    stage1_threads = 768
    stage1_blocks = min(cld(length(u), stage1_threads), 264)
    stage2_threads = 1024
    stage2_blocks = min(cld(length(u), stage2_threads), 264)
    sqrt_dt = sqrt(dt)
    for _ in 1:steps
        Random.randn!(buffers.rng, buffers.links)
        @cuda threads=stage1_threads blocks=stage1_blocks heun_stage1_kernel!(
            buffers.v, buffers.k1, buffers.q, u, buffers.links,
            sqrt_dt, dt, p.two_kappa, p.m02, p.c3, p.inv_gamma,
            p.noise_amplitude, buffers.lattice, n,
        )
        @cuda threads=stage2_threads blocks=stage2_blocks heun_stage2_kernel!(
            u, buffers.v, buffers.k1, buffers.q, dt,
            p.two_kappa, p.m02, p.c3, p.inv_gamma,
            buffers.lattice, n,
        )
    end
    return u
end

function sosra_steps!(u, buffers, steps, dt, p)
    dt = Float32(dt)
    for _ in 1:steps
        modelb_fixed_step!(
            u, buffers, dt,
            p.two_kappa, p.m02, p.c3, p.inv_gamma, p.noise_amplitude,
        )
    end
    return u
end

function realspace_observables_kernel!(
    obs, u, κ, m02, λ, lattice, nplane,
)
    tid = threadIdx().x
    ensemble = blockIdx().x
    threads = blockDim().x
    mask, shift, plane_shift = modelb_fixed_lattice(lattice)
    base = (ensemble - Int32(1)) << plane_shift
    shared = CuDynamicSharedArray(Float64, (threads, Int32(5)))

    s1 = 0.0
    s2 = 0.0
    s4 = 0.0
    snn = 0.0
    sS = 0.0
    @inbounds for local_index in tid:threads:nplane
        local0 = local_index - Int32(1)
        x0 = local0 & mask
        y0 = (local0 >> shift) & mask
        xp1 = (x0 + Int32(1)) & mask
        yp1 = (y0 + Int32(1)) & mask
        row = base + (y0 << shift)
        right_index = row + xp1 + Int32(1)
        up_index = base + (yp1 << shift) + x0 + Int32(1)

        ϕ = Float64(u[base + local_index])
        right = Float64(u[right_index])
        up = Float64(u[up_index])
        ϕ2 = ϕ * ϕ
        ϕ4 = ϕ2 * ϕ2
        s1 += ϕ
        s2 += ϕ2
        s4 += ϕ4
        snn += 0.5 * ϕ * (right + up)
        # S = -2κ Σ_<xy> ϕxϕy + (1-2λ)Σ_x ϕx² + λΣ_x ϕx⁴.
        sS += Float64(κ) * (
            (ϕ - right) * (ϕ - right) + (ϕ - up) * (ϕ - up)
        ) + 0.5 * Float64(m02) * ϕ2 + Float64(λ) * ϕ4
    end

    @inbounds begin
        shared[tid, 1] = s1
        shared[tid, 2] = s2
        shared[tid, 3] = s4
        shared[tid, 4] = snn
        shared[tid, 5] = sS
    end
    sync_threads()

    offset = threads >> 1
    while offset > 0
        if tid <= offset
            @inbounds for j in 1:5
                shared[tid, j] += shared[tid + offset, j]
            end
        end
        sync_threads()
        offset >>= 1
    end

    if tid == 1
        scale = 1.0 / Float64(nplane)
        @inbounds begin
            obs[1, ensemble] = shared[1, 1] * scale
            obs[2, ensemble] = Float64(u[base + Int32(1)])
            obs[3, ensemble] = shared[1, 2] * scale
            obs[4, ensemble] = shared[1, 3] * scale
            obs[5, ensemble] = shared[1, 4] * scale
            obs[6, ensemble] = shared[1, 5] * scale
        end
    end
    return nothing
end

function realspace_observables!(obs, u, p)
    L = size(u, 1)
    @assert size(obs) == (6, size(u, 3))
    threads = min(256, L * L)
    @assert ispow2(threads)
    @cuda threads=threads blocks=size(u, 3) shmem=5 * threads * sizeof(Float64) realspace_observables_kernel!(
        obs, u, p.κ, p.m02, p.λ, Val(L), Int32(L * L),
    )
    return obs
end
