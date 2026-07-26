using CUDA

include(joinpath(@__DIR__, "..", "..", "solver", "ModelB_fixed.jl"))

const GPUARRAYS_RNG = CUDA.CUDACore.GPUArrays

@inline function link_normals4(linear, step, seed)
    words = GPUARRAYS_RNG.philox4x32_10(
        UInt64(linear), step, seed,
    )
    return GPUARRAYS_RNG.philox_to_normals(Float32, words...)
end

@inline function x_link_normals(linear, step, seed)
    a1, a2, _, _ = GPUARRAYS_RNG.philox4x32_10(
        UInt64(linear), step, seed,
    )
    return GPUARRAYS_RNG.boxmuller(Float32, a1, a2)
end

@inline function y_link_normals(linear, step, seed)
    _, _, a3, a4 = GPUARRAYS_RNG.philox4x32_10(
        UInt64(linear), step, seed,
    )
    return GPUARRAYS_RNG.boxmuller(Float32, a3, a4)
end

@inline function fused_link_divergences(
    linear, x_previous, y_previous, step, seed,
)
    wx, zx, wy, zy = link_normals4(linear, step, seed)
    wx_previous, zx_previous =
        x_link_normals(x_previous, step, seed)
    wy_previous, zy_previous =
        y_link_normals(y_previous, step, seed)
    return (
        wx - wx_previous + wy - wy_previous,
        zx - zx_previous + zy - zy_previous,
    )
end

function fused_philox_buffers(u; seed)
    check_modelb_fixed_state(u)
    return (
        h01=similar(u),
        h02=similar(u),
        k1=similar(u),
        q=similar(u),
        seed=UInt64[seed],
        step=UInt64[0],
        lattice=Val(size(u, 1)),
    )
end

function seed_fused_philox!(buffers, seed)
    buffers.seed[1] = UInt64(seed)
    buffers.step[1] = UInt64(0)
    return nothing
end

function gpuarrays_buffers(u; seed)
    check_modelb_fixed_state(u)
    return (
        h01=similar(u),
        h02=similar(u),
        k1=similar(u),
        q=similar(u),
        links=similar(u, Float32, (size(u)..., 4)),
        rng=CUDA.RNG(seed),
        lattice=Val(size(u, 1)),
    )
end

Base.@noinline function seed_gpuarrays!(buffers, seed)
    Random.seed!(buffers.rng, seed)
    @assert buffers.rng.counter == 0
    return nothing
end

function padded_curand_buffers(u; seed)
    check_modelb_fixed_state(u)
    rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    Random.seed!(rng, seed)
    return (
        h01=similar(u),
        h02=similar(u),
        k1=similar(u),
        q=similar(u),
        links=similar(u, Float32, nextpow(2, 4 * length(u))),
        rng=rng,
        lattice=Val(size(u, 1)),
    )
end

function direct_curand_fill!(rng, links)
    @assert iseven(length(links))
    CUDA.cuRAND.update_stream(rng)
    CUDA.cuRAND.curandGenerateNormal(
        rng, links, length(links), 0.0f0, 1.0f0,
    )
    return links
end

function direct_curand_step!(
    u, buffers, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    direct_curand_fill!(buffers.rng, buffers.links)
    lattice = buffers.lattice
    n = Int32(length(u))
    stage1_threads = 768
    stage1_blocks = min(cld(length(u), stage1_threads), 264)
    stage_threads = 1024
    stage_blocks = min(cld(length(u), stage_threads), 264)

    @cuda threads=stage1_threads blocks=stage1_blocks modelb_fixed_stage1_kernel!(
        buffers.h01, buffers.h02, buffers.k1, buffers.q,
        u, buffers.links, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
        lattice, n,
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage2_kernel!(
        u, buffers.h02, buffers.h01, buffers.k1, buffers.q, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, n,
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage3_kernel!(
        u, buffers.h02, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, n,
    )
    return nothing
end

function fused_philox_stage1_kernel!(
    h01, h02, k1, q, u, sqrt_dt, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
    seed, step, lattice, n,
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
        div_w, div_z = fused_link_divergences(
            linear, x_previous, y_previous, step, seed,
        )
        dW = sqrt_dt * div_w
        dZ = sqrt_dt * div_z
        noise = 0.5f0 * noise_amplitude * (
            dW + SOSRA_INV_SQRT3 * dZ
        )
        drift = modelb_fixed_drift(
            u, linear, two_kappa, m02, c3, inv_gamma, lattice,
        )
        old_u = u[linear]
        q[linear] = noise
        k1[linear] = drift
        h01[linear] =
            old_u + dt * SOSRA_A21 * drift + SOSRA_B21 * noise
        h02[linear] =
            old_u + dt * SOSRA_ALPHA1 * drift + noise_amplitude * dW
    end
    return nothing
end

function tiled_philox_stage1_kernel!(
    h01, h02, k1, q, u, sqrt_dt, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
    seed, step, lattice, ::Val{TY},
) where TY
    tx = threadIdx().x
    ty = threadIdx().y
    local_index = (ty - Int32(1)) * Int32(32) + tx
    tile_sites = Int32(32 * TY)
    mask, shift, plane_shift = modelb_fixed_lattice(lattice)
    x0 = (blockIdx().x - Int32(1)) * Int32(32) + tx - Int32(1)
    y0 = (blockIdx().y - Int32(1)) * Int32(TY) + ty - Int32(1)
    base = (blockIdx().z - Int32(1)) << plane_shift
    row = base + (y0 << shift)
    linear = row + x0 + Int32(1)
    shared = CuStaticSharedArray(
        Float32,
        4 * 32 * TY + 2 * TY + 64,
    )

    wx, zx, wy, zy = link_normals4(linear, step, seed)
    @inbounds begin
        shared[local_index] = wx
        shared[tile_sites + local_index] = zx
        shared[2 * tile_sites + local_index] = wy
        shared[3 * tile_sites + local_index] = zy
    end

    if local_index <= TY
        halo_y =
            (blockIdx().y - Int32(1)) * Int32(TY) +
            local_index - Int32(1)
        halo_x = (
            (blockIdx().x - Int32(1)) * Int32(32) - Int32(1)
        ) & mask
        halo_linear =
            base + (halo_y << shift) + halo_x + Int32(1)
        halo_wx, halo_zx =
            x_link_normals(halo_linear, step, seed)
        @inbounds begin
            shared[4 * tile_sites + local_index] = halo_wx
            shared[4 * tile_sites + Int32(TY) + local_index] =
                halo_zx
        end
    end
    if local_index <= 32
        halo_x =
            (blockIdx().x - Int32(1)) * Int32(32) +
            local_index - Int32(1)
        halo_y = (
            (blockIdx().y - Int32(1)) * Int32(TY) - Int32(1)
        ) & mask
        halo_linear =
            base + (halo_y << shift) + halo_x + Int32(1)
        halo_wy, halo_zy =
            y_link_normals(halo_linear, step, seed)
        @inbounds begin
            shared[
                4 * tile_sites + Int32(2 * TY) + local_index
            ] = halo_wy
            shared[
                4 * tile_sites + Int32(2 * TY + 32) + local_index
            ] = halo_zy
        end
    end
    sync_threads()

    @inbounds begin
        current_wx = shared[local_index]
        current_zx = shared[tile_sites + local_index]
        current_wy = shared[2 * tile_sites + local_index]
        current_zy = shared[3 * tile_sites + local_index]
        previous_wx =
            tx > 1 ?
            shared[local_index - Int32(1)] :
            shared[4 * tile_sites + ty]
        previous_zx =
            tx > 1 ?
            shared[tile_sites + local_index - Int32(1)] :
            shared[4 * tile_sites + Int32(TY) + ty]
        previous_wy =
            ty > 1 ?
            shared[2 * tile_sites + local_index - Int32(32)] :
            shared[
                4 * tile_sites + Int32(2 * TY) + tx
            ]
        previous_zy =
            ty > 1 ?
            shared[3 * tile_sites + local_index - Int32(32)] :
            shared[
                4 * tile_sites + Int32(2 * TY + 32) + tx
            ]

        dW = sqrt_dt * (
            current_wx - previous_wx + current_wy - previous_wy
        )
        dZ = sqrt_dt * (
            current_zx - previous_zx + current_zy - previous_zy
        )
        noise = 0.5f0 * noise_amplitude * (
            dW + SOSRA_INV_SQRT3 * dZ
        )
        drift = modelb_fixed_drift(
            u, linear, two_kappa, m02, c3, inv_gamma, lattice,
        )
        old_u = u[linear]
        q[linear] = noise
        k1[linear] = drift
        h01[linear] =
            old_u + dt * SOSRA_A21 * drift + SOSRA_B21 * noise
        h02[linear] =
            old_u + dt * SOSRA_ALPHA1 * drift + noise_amplitude * dW
    end
    return nothing
end

function fused_philox_step!(
    u, buffers, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    lattice = buffers.lattice
    n = Int32(length(u))
    stage1_threads = 768
    stage1_blocks = min(cld(length(u), stage1_threads), 132)
    stage_threads = 1024
    stage_blocks = min(cld(length(u), stage_threads), 264)
    seed = buffers.seed[1]
    step = buffers.step[1]

    @cuda threads=stage1_threads blocks=stage1_blocks fused_philox_stage1_kernel!(
        buffers.h01, buffers.h02, buffers.k1, buffers.q,
        u, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
        seed, step, lattice, n,
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage2_kernel!(
        u, buffers.h02, buffers.h01, buffers.k1, buffers.q, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, n,
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage3_kernel!(
        u, buffers.h02, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, n,
    )
    buffers.step[1] = step + UInt64(1)
    return nothing
end

function tiled_philox_step!(
    u, buffers, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    L = size(u, 1)
    @assert L % 32 == 0
    tile_y = 8
    @assert L % tile_y == 0
    lattice = buffers.lattice
    stage_threads = 1024
    stage_blocks = min(cld(length(u), stage_threads), 264)
    seed = buffers.seed[1]
    step = buffers.step[1]

    @cuda threads=(32, tile_y) blocks=(L ÷ 32, L ÷ tile_y, size(u, 3)) tiled_philox_stage1_kernel!(
        buffers.h01, buffers.h02, buffers.k1, buffers.q,
        u, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, noise_amplitude,
        seed, step, lattice, Val(tile_y),
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage2_kernel!(
        u, buffers.h02, buffers.h01, buffers.k1, buffers.q, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, Int32(length(u)),
    )
    @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage3_kernel!(
        u, buffers.h02, dt,
        two_kappa, m02, c3, inv_gamma,
        lattice, Int32(length(u)),
    )
    buffers.step[1] = step + UInt64(1)
    return nothing
end

function fused_philox_steps!(
    u, buffers, steps, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    for _ in 1:steps
        fused_philox_step!(
            u, buffers, dt,
            two_kappa, m02, c3, inv_gamma, noise_amplitude,
        )
    end
    return u
end

function tiled_philox_steps!(
    u, buffers, steps, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    for _ in 1:steps
        tiled_philox_step!(
            u, buffers, dt,
            two_kappa, m02, c3, inv_gamma, noise_amplitude,
        )
    end
    return u
end

function fill_link_normals_kernel!(samples, seed, step, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        wx, zx, wy, zy = link_normals4(linear, step, seed)
        samples[1, linear] = wx
        samples[2, linear] = zx
        samples[3, linear] = wy
        samples[4, linear] = zy
    end
    return nothing
end

function fill_divergences_kernel!(
    divergence, seed, step, lattice, nplane, n,
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
        div_w, div_z = fused_link_divergences(
            linear, x_previous, y_previous, step, seed,
        )
        divergence[linear] = div_w
        divergence[n + linear] = div_z
    end
    return nothing
end

function overlapped_curand_buffers(u; seed)
    check_modelb_fixed_state(u)
    rng_stream = CuStream()
    rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10;
        stream=rng_stream,
    )
    CUDA.stream!(rng_stream) do
        Random.seed!(rng, seed)
    end
    CUDA.synchronize(rng_stream)
    return (
        h01=similar(u),
        h02=similar(u),
        k1=similar(u),
        q=similar(u),
        links=(
            similar(u, Float32, (size(u)..., 4)),
            similar(u, Float32, (size(u)..., 4)),
        ),
        rng=rng,
        rng_stream=rng_stream,
        ready=(
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
        ),
        consumed=(
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
        ),
        lattice=Val(size(u, 1)),
    )
end

function seed_overlapped_curand!(buffers, seed)
    CUDA.synchronize()
    CUDA.stream!(buffers.rng_stream) do
        Random.seed!(buffers.rng, seed)
    end
    CUDA.synchronize(buffers.rng_stream)
    return nothing
end

function overlapped_curand_steps!(
    u, buffers, steps, dt,
    two_kappa, m02, c3, inv_gamma, noise_amplitude,
)
    @assert steps > 0
    n = Int32(length(u))
    stage1_threads = 768
    stage1_blocks = min(cld(length(u), stage1_threads), 264)
    stage_threads = 1024
    stage_blocks = min(cld(length(u), stage_threads), 264)
    compute_stream = CUDA.stream()

    CUDA.stream!(buffers.rng_stream) do
        Random.randn!(buffers.rng, buffers.links[1])
        CUDA.record(buffers.ready[1])
    end

    current = 1
    for step in 1:steps
        CUDA.wait(buffers.ready[current], compute_stream)
        @cuda threads=stage1_threads blocks=stage1_blocks modelb_fixed_stage1_kernel!(
            buffers.h01, buffers.h02, buffers.k1, buffers.q,
            u, buffers.links[current], sqrt(dt), dt,
            two_kappa, m02, c3, inv_gamma, noise_amplitude,
            buffers.lattice, n,
        )
        CUDA.record(buffers.consumed[current], compute_stream)

        next = 3 - current
        if step < steps
            CUDA.stream!(buffers.rng_stream) do
                if step > 1
                    CUDA.wait(buffers.consumed[next])
                end
                Random.randn!(buffers.rng, buffers.links[next])
                CUDA.record(buffers.ready[next])
            end
        end

        @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage2_kernel!(
            u, buffers.h02, buffers.h01, buffers.k1, buffers.q, dt,
            two_kappa, m02, c3, inv_gamma,
            buffers.lattice, n,
        )
        @cuda threads=stage_threads blocks=stage_blocks modelb_fixed_stage3_kernel!(
            u, buffers.h02, dt,
            two_kappa, m02, c3, inv_gamma,
            buffers.lattice, n,
        )
        current = next
    end
    return u
end
