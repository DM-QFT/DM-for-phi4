module FusedFixedSOSRA128

using CUDA
using Random

export ModelBParameters,
       FixedSOSRAWorkspace,
       allocate_workspace,
       seed_packed_rng!,
       fill_packed_normals!,
       launch_packed_increments!,
       launch_stage1!,
       launch_stage2!,
       launch_stage3!,
       fixed_sosra_step!,
       solve_fixed_sosra!

# This is deliberately a narrow experiment: L=128, Float32, three-dimensional
# state (x,y,ensemble), fixed step, constant additive diffusion.  It does not
# extend or override any SciML method and does not alter ../ModelB.jl.

const L128_MASK_I32 = Int32(127)
const L128_SHIFT = 7
const L128_PLANE_SHIFT = 14

const SOSRA_ALPHA1 = 0.2889875f0
const SOSRA_ALPHA2 = 0.68598807f0
const SOSRA_ALPHA3 = 0.025024459f0
const SOSRA_A21 = 0.6923962f0
const SOSRA_A31 = -3.1609142f0
const SOSRA_A32 = 4.1609144f0
const SOSRA_B21 = 1.3371633f0
const SOSRA_B31_PLUS_B32 = 1.442371f0 + 1.8632741f0
const INV_SQRT3 = inv(sqrt(3.0f0))

struct ModelBParameters
    two_kappa::Float32
    m02::Float32
    c3::Float32
    inv_gamma::Float32
    sigma::Float32
end

function ModelBParameters(;
    gamma=1.0f0,
    temperature=1.0f0,
    kappa=0.2705f0,
    lambda=0.022f0,
)
    gamma_f = Float32(gamma)
    temperature_f = Float32(temperature)
    kappa_f = Float32(kappa)
    lambda_f = Float32(lambda)
    inv_gamma = inv(gamma_f)
    return ModelBParameters(
        2.0f0 * kappa_f,
        2.0f0 * (1.0f0 - 2.0f0 * lambda_f) - 8.0f0 * kappa_f,
        4.0f0 * lambda_f,
        inv_gamma,
        sqrt(2.0f0 * temperature_f * inv_gamma),
    )
end

struct FixedSOSRAWorkspace{A3,A4,R}
    h01::A3
    h02::A3
    k1::A3
    k2::A3
    dW::A3
    dZ::A3
    links::A4
    rng::R
end

function _check_state(u)
    @assert u isa CuArray
    @assert eltype(u) === Float32
    @assert ndims(u) == 3
    @assert size(u, 1) == 128 && size(u, 2) == 128
    @assert length(u) <= typemax(Int32)
    return nothing
end

function _make_packed_rng(kind::Symbol, seed::Integer)
    if kind === :library_philox || kind === :library_philox_dynamic
        rng = CUDA.cuRAND.LibraryRNG(
            CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10;
            stream=CUDA.stream(),
        )
        if kind === :library_philox_dynamic
            CUDA.cuRAND.curandSetGeneratorOrdering(
                rng, CUDA.cuRAND.CURAND_ORDERING_PSEUDO_DYNAMIC
            )
        end
        Random.seed!(rng, seed)
        return rng
    elseif kind === :library_default
        rng = CUDA.cuRAND.LibraryRNG(
            CUDA.cuRAND.CURAND_RNG_PSEUDO_DEFAULT;
            stream=CUDA.stream(),
        )
        Random.seed!(rng, seed)
        return rng
    elseif kind === :implicit_default
        CUDA.seed!(seed)
        return nothing
    end
    throw(ArgumentError(
        "rng_kind must be :library_philox, :library_philox_dynamic, " *
        ":library_default, or :implicit_default"
    ))
end

function allocate_workspace(u; rng_kind=:library_philox, seed=20260718)
    _check_state(u)
    links = similar(u, Float32, (size(u)..., 4))
    rng = _make_packed_rng(rng_kind, seed)
    return FixedSOSRAWorkspace(
        similar(u), similar(u), similar(u), similar(u),
        similar(u), similar(u), links, rng,
    )
end

# One cuRAND call fills W.x, W.y, Z.x, Z.y as four contiguous state-sized slabs.
# This intentionally changes the random-number consumption order relative to
# four separate production fills while preserving the required distribution.
function seed_packed_rng!(workspace::FixedSOSRAWorkspace, seed::Integer)
    if isnothing(workspace.rng)
        CUDA.seed!(seed)
    else
        Random.seed!(workspace.rng, seed)
    end
    return workspace
end

function fill_packed_normals!(workspace::FixedSOSRAWorkspace)
    if isnothing(workspace.rng)
        Random.randn!(workspace.links)
    else
        Random.randn!(workspace.rng, workspace.links)
    end
    return workspace.links
end

@inline function _linear_geometry(linear)
    linear0 = linear - Int32(1)
    x0 = linear0 & L128_MASK_I32
    y0 = (linear0 >> L128_SHIFT) & L128_MASK_I32
    ensemble0 = linear0 >> L128_PLANE_SHIFT
    row = (ensemble0 << L128_PLANE_SHIFT) + (y0 << L128_SHIFT)
    row_m1 = (ensemble0 << L128_PLANE_SHIFT) +
             (((y0 - Int32(1)) & L128_MASK_I32) << L128_SHIFT)
    xm1 = (x0 - Int32(1)) & L128_MASK_I32
    return x0, y0, row, row_m1, xm1
end

function packed_conserved_increments_kernel!(dW, dZ, links, sqrt_h, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        x0, _, row, row_m1, xm1 = _linear_geometry(linear)
        xprev = row + xm1 + Int32(1)
        yprev = row_m1 + x0 + Int32(1)

        wx = links[linear]
        wy = links[n + linear]
        zx = links[(n << 1) + linear]
        zy = links[3 * n + linear]
        dW[linear] = sqrt_h * (
            wx - links[xprev] + wy - links[n + yprev]
        )
        dZ[linear] = sqrt_h * (
            zx - links[(n << 1) + xprev] +
            zy - links[3 * n + yprev]
        )
    end
    return nothing
end

function launch_packed_increments!(
    workspace::FixedSOSRAWorkspace,
    dt;
    threads=512,
    blocks=min(cld(length(workspace.dW), threads), 528),
)
    n = Int32(length(workspace.dW))
    @cuda threads=threads blocks=blocks packed_conserved_increments_kernel!(
        workspace.dW, workspace.dZ, workspace.links, sqrt(Float32(dt)), n
    )
    return nothing
end

@inline _cube(x) = x * x * x

@inline function _drift_at128(u, linear, two_kappa, m02, c3, inv_gamma)
    linear0 = linear - Int32(1)
    x0 = linear0 & L128_MASK_I32
    y0 = (linear0 >> L128_SHIFT) & L128_MASK_I32
    ensemble0 = linear0 >> L128_PLANE_SHIFT

    xm1 = (x0 - Int32(1)) & L128_MASK_I32
    xp1 = (x0 + Int32(1)) & L128_MASK_I32
    xm2 = (x0 - Int32(2)) & L128_MASK_I32
    xp2 = (x0 + Int32(2)) & L128_MASK_I32
    ym1 = (y0 - Int32(1)) & L128_MASK_I32
    yp1 = (y0 + Int32(1)) & L128_MASK_I32
    ym2 = (y0 - Int32(2)) & L128_MASK_I32
    yp2 = (y0 + Int32(2)) & L128_MASK_I32

    base = ensemble0 << L128_PLANE_SHIFT
    row = base + (y0 << L128_SHIFT)
    row_m1 = base + (ym1 << L128_SHIFT)
    row_p1 = base + (yp1 << L128_SHIFT)
    row_m2 = base + (ym2 << L128_SHIFT)
    row_p2 = base + (yp2 << L128_SHIFT)

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
    minus_biharmonic =
        -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
    laplacian_phi = nearest_sum - 4 * center
    laplacian_phi3 =
        _cube(left) + _cube(right) + _cube(down) + _cube(up) - 4 * _cube(center)
    return inv_gamma * (
        two_kappa * minus_biharmonic + m02 * laplacian_phi + c3 * laplacian_phi3
    )
end

@inline function _q_at(dW, dZ, linear, sigma)
    return 0.5f0 * sigma * (@inbounds(dW[linear]) + INV_SQRT3 * @inbounds(dZ[linear]))
end

function sosra_stage1_kernel!(h01, k1, u, dW, dZ, h, params, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        drift = _drift_at128(
            u, linear, params.two_kappa, params.m02, params.c3, params.inv_gamma
        )
        k1[linear] = drift
        q = _q_at(dW, dZ, linear, params.sigma)
        h01[linear] = u[linear] + h * SOSRA_A21 * drift + SOSRA_B21 * q
    end
    return nothing
end

function sosra_stage2_kernel!(h02, k2, u, h01, k1, dW, dZ, h, params, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        drift = _drift_at128(
            h01, linear, params.two_kappa, params.m02, params.c3, params.inv_gamma
        )
        k2[linear] = drift
        q = _q_at(dW, dZ, linear, params.sigma)
        h02[linear] = u[linear] + h * (
            SOSRA_A31 * k1[linear] + SOSRA_A32 * drift
        ) + SOSRA_B31_PLUS_B32 * q
    end
    return nothing
end

function sosra_stage3_kernel!(u, h02, k1, k2, dW, h, params, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        k3 = _drift_at128(
            h02, linear, params.two_kappa, params.m02, params.c3, params.inv_gamma
        )
        u[linear] = u[linear] + h * (
            SOSRA_ALPHA1 * k1[linear] +
            SOSRA_ALPHA2 * k2[linear] +
            SOSRA_ALPHA3 * k3
        ) + params.sigma * dW[linear]
    end
    return nothing
end

function launch_stage1!(
    u, workspace::FixedSOSRAWorkspace, h, params::ModelBParameters;
    threads=1024,
    blocks=min(cld(length(u), threads), 264),
)
    @cuda threads=threads blocks=blocks sosra_stage1_kernel!(
        workspace.h01, workspace.k1, u, workspace.dW, workspace.dZ,
        Float32(h), params, Int32(length(u)),
    )
    return nothing
end

function launch_stage2!(
    u, workspace::FixedSOSRAWorkspace, h, params::ModelBParameters;
    threads=1024,
    blocks=min(cld(length(u), threads), 264),
)
    @cuda threads=threads blocks=blocks sosra_stage2_kernel!(
        workspace.h02, workspace.k2, u, workspace.h01, workspace.k1,
        workspace.dW, workspace.dZ, Float32(h), params, Int32(length(u)),
    )
    return nothing
end

function launch_stage3!(
    u, workspace::FixedSOSRAWorkspace, h, params::ModelBParameters;
    threads=1024,
    blocks=min(cld(length(u), threads), 264),
)
    @cuda threads=threads blocks=blocks sosra_stage3_kernel!(
        u, workspace.h02, workspace.k1, workspace.k2, workspace.dW,
        Float32(h), params, Int32(length(u)),
    )
    return nothing
end

function fixed_sosra_step!(
    u,
    workspace::FixedSOSRAWorkspace,
    params::ModelBParameters=ModelBParameters();
    dt=0.002f0,
    threads=1024,
    blocks=min(cld(length(u), threads), 264),
    noise_threads=512,
    noise_blocks=min(cld(length(u), noise_threads), 528),
)
    fill_packed_normals!(workspace)
    launch_packed_increments!(
        workspace, dt; threads=noise_threads, blocks=noise_blocks
    )
    launch_stage1!(u, workspace, dt, params; threads, blocks)
    launch_stage2!(u, workspace, dt, params; threads, blocks)
    launch_stage3!(u, workspace, dt, params; threads, blocks)
    return u
end

function solve_fixed_sosra!(
    u,
    workspace::FixedSOSRAWorkspace=allocate_workspace(u),
    params::ModelBParameters=ModelBParameters();
    dt=0.002f0,
    t_end=2.0f0,
    steps=nothing,
    kwargs...,
)
    _check_state(u)
    dt_f = Float32(dt)
    nsteps = isnothing(steps) ? round(Int, Float32(t_end) / dt_f) : Int(steps)
    @assert nsteps >= 0
    @assert isapprox(nsteps * dt_f, Float32(t_end); atol=eps(Float32(t_end))) ||
            !isnothing(steps)
    for _ in 1:nsteps
        fixed_sosra_step!(u, workspace, params; dt=dt_f, kwargs...)
    end
    return u
end

end # module FusedFixedSOSRA128
