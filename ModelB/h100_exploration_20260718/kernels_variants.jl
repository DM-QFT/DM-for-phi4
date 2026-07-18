module H100KernelVariants

using CUDA
using CUDA: i32
using DiffEqNoiseProcess
using Random
using Test

export launch_drift_linear128!,
       launch_drift_shared128_32x8!,
       launch_drift_shared128_16x16!,
       launch_noise_linear128!,
       launch_noise_scaled128!,
       launch_noise_bridge128!,
       modelb_noise_dist_fused128!,
       modelb_noise_bridge_fused128!,
       modelb_noise_process_fused128,
       drift_reference_cpu!,
       noise_reference_cpu!,
       validate_against_modelb!

const L128 = 128
const L128_MASK = 127
const L128_SHIFT = 7
const L128_PLANE_SHIFT = 14
const L128_MASK_I32 = 127i32

@inline cube128(x) = x * x * x

@inline function linear_index128(ensemble0, x0, y0)
    return (ensemble0 << L128_PLANE_SHIFT) +
           (y0 << L128_SHIFT) + x0 + one(x0)
end

function check_l128_arrays(first, rest...)
    @assert ndims(first) == 3
    @assert size(first, 1) == L128 && size(first, 2) == L128
    @assert length(first) <= typemax(Int32)
    @assert eltype(first) === Float32
    for a in rest
        @assert size(a) == size(first)
        @assert eltype(a) === Float32
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Variant 1: L=128-specialized, direct linear addressing.
#
# Julia arrays are column-major, so x is the contiguous coordinate.  For L=128,
# the x/y coordinates and periodic wrap can be obtained with shifts and masks;
# there is no CartesianIndices conversion, integer division, or general modulo.
# -----------------------------------------------------------------------------

function modelb_drift_linear128_kernel!(du, u, two_kappa, m02, c3, inv_gamma, n)
    id = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for linear in id:stride:n
        linear0 = linear - 1i32
        x0 = linear0 & L128_MASK_I32
        y0 = (linear0 >> L128_SHIFT) & L128_MASK_I32
        ensemble0 = linear0 >> L128_PLANE_SHIFT

        xm1 = (x0 - 1i32) & L128_MASK_I32
        xp1 = (x0 + 1i32) & L128_MASK_I32
        xm2 = (x0 - 2i32) & L128_MASK_I32
        xp2 = (x0 + 2i32) & L128_MASK_I32
        ym1 = (y0 - 1i32) & L128_MASK_I32
        yp1 = (y0 + 1i32) & L128_MASK_I32
        ym2 = (y0 - 2i32) & L128_MASK_I32
        yp2 = (y0 + 2i32) & L128_MASK_I32

        row = (ensemble0 << L128_PLANE_SHIFT) + (y0 << L128_SHIFT)
        row_m1 = (ensemble0 << L128_PLANE_SHIFT) + (ym1 << L128_SHIFT)
        row_p1 = (ensemble0 << L128_PLANE_SHIFT) + (yp1 << L128_SHIFT)
        row_m2 = (ensemble0 << L128_PLANE_SHIFT) + (ym2 << L128_SHIFT)
        row_p2 = (ensemble0 << L128_PLANE_SHIFT) + (yp2 << L128_SHIFT)

        center = u[row + x0 + 1i32]
        left = u[row + xm1 + 1i32]
        right = u[row + xp1 + 1i32]
        down = u[row_m1 + x0 + 1i32]
        up = u[row_p1 + x0 + 1i32]

        nearest_sum = left + right + down + up
        diagonal_sum =
            u[row_m1 + xm1 + 1i32] + u[row_p1 + xm1 + 1i32] +
            u[row_m1 + xp1 + 1i32] + u[row_p1 + xp1 + 1i32]
        second_neighbor_sum =
            u[row + xm2 + 1i32] + u[row + xp2 + 1i32] +
            u[row_m2 + x0 + 1i32] + u[row_p2 + x0 + 1i32]

        minus_biharmonic =
            -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
        laplacian_phi = nearest_sum - 4 * center
        laplacian_phi3 =
            cube128(left) + cube128(right) + cube128(down) + cube128(up) -
            4 * cube128(center)

        du[linear] = inv_gamma * (
            two_kappa * minus_biharmonic +
            m02 * laplacian_phi +
            c3 * laplacian_phi3
        )
    end
    return nothing
end

function launch_drift_linear128!(
    du, u, two_kappa, m02, c3, inv_gamma;
    threads=896,
    blocks=min(cld(length(u), threads), 132),
)
    check_l128_arrays(du, u)
    @cuda threads=threads blocks=blocks modelb_drift_linear128_kernel!(
        du, u, two_kappa, m02, c3, inv_gamma, Int32(length(u))
    )
    return nothing
end

# -----------------------------------------------------------------------------
# Variant 2: two-dimensional shared-memory tiles with a radius-two halo.
#
# Both shapes use 256 threads.  32x8 gives fully contiguous warps; 16x16 has a
# smaller halo/center ratio.  Explicit halo loads avoid div/rem instructions in
# the cooperative load.  A block belongs to exactly one ensemble.
# -----------------------------------------------------------------------------

function modelb_drift_shared128_32x8_kernel!(du, u, two_kappa, m02, c3, inv_gamma)
    tx = threadIdx().x
    ty = threadIdx().y
    ensemble0 = blockIdx().z - 1i32
    x0 = ((blockIdx().x - 1i32) << 5) + tx - 1i32
    y0 = ((blockIdx().y - 1i32) << 3) + ty - 1i32
    tile = @cuStaticSharedMem(Float32, (36, 12))

    # Tile interior.
    @inbounds tile[tx + 2i32, ty + 2i32] = u[linear_index128(ensemble0, x0, y0)]

    # Left/right halo columns, excluding the four corner patches.
    if tx <= 2i32
        xl = (x0 - 2i32) & L128_MASK_I32
        xr = (x0 + 32i32) & L128_MASK_I32
        @inbounds begin
            tile[tx, ty + 2i32] = u[linear_index128(ensemble0, xl, y0)]
            tile[tx + 34i32, ty + 2i32] = u[linear_index128(ensemble0, xr, y0)]
        end
    end

    # Bottom/top halo rows, excluding the four corner patches.
    if ty <= 2i32
        yd = (y0 - 2i32) & L128_MASK_I32
        yu = (y0 + 8i32) & L128_MASK_I32
        @inbounds begin
            tile[tx + 2i32, ty] = u[linear_index128(ensemble0, x0, yd)]
            tile[tx + 2i32, ty + 10i32] = u[linear_index128(ensemble0, x0, yu)]
        end
    end

    # Four 2x2 halo corners.
    if tx <= 2i32 && ty <= 2i32
        xl = (x0 - 2i32) & L128_MASK_I32
        xr = (x0 + 32i32) & L128_MASK_I32
        yd = (y0 - 2i32) & L128_MASK_I32
        yu = (y0 + 8i32) & L128_MASK_I32
        @inbounds begin
            tile[tx, ty] = u[linear_index128(ensemble0, xl, yd)]
            tile[tx + 34i32, ty] = u[linear_index128(ensemble0, xr, yd)]
            tile[tx, ty + 10i32] = u[linear_index128(ensemble0, xl, yu)]
            tile[tx + 34i32, ty + 10i32] = u[linear_index128(ensemble0, xr, yu)]
        end
    end

    sync_threads()

    sx = tx + 2i32
    sy = ty + 2i32
    @inbounds begin
        center = tile[sx, sy]
        left, right = tile[sx - 1i32, sy], tile[sx + 1i32, sy]
        down, up = tile[sx, sy - 1i32], tile[sx, sy + 1i32]
        nearest_sum = left + right + down + up
        diagonal_sum =
            tile[sx - 1i32, sy - 1i32] + tile[sx - 1i32, sy + 1i32] +
            tile[sx + 1i32, sy - 1i32] + tile[sx + 1i32, sy + 1i32]
        second_neighbor_sum =
            tile[sx - 2i32, sy] + tile[sx + 2i32, sy] +
            tile[sx, sy - 2i32] + tile[sx, sy + 2i32]

        minus_biharmonic =
            -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
        laplacian_phi = nearest_sum - 4 * center
        laplacian_phi3 =
            cube128(left) + cube128(right) + cube128(down) + cube128(up) -
            4 * cube128(center)

        du[linear_index128(ensemble0, x0, y0)] = inv_gamma * (
            two_kappa * minus_biharmonic +
            m02 * laplacian_phi +
            c3 * laplacian_phi3
        )
    end
    return nothing
end

function launch_drift_shared128_32x8!(du, u, two_kappa, m02, c3, inv_gamma)
    check_l128_arrays(du, u)
    @cuda threads=(32, 8) blocks=(4, 16, size(u, 3)) modelb_drift_shared128_32x8_kernel!(
        du, u, two_kappa, m02, c3, inv_gamma
    )
    return nothing
end

function modelb_drift_shared128_16x16_kernel!(du, u, two_kappa, m02, c3, inv_gamma)
    tx = threadIdx().x
    ty = threadIdx().y
    ensemble0 = blockIdx().z - 1i32
    x0 = ((blockIdx().x - 1i32) << 4) + tx - 1i32
    y0 = ((blockIdx().y - 1i32) << 4) + ty - 1i32
    tile = @cuStaticSharedMem(Float32, (20, 20))

    @inbounds tile[tx + 2i32, ty + 2i32] = u[linear_index128(ensemble0, x0, y0)]

    if tx <= 2i32
        xl = (x0 - 2i32) & L128_MASK_I32
        xr = (x0 + 16i32) & L128_MASK_I32
        @inbounds begin
            tile[tx, ty + 2i32] = u[linear_index128(ensemble0, xl, y0)]
            tile[tx + 18i32, ty + 2i32] = u[linear_index128(ensemble0, xr, y0)]
        end
    end
    if ty <= 2i32
        yd = (y0 - 2i32) & L128_MASK_I32
        yu = (y0 + 16i32) & L128_MASK_I32
        @inbounds begin
            tile[tx + 2i32, ty] = u[linear_index128(ensemble0, x0, yd)]
            tile[tx + 2i32, ty + 18i32] = u[linear_index128(ensemble0, x0, yu)]
        end
    end
    if tx <= 2i32 && ty <= 2i32
        xl = (x0 - 2i32) & L128_MASK_I32
        xr = (x0 + 16i32) & L128_MASK_I32
        yd = (y0 - 2i32) & L128_MASK_I32
        yu = (y0 + 16i32) & L128_MASK_I32
        @inbounds begin
            tile[tx, ty] = u[linear_index128(ensemble0, xl, yd)]
            tile[tx + 18i32, ty] = u[linear_index128(ensemble0, xr, yd)]
            tile[tx, ty + 18i32] = u[linear_index128(ensemble0, xl, yu)]
            tile[tx + 18i32, ty + 18i32] = u[linear_index128(ensemble0, xr, yu)]
        end
    end

    sync_threads()

    sx = tx + 2i32
    sy = ty + 2i32
    @inbounds begin
        center = tile[sx, sy]
        left, right = tile[sx - 1i32, sy], tile[sx + 1i32, sy]
        down, up = tile[sx, sy - 1i32], tile[sx, sy + 1i32]
        nearest_sum = left + right + down + up
        diagonal_sum =
            tile[sx - 1i32, sy - 1i32] + tile[sx - 1i32, sy + 1i32] +
            tile[sx + 1i32, sy - 1i32] + tile[sx + 1i32, sy + 1i32]
        second_neighbor_sum =
            tile[sx - 2i32, sy] + tile[sx + 2i32, sy] +
            tile[sx, sy - 2i32] + tile[sx, sy + 2i32]

        minus_biharmonic =
            -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
        laplacian_phi = nearest_sum - 4 * center
        laplacian_phi3 =
            cube128(left) + cube128(right) + cube128(down) + cube128(up) -
            4 * cube128(center)

        du[linear_index128(ensemble0, x0, y0)] = inv_gamma * (
            two_kappa * minus_biharmonic +
            m02 * laplacian_phi +
            c3 * laplacian_phi3
        )
    end
    return nothing
end

function launch_drift_shared128_16x16!(du, u, two_kappa, m02, c3, inv_gamma)
    check_l128_arrays(du, u)
    @cuda threads=(16, 16) blocks=(8, 8, size(u, 3)) modelb_drift_shared128_16x16_kernel!(
        du, u, two_kappa, m02, c3, inv_gamma
    )
    return nothing
end

# -----------------------------------------------------------------------------
# Variant 3: direct-index conserved noise and fusion of the following scale or
# scale-plus-bridge broadcast.  The noise_work array is no longer read/written.
# -----------------------------------------------------------------------------

@inline function conserved_noise_at128(xi_x, xi_y, linear)
    linear0 = linear - 1i32
    x0 = linear0 & L128_MASK_I32
    y0 = (linear0 >> L128_SHIFT) & L128_MASK_I32
    ensemble0 = linear0 >> L128_PLANE_SHIFT
    xm1 = (x0 - 1i32) & L128_MASK_I32
    ym1 = (y0 - 1i32) & L128_MASK_I32
    row = (ensemble0 << L128_PLANE_SHIFT) + (y0 << L128_SHIFT)
    row_m1 = (ensemble0 << L128_PLANE_SHIFT) + (ym1 << L128_SHIFT)
    return @inbounds (
        xi_x[linear] - xi_x[row + xm1 + 1i32] +
        xi_y[linear] - xi_y[row_m1 + x0 + 1i32]
    )
end

function conserved_noise_linear128_kernel!(out, xi_x, xi_y, n)
    id = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        out[linear] = conserved_noise_at128(xi_x, xi_y, linear)
    end
    return nothing
end

function conserved_noise_scaled128_kernel!(out, xi_x, xi_y, scale, n)
    id = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        out[linear] = scale * conserved_noise_at128(xi_x, xi_y, linear)
    end
    return nothing
end

function conserved_noise_bridge128_kernel!(out, xi_x, xi_y, scale, q, Wh, n)
    id = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        out[linear] = scale * conserved_noise_at128(xi_x, xi_y, linear) + q * Wh[linear]
    end
    return nothing
end

function launch_noise_linear128!(
    out, xi_x, xi_y;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_l128_arrays(out, xi_x, xi_y)
    @cuda threads=threads blocks=blocks conserved_noise_linear128_kernel!(
        out, xi_x, xi_y, Int32(length(out))
    )
    return nothing
end

function launch_noise_scaled128!(
    out, xi_x, xi_y, scale;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_l128_arrays(out, xi_x, xi_y)
    @cuda threads=threads blocks=blocks conserved_noise_scaled128_kernel!(
        out, xi_x, xi_y, scale, Int32(length(out))
    )
    return nothing
end

function launch_noise_bridge128!(
    out, xi_x, xi_y, scale, q, Wh;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_l128_arrays(out, xi_x, xi_y, Wh)
    @cuda threads=threads blocks=blocks conserved_noise_bridge128_kernel!(
        out, xi_x, xi_y, scale, q, Wh, Int32(length(out))
    )
    return nothing
end

function modelb_noise_dist_fused128!(rand_vec, W, dt, u, p, t, rng)
    _, xi_x, xi_y = p
    scale = sqrt(abs(dt))
    CUDA.@sync begin
        DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
        DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_y)
        launch_noise_scaled128!(rand_vec, xi_x, xi_y, scale)
    end
    return nothing
end

function modelb_noise_bridge_fused128!(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    _, xi_x, xi_y = p
    scale = sqrt((1 - q) * q * abs(h))
    CUDA.@sync begin
        DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
        DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_y)
        launch_noise_bridge128!(rand_vec, xi_x, xi_y, scale, q, Wh)
    end
    return nothing
end

function modelb_noise_process_fused128(t0, template)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        modelb_noise_dist_fused128!,
        modelb_noise_bridge_fused128!;
        save_everystep=false,
    )
end

# -----------------------------------------------------------------------------
# CPU references and an opt-in GPU equivalence test.
#
# Usage after loading the unchanged production implementation:
#
#   include("../ModelB.jl")
#   include("kernels_variants.jl")
#   using .H100KernelVariants
#   validate_against_modelb!(launch_modelb_drift!, launch_conserved_noise!)
# -----------------------------------------------------------------------------

function drift_reference_cpu!(du, u, two_kappa, m02, c3, inv_gamma)
    check_l128_arrays(du, u)
    @inbounds for ensemble in axes(u, 3), y in 1:L128, x in 1:L128
        xm1 = ((x - 2) & L128_MASK) + 1
        xp1 = (x & L128_MASK) + 1
        xm2 = ((x - 3) & L128_MASK) + 1
        xp2 = ((x + 1) & L128_MASK) + 1
        ym1 = ((y - 2) & L128_MASK) + 1
        yp1 = (y & L128_MASK) + 1
        ym2 = ((y - 3) & L128_MASK) + 1
        yp2 = ((y + 1) & L128_MASK) + 1

        center = u[x, y, ensemble]
        left, right = u[xm1, y, ensemble], u[xp1, y, ensemble]
        down, up = u[x, ym1, ensemble], u[x, yp1, ensemble]
        nearest_sum = left + right + down + up
        diagonal_sum =
            u[xm1, ym1, ensemble] + u[xm1, yp1, ensemble] +
            u[xp1, ym1, ensemble] + u[xp1, yp1, ensemble]
        second_neighbor_sum =
            u[xm2, y, ensemble] + u[xp2, y, ensemble] +
            u[x, ym2, ensemble] + u[x, yp2, ensemble]

        minus_biharmonic =
            -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
        laplacian_phi = nearest_sum - 4 * center
        laplacian_phi3 =
            cube128(left) + cube128(right) + cube128(down) + cube128(up) -
            4 * cube128(center)
        du[x, y, ensemble] = inv_gamma * (
            two_kappa * minus_biharmonic +
            m02 * laplacian_phi +
            c3 * laplacian_phi3
        )
    end
    return nothing
end

function noise_reference_cpu!(out, xi_x, xi_y)
    check_l128_arrays(out, xi_x, xi_y)
    @inbounds for ensemble in axes(out, 3), y in 1:L128, x in 1:L128
        xm1 = ((x - 2) & L128_MASK) + 1
        ym1 = ((y - 2) & L128_MASK) + 1
        out[x, y, ensemble] =
            xi_x[x, y, ensemble] - xi_x[xm1, y, ensemble] +
            xi_y[x, y, ensemble] - xi_y[x, ym1, ensemble]
    end
    return nothing
end

max_abs_error(a, b) = maximum(abs, a .- b)

function validate_against_modelb!(
    original_drift!, original_noise!;
    ensembles=2,
    seed=20260718,
    rtol=2.0f-5,
    atol=2.0f-5,
)
    Random.seed!(seed)
    shape = (L128, L128, ensembles)
    u = randn(Float32, shape)
    xi_x = randn(Float32, shape)
    xi_y = randn(Float32, shape)
    Wh = randn(Float32, shape)
    two_kappa = 0.541f0
    m02 = -0.252f0
    c3 = 0.088f0
    inv_gamma = 1.0f0
    scale = 0.071f0
    q = 0.37f0

    drift_cpu = similar(u)
    noise_cpu = similar(u)
    drift_reference_cpu!(drift_cpu, u, two_kappa, m02, c3, inv_gamma)
    noise_reference_cpu!(noise_cpu, xi_x, xi_y)

    gu = CuArray(u)
    gxi_x, gxi_y, gWh = CuArray(xi_x), CuArray(xi_y), CuArray(Wh)
    d_original = similar(gu)
    d_linear = similar(gu)
    d_shared32x8 = similar(gu)
    d_shared16x16 = similar(gu)
    n_original = similar(gu)
    n_linear = similar(gu)
    n_scaled = similar(gu)
    n_bridge = similar(gu)

    # The production drift sits exactly at its register-limited 896-thread
    # launch bound.  Compile and execute it once before collecting validation
    # output; this also keeps JIT effects out of the comparison.
    original_drift!(d_original, gu, two_kappa, m02, c3, inv_gamma)
    original_noise!(n_original, gxi_x, gxi_y)
    CUDA.synchronize()

    original_drift!(d_original, gu, two_kappa, m02, c3, inv_gamma)
    launch_drift_linear128!(d_linear, gu, two_kappa, m02, c3, inv_gamma)
    launch_drift_shared128_32x8!(d_shared32x8, gu, two_kappa, m02, c3, inv_gamma)
    launch_drift_shared128_16x16!(d_shared16x16, gu, two_kappa, m02, c3, inv_gamma)
    original_noise!(n_original, gxi_x, gxi_y)
    launch_noise_linear128!(n_linear, gxi_x, gxi_y)
    launch_noise_scaled128!(n_scaled, gxi_x, gxi_y, scale)
    launch_noise_bridge128!(n_bridge, gxi_x, gxi_y, scale, q, gWh)
    CUDA.synchronize()

    drift_original = Array(d_original)
    drift_linear = Array(d_linear)
    drift_shared32x8 = Array(d_shared32x8)
    drift_shared16x16 = Array(d_shared16x16)
    noise_original = Array(n_original)
    noise_linear = Array(n_linear)
    noise_scaled = Array(n_scaled)
    noise_bridge = Array(n_bridge)
    scaled_expected = scale .* noise_cpu
    bridge_expected = scale .* noise_cpu .+ q .* Wh

    errors = (
        original_drift_cpu=max_abs_error(drift_original, drift_cpu),
        linear_drift_original=max_abs_error(drift_linear, drift_original),
        shared32x8_drift_original=max_abs_error(drift_shared32x8, drift_original),
        shared16x16_drift_original=max_abs_error(drift_shared16x16, drift_original),
        original_noise_cpu=max_abs_error(noise_original, noise_cpu),
        linear_noise_original=max_abs_error(noise_linear, noise_original),
        fused_scale=max_abs_error(noise_scaled, scaled_expected),
        fused_bridge=max_abs_error(noise_bridge, bridge_expected),
    )

    @testset "L=128 H100 kernel variants" begin
        @test isapprox(drift_original, drift_cpu; rtol=rtol, atol=atol)
        @test isapprox(drift_linear, drift_original; rtol=rtol, atol=atol)
        @test isapprox(drift_shared32x8, drift_original; rtol=rtol, atol=atol)
        @test isapprox(drift_shared16x16, drift_original; rtol=rtol, atol=atol)
        @test isapprox(noise_original, noise_cpu; rtol=rtol, atol=atol)
        @test isapprox(noise_linear, noise_original; rtol=rtol, atol=atol)
        @test isapprox(noise_scaled, scaled_expected; rtol=rtol, atol=atol)
        @test isapprox(noise_bridge, bridge_expected; rtol=rtol, atol=atol)
    end
    return errors
end

end # module H100KernelVariants
