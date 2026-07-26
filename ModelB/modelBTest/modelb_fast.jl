using CUDA, Random

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

# Two experimental rewrites of the fused SOSRA step, both algebraically identical
# to solver/ModelB_fixed.jl.  They exist to test where the 0.30 ms/step goes.
#
#  variant A: same cuRAND link fields, but the stage state is regrouped so that
#             stage 1 writes 3 arrays instead of 4 and stage 2 reads 3 instead of 5.
#  variant B: A plus counter-based link noise generated in registers, so the four
#             L*L*Ne link fields never touch DRAM.
#
# Regrouping: with k1 = f(u), k2 = f(h01), k3 = f(h02) and q the SOSRA stage
# noise, the original updates are linear in k1, so u's contribution can be folded
# into the accumulators at stage 1:
#   h01 = u + dt*a21*k1 + b21*q
#   acc = u + dt*α1*k1  + amp*dW        -> u_final = acc + dt*α2*k2 + dt*α3*k3
#   pre = u + dt*a31*k1 + (b31+b32)*q   -> h02     = pre + dt*a32*k2

# ---------------------------------------------------------------- variant A

function fastA_stage1_kernel!(
    h01, acc, pre, u, links, sqrt_dt, dt,
    two_kappa, m02, c3, inv_gamma, amp, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    mask, shift, _ = modelb_fixed_lattice(lattice)

    @inbounds for linear in id:stride:n
        x0, y0, base, row = modelb_fixed_geometry(linear, lattice)
        x_previous = row + ((x0 - Int32(1)) & mask) + Int32(1)
        y_previous = base + (((y0 - Int32(1)) & mask) << shift) + x0 + Int32(1)

        dw = sqrt_dt * (
            links[linear] - links[x_previous] +
            links[n + linear] - links[n + y_previous]
        )
        dz = sqrt_dt * (
            links[(n << 1) + linear] - links[(n << 1) + x_previous] +
            links[3 * n + linear] - links[3 * n + y_previous]
        )
        q = 0.5f0 * amp * (dw + SOSRA_INV_SQRT3 * dz)
        k1 = modelb_fixed_drift(u, linear, two_kappa, m02, c3, inv_gamma, lattice)
        u0 = u[linear]

        h01[linear] = u0 + dt * SOSRA_A21 * k1 + SOSRA_B21 * q
        acc[linear] = u0 + dt * SOSRA_ALPHA1 * k1 + amp * dw
        pre[linear] = u0 + dt * SOSRA_A31 * k1 + SOSRA_B31_B32 * q
    end
    return nothing
end

function fast_stage2_kernel!(
    u, acc, pre, h01, dt, two_kappa, m02, c3, inv_gamma, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        k2 = modelb_fixed_drift(h01, linear, two_kappa, m02, c3, inv_gamma, lattice)
        u[linear] = acc[linear] + dt * SOSRA_ALPHA2 * k2
        pre[linear] = pre[linear] + dt * SOSRA_A32 * k2
    end
    return nothing
end

function fast_stage3_kernel!(
    u, pre, dt, two_kappa, m02, c3, inv_gamma, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        k3 = modelb_fixed_drift(pre, linear, two_kappa, m02, c3, inv_gamma, lattice)
        u[linear] = u[linear] + dt * SOSRA_ALPHA3 * k3
    end
    return nothing
end

# ---------------------------------------------------------------- variant B

# Philox4x32: counter-based, so link noise at any site can be recomputed in
# registers from (site, step) instead of being read back from a stored field.
const PHILOX_M0 = 0xD2511F53
const PHILOX_M1 = 0xCD9E8D57
const PHILOX_W0 = 0x9E3779B9
const PHILOX_W1 = 0xBB67AE85

@inline function philox4x32(c0, c1, c2, c3, k0, k1, ::Val{R}) where R
    for _ in 1:R
        p0 = UInt64(PHILOX_M0) * UInt64(c0)
        p1 = UInt64(PHILOX_M1) * UInt64(c2)
        hi0 = UInt32(p0 >> 32); lo0 = UInt32(p0 & 0xffffffff)
        hi1 = UInt32(p1 >> 32); lo1 = UInt32(p1 & 0xffffffff)
        c0, c1, c2, c3 = hi1 ⊻ c1 ⊻ k0, lo1, hi0 ⊻ c3 ⊻ k1, lo0
        k0 += PHILOX_W0; k1 += PHILOX_W1
    end
    return c0, c1, c2, c3
end

# 2^-24 keeps the mantissa exact; +1 avoids log(0).
@inline uniform01(x::UInt32) = (Float32(x >> 8) + 1.0f0) * 5.9604645f-8

# __logf/__sinf (2 ulp) are ample for noise and halve the generation time.
@inline function box_muller(a::UInt32, b::UInt32)
    @fastmath begin
        r = sqrt(-2.0f0 * log(uniform01(a)))
        s, c = sincos(6.2831855f0 * uniform01(b))
        return r * c, r * s
    end
end

# Four link noises at one site: (wx, zx) from the first uniform pair,
# (wy, zy) from the second.
@inline function site_links(site::Int32, step::UInt32, k0, k1)
    a, b, c, d = philox4x32(reinterpret(UInt32, site), step, 0x0, 0x0, k0, k1, Val(10))
    wx, zx = box_muller(a, b)
    wy, zy = box_muller(c, d)
    return wx, zx, wy, zy
end

function fastB_stage1_kernel!(
    h01, acc, pre, u, sqrt_dt, dt, step, k0, k1,
    two_kappa, m02, c3, inv_gamma, amp, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    mask, shift, _ = modelb_fixed_lattice(lattice)

    @inbounds for linear in id:stride:n
        x0, y0, base, row = modelb_fixed_geometry(linear, lattice)
        x_previous = row + ((x0 - Int32(1)) & mask) + Int32(1)
        y_previous = base + (((y0 - Int32(1)) & mask) << shift) + x0 + Int32(1)

        wx, zx, wy, zy = site_links(linear, step, k0, k1)
        wxm, zxm, _, _ = site_links(x_previous, step, k0, k1)
        _, _, wym, zym = site_links(y_previous, step, k0, k1)

        dw = sqrt_dt * (wx - wxm + wy - wym)
        dz = sqrt_dt * (zx - zxm + zy - zym)
        q = 0.5f0 * amp * (dw + SOSRA_INV_SQRT3 * dz)
        kk1 = modelb_fixed_drift(u, linear, two_kappa, m02, c3, inv_gamma, lattice)
        u0 = u[linear]

        h01[linear] = u0 + dt * SOSRA_A21 * kk1 + SOSRA_B21 * q
        acc[linear] = u0 + dt * SOSRA_ALPHA1 * kk1 + amp * dw
        pre[linear] = u0 + dt * SOSRA_A31 * kk1 + SOSRA_B31_B32 * q
    end
    return nothing
end

# ---------------------------------------------------------------- variant C

# Same counter-based noise, but generated by its own kernel into the two
# divergences dW and dZ.  Costs 2 extra array passes and keeps stage 1 at 32
# registers instead of 72.
function noise_div_kernel!(dwa, dza, sqrt_dt, step, k0, k1, lattice, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    mask, shift, _ = modelb_fixed_lattice(lattice)
    @inbounds for linear in id:stride:n
        x0, y0, base, row = modelb_fixed_geometry(linear, lattice)
        xp = row + ((x0 - Int32(1)) & mask) + Int32(1)
        yp = base + (((y0 - Int32(1)) & mask) << shift) + x0 + Int32(1)
        wx, zx, wy, zy = site_links(linear, step, k0, k1)
        wxm, zxm, _, _ = site_links(xp, step, k0, k1)
        _, _, wym, zym = site_links(yp, step, k0, k1)
        dwa[linear] = sqrt_dt * (wx - wxm + wy - wym)
        dza[linear] = sqrt_dt * (zx - zxm + zy - zym)
    end
    return nothing
end

function fastC_stage1_kernel!(
    h01, acc, pre, u, dwa, dza, dt,
    two_kappa, m02, c3, inv_gamma, amp, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        dw = dwa[linear]
        q = 0.5f0 * amp * (dw + SOSRA_INV_SQRT3 * dza[linear])
        k1 = modelb_fixed_drift(u, linear, two_kappa, m02, c3, inv_gamma, lattice)
        u0 = u[linear]
        h01[linear] = u0 + dt * SOSRA_A21 * k1 + SOSRA_B21 * q
        acc[linear] = u0 + dt * SOSRA_ALPHA1 * k1 + amp * dw
        pre[linear] = u0 + dt * SOSRA_A31 * k1 + SOSRA_B31_B32 * q
    end
    return nothing
end

# ---------------------------------------------------------------- drivers

fast_buffers(u) = (h01=similar(u), acc=similar(u), pre=similar(u),
                   lattice=Val(size(u, 1)))

function fastA_step!(u, b, links, dt, two_kappa, m02, c3, inv_gamma, amp)
    n = Int32(length(u))
    t1, b1 = 768, min(cld(length(u), 768), 264)
    t2, b2 = 1024, min(cld(length(u), 1024), 264)
    @cuda threads=t1 blocks=b1 fastA_stage1_kernel!(
        b.h01, b.acc, b.pre, u, links, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, amp, b.lattice, n)
    @cuda threads=t2 blocks=b2 fast_stage2_kernel!(
        u, b.acc, b.pre, b.h01, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    @cuda threads=t2 blocks=b2 fast_stage3_kernel!(
        u, b.pre, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    return nothing
end

function fastB_step!(u, b, step, key, dt, two_kappa, m02, c3, inv_gamma, amp)
    n = Int32(length(u))
    t1, b1 = 768, min(cld(length(u), 768), 264)
    t2, b2 = 1024, min(cld(length(u), 1024), 264)
    @cuda threads=t1 blocks=b1 fastB_stage1_kernel!(
        b.h01, b.acc, b.pre, u, sqrt(dt), dt, UInt32(step), key[1], key[2],
        two_kappa, m02, c3, inv_gamma, amp, b.lattice, n)
    @cuda threads=t2 blocks=b2 fast_stage2_kernel!(
        u, b.acc, b.pre, b.h01, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    @cuda threads=t2 blocks=b2 fast_stage3_kernel!(
        u, b.pre, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    return nothing
end

fastC_buffers(u) = (h01=similar(u), acc=similar(u), pre=similar(u),
                    dw=similar(u), dz=similar(u), lattice=Val(size(u, 1)))

function fastC_step!(u, b, step, key, dt, two_kappa, m02, c3, inv_gamma, amp)
    n = Int32(length(u))
    t2, b2 = 1024, min(cld(length(u), 1024), 264)
    @cuda threads=256 blocks=min(cld(length(u), 256), 1056) noise_div_kernel!(
        b.dw, b.dz, sqrt(dt), UInt32(step), key[1], key[2], b.lattice, n)
    @cuda threads=t2 blocks=b2 fastC_stage1_kernel!(
        b.h01, b.acc, b.pre, u, b.dw, b.dz, dt,
        two_kappa, m02, c3, inv_gamma, amp, b.lattice, n)
    @cuda threads=t2 blocks=b2 fast_stage2_kernel!(
        u, b.acc, b.pre, b.h01, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    @cuda threads=t2 blocks=b2 fast_stage3_kernel!(
        u, b.pre, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    return nothing
end

# ------------------------------------------------- Euler--Maruyama reference
# One stencil, one Wiener field, no stage storage: 2 array passes per step.
# Pack (wx, wy) into a single Box--Muller pair so a site needs 3 pairs total.
@inline function site_w(site::Int32, step::UInt32, k0, k1)
    a, b, _, _ = philox4x32(reinterpret(UInt32, site), step, 0x2, 0x0, k0, k1, Val(10))
    return box_muller(a, b)
end

function em_kernel!(
    unew, u, sqrt_dt, dt, step, k0, k1,
    two_kappa, m02, c3, inv_gamma, amp, lattice, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    mask, shift, _ = modelb_fixed_lattice(lattice)
    @inbounds for linear in id:stride:n
        x0, y0, base, row = modelb_fixed_geometry(linear, lattice)
        xp = row + ((x0 - Int32(1)) & mask) + Int32(1)
        yp = base + (((y0 - Int32(1)) & mask) << shift) + x0 + Int32(1)
        wx, wy = site_w(linear, step, k0, k1)
        wxm, _ = site_w(xp, step, k0, k1)
        _, wym = site_w(yp, step, k0, k1)
        drift = modelb_fixed_drift(u, linear, two_kappa, m02, c3, inv_gamma, lattice)
        unew[linear] = u[linear] + dt * drift +
                       amp * sqrt_dt * (wx - wxm + wy - wym)
    end
    return nothing
end

function em_step!(u, unew, step, key, dt, two_kappa, m02, c3, inv_gamma, amp)
    n = Int32(length(u))
    @cuda threads=768 blocks=min(cld(length(u), 768), 264) em_kernel!(
        unew, u, sqrt(dt), dt, UInt32(step), key[1], key[2],
        two_kappa, m02, c3, inv_gamma, amp, Val(size(u, 1)), n)
    return nothing
end

modelb_coefficients(; γ=1.0f0, T=1.0f0, κ=0.28f0, λ=0.022f0) = (
    2.0f0 * κ,
    2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ,
    4.0f0 * λ,
    1.0f0 / γ,
    sqrt(2.0f0 * T / γ),
)
