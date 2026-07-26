using CUDA, Random, Statistics, Printf

include(joinpath(@__DIR__, "modelb_fast.jl"))

# Why is inline Philox slower than reading stored link fields?  Isolate the noise
# generation from the drift stencil and look at registers/occupancy.

@inline function box_muller_fast(a::UInt32, b::UInt32)
    @fastmath begin
        r = sqrt(-2.0f0 * log(uniform01(a)))
        s, c = sincos(6.2831855f0 * uniform01(b))
        return r * c, r * s
    end
end

@inline function site_links_r(site::Int32, step::UInt32, k0, k1, ::Val{R}, fast) where R
    a, b, c, d = philox4x32(reinterpret(UInt32, site), step, 0x0, 0x0, k0, k1, Val(R))
    if fast
        wx, zx = box_muller_fast(a, b); wy, zy = box_muller_fast(c, d)
    else
        wx, zx = box_muller(a, b); wy, zy = box_muller(c, d)
    end
    return wx, zx, wy, zy
end

# Generate the two conserved noise divergences directly into dw, dz.
function noise_div_kernel!(dw, dz, sqrt_dt, step, k0, k1, lattice, n, ::Val{R}, fast) where R
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    mask, shift, _ = modelb_fixed_lattice(lattice)
    @inbounds for linear in id:stride:n
        x0, y0, base, row = modelb_fixed_geometry(linear, lattice)
        xp = row + ((x0 - Int32(1)) & mask) + Int32(1)
        yp = base + (((y0 - Int32(1)) & mask) << shift) + x0 + Int32(1)
        wx, zx, wy, zy = site_links_r(linear, step, k0, k1, Val(R), fast)
        wxm, zxm, _, _ = site_links_r(xp, step, k0, k1, Val(R), fast)
        _, _, wym, zym = site_links_r(yp, step, k0, k1, Val(R), fast)
        dw[linear] = sqrt_dt * (wx - wxm + wy - wym)
        dz[linear] = sqrt_dt * (zx - zxm + zy - zym)
    end
    return nothing
end

# Stage 1 taking the precomputed divergences instead of link fields.
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

const L, Ne = 128, 10240
const n32 = Int32(L * L * Ne)
const lat = Val(L)
u = CUDA.randn(Float32, L, L, Ne)
dwa, dza = similar(u), similar(u)
links = similar(u, Float32, (L, L, Ne, 4))
rng = CUDA.cuRAND.LibraryRNG(CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)
Random.seed!(rng, UInt64(1))

bench(f) = (for _ in 1:5; f(); end; CUDA.synchronize();
            median([CUDA.@elapsed(f()) for _ in 1:20]))

t_curand = bench(() -> Random.randn!(rng, links))
launch_div(R, fast) = @cuda threads=256 blocks=min(cld(length(u), 256), 1056) noise_div_kernel!(
    dwa, dza, 0.07f0, UInt32(3), UInt32(9), UInt32(0), lat, n32, Val(R), fast)

@printf("noise generation only, %d elements\n", length(u))
@printf("%-38s %10s\n", "", "ms")
@printf("%-38s %10.4f\n", "cuRAND randn! into 4 link fields", 1e3t_curand)
for R in (10, 7, 4), fast in (false, true)
    t = bench(() -> launch_div(R, fast))
    @printf("%-38s %10.4f\n", "inline Philox-$R div->2 fields$(fast ? " (fastmath)" : "")", 1e3t)
end

# Register pressure of the three stage-1 variants.
println()
k_a = @cuda launch=false fastA_stage1_kernel!(
    dwa, dza, u, u, links, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, lat, n32)
k_b = @cuda launch=false fastB_stage1_kernel!(
    dwa, dza, u, u, 1.0f0, 1.0f0, UInt32(1), UInt32(1), UInt32(1),
    1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, lat, n32)
k_c = @cuda launch=false fastC_stage1_kernel!(
    dwa, dza, u, u, dwa, dza, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, lat, n32)
k_n = @cuda launch=false noise_div_kernel!(
    dwa, dza, 1.0f0, UInt32(1), UInt32(1), UInt32(1), lat, n32, Val(10), true)
for (name, k) in (("stage1 A (reads links)", k_a), ("stage1 B (inline philox)", k_b),
                  ("stage1 C (reads dw,dz)", k_c), ("noise div kernel", k_n))
    @printf("%-38s regs=%3d  maxthreads=%5d\n", name,
            CUDA.registers(k), CUDA.maxthreads(k))
end
