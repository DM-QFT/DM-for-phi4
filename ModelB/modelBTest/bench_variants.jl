using CUDA, Random, Statistics, Printf

include(joinpath(@__DIR__, "modelb_fast.jl"))

# Baseline step driven by an externally supplied link field, so variant A can be
# compared against it with an identical noise realisation.
function baseline_step_with_links!(u, b, links, dt, two_kappa, m02, c3, inv_gamma, amp)
    n = Int32(length(u))
    t1, b1 = 768, min(cld(length(u), 768), 264)
    t2, b2 = 1024, min(cld(length(u), 1024), 264)
    @cuda threads=t1 blocks=b1 modelb_fixed_stage1_kernel!(
        b.h01, b.h02, b.k1, b.q, u, links, sqrt(dt), dt,
        two_kappa, m02, c3, inv_gamma, amp, b.lattice, n)
    @cuda threads=t2 blocks=b2 modelb_fixed_stage2_kernel!(
        u, b.h02, b.h01, b.k1, b.q, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    @cuda threads=t2 blocks=b2 modelb_fixed_stage3_kernel!(
        u, b.h02, dt, two_kappa, m02, c3, inv_gamma, b.lattice, n)
    return nothing
end

const L, Ne = 128, 512
const dt = 0.005f0
const coef = modelb_coefficients()

# ------------------------------------------------- correctness: A == baseline

CUDA.seed!(20260726)
u0 = CUDA.randn(Float32, L, L, Ne)
links = similar(u0, Float32, (L, L, Ne, 4))
rng = CUDA.cuRAND.LibraryRNG(CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)

ua = copy(u0); ba = modelb_fixed_buffers(ua; seed=UInt64(1))
ub = copy(u0); bb = fast_buffers(ub)
Random.seed!(rng, UInt64(7))
for _ in 1:200
    Random.randn!(rng, links)
    baseline_step_with_links!(ua, ba, links, dt, coef...)
    fastA_step!(ub, bb, links, dt, coef...)
end
CUDA.synchronize()
@printf("variant A vs baseline, same noise, 200 steps: max|Δφ| = %.3e (rms φ = %.3f)\n",
        maximum(abs, ua .- ub), sqrt(mean(abs2, ua)))

# ------------------------------------------- correctness: B statistically ==

function run_baseline(u, steps, seed)
    b = modelb_fixed_buffers(u; seed)
    for _ in 1:steps
        modelb_fixed_step!(u, b, dt, coef...)
    end
    CUDA.synchronize(); u
end

function run_fastB(u, steps, seed)
    b = fast_buffers(u)
    key = (UInt32(seed & 0xffffffff), UInt32(seed >> 32))
    for s in 1:steps
        fastB_step!(u, b, s, key, dt, coef...)
    end
    CUDA.synchronize(); u
end

steps = round(Int, 10.0f0 / dt)
stats(u) = (mean(u), mean(abs2, u), mean(abs2, mean(u; dims=(1, 2))))
sa = stats(run_baseline(copy(u0), steps, UInt64(11)))
sb = stats(run_fastB(copy(u0), steps, UInt64(11)))
@printf("t=10 from identical IC (different noise streams):\n")
@printf("  baseline : <φ> = %+.6f  <φ²> = %.5f  <M²> = %.3e\n", sa...)
@printf("  variant B: <φ> = %+.6f  <φ²> = %.5f  <M²> = %.3e\n", sb...)
m0 = mean(u0; dims=(1, 2))
mb = mean(run_fastB(copy(u0), 2000, UInt64(3)); dims=(1, 2))
@printf("  variant B conservation: max|ΔM| = %.3e\n", maximum(abs, mb .- m0))

# --------------------------------------------------------------- throughput

function timed(f, u; steps=200)
    for _ in 1:20; f(u, 1); end
    CUDA.synchronize()
    median([CUDA.@elapsed(begin; for s in 1:steps; f(u, s); end; end) / steps for _ in 1:5])
end

println()
@printf("%-28s %10s %10s %10s\n", "step variant", "ms/step", "speedup", "buffers")
for (Lb, Neb) in ((128, 512), (128, 10240))
    u = CUDA.randn(Float32, Lb, Lb, Neb)
    n = length(u)
    bbase = modelb_fixed_buffers(u; seed=UInt64(5))
    lk = similar(u, Float32, (Lb, Lb, Neb, 4))
    Random.randn!(rng, lk)
    bfast = fast_buffers(u)
    bc = fastC_buffers(u)
    key = (UInt32(9), UInt32(0))

    t_base = timed((x, s) -> modelb_fixed_step!(x, bbase, dt, coef...), u)
    t_nofuse = timed((x, s) -> fastA_step!(x, bfast, lk, dt, coef...), u)
    t_a = timed((x, s) -> (Random.randn!(rng, lk);
                           fastA_step!(x, bfast, lk, dt, coef...)), u)
    t_b = timed((x, s) -> fastB_step!(x, bfast, s, key, dt, coef...), u)
    t_c = timed((x, s) -> fastC_step!(x, bc, s, key, dt, coef...), u)
    println("--- L=$Lb Ne=$Neb  ($(n) elements)")
    @printf("%-28s %10.4f %10.2f %10d\n", "solver/ModelB_fixed", 1e3t_base, 1.0, 8)
    @printf("%-28s %10.4f %10.2f %10d\n", "  (stages only, no noise)", 1e3t_nofuse, t_base / t_nofuse, 7)
    @printf("%-28s %10.4f %10.2f %10d\n", "A: regrouped + cuRAND", 1e3t_a, t_base / t_a, 7)
    @printf("%-28s %10.4f %10.2f %10d\n", "B: A + inline Philox", 1e3t_b, t_base / t_b, 3)
    @printf("%-28s %10.4f %10.2f %10d\n", "C: B, noise in own kernel", 1e3t_c, t_base / t_c, 5)
end
