using CUDA, JLD2, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Test B diverged at dt=0.002 but not at dt=0.001.  Find out whether the shared
# noise plumbing is broken or the trajectories genuinely separate.

const L = 128
const κ, λ4, γ, T = 0.2705f0, 0.022f0, 1.0f0, 1.0f0
const c = modelb_coefficients(; γ, T, κ, λ=λ4)

rng = CUDA.cuRAND.LibraryRNG(CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)
a = CUDA.zeros(Float32, L, L, 1, 4)
b = similar(a)
Random.seed!(rng, UInt64(1001)); Random.randn!(rng, a)
Random.seed!(rng, UInt64(1001)); Random.randn!(rng, b)
@printf("1. reseeding cuRAND is deterministic: max|Δ| = %.3e\n", maximum(abs, a .- b))
Random.seed!(rng, UInt64(1002)); Random.randn!(rng, b)
@printf("   consecutive seeds 1001 vs 1002 correlate: %.4f\n",
        Statistics.cor(vec(Array(a)), vec(Array(b))))

path = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                         "cfgs_wolff_fahmc_k=$(κ)_l=$(λ4)_$(L)^2.jld2"))
u0 = jldopen(path, "r") do f
    d = JLD2.readmmap(JLD2.get_dataset(f, "cfgs"))
    CuArray(reshape(Array{Float32}(d[:, :, 5000]), L, L, 1))
end

# step both schemes forward together, sharing the link field each step
function lockstep(dt, nsteps, report)
    two_kappa, m02, c3, inv_gamma, amp = c
    us = copy(u0)
    bs = modelb_fixed_buffers(us; seed=UInt64(1))
    n = Int32(length(us))
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, copy(u0))
    ak, cube1 = similar(uk), similar(uk)
    H = plan.H
    nr, nk = Int32(length(plan.real)), Int32(length(uk))
    invn = 1.0f0 / Float32(L * L)
    links = CUDA.zeros(Float32, L, L, 1, 4)
    dW = similar(plan.real)
    @printf("\ndt=%g\n%8s %14s %14s %14s\n", dt, "step", "rms Δφ/rms φ", "rms φ SOSRA", "rms φ ETD")
    for s in 1:nsteps
        Random.seed!(rng, UInt64(1000 + s)); Random.randn!(rng, links)
        @cuda threads=768 blocks=min(cld(length(us), 768), 264) modelb_fixed_stage1_kernel!(
            bs.h01, bs.h02, bs.k1, bs.q, us, links, sqrt(dt), dt, c..., bs.lattice, n)
        @cuda threads=1024 blocks=min(cld(length(us), 1024), 264) modelb_fixed_stage2_kernel!(
            us, bs.h02, bs.h01, bs.k1, bs.q, dt, two_kappa, m02, c3, inv_gamma, bs.lattice, n)
        @cuda threads=1024 blocks=min(cld(length(us), 1024), 264) modelb_fixed_stage3_kernel!(
            us, bs.h02, dt, two_kappa, m02, c3, inv_gamma, bs.lattice, n)

        xi_x = @view links[:, :, :, 1]
        xi_y = @view links[:, :, :, 2]
        @. dW = amp * sqrt(dt) * (xi_x - $circshift(xi_x, (1, 0, 0)) +
                                  xi_y - $circshift(xi_y, (0, 1, 0)))
        noisek = plan.forward * dW
        mul!(plan.real, plan.backward, uk)
        @cuda threads=1024 blocks=min(cld(nr, 1024), 264) etd_cube_kernel!(plan.real, invn, nr)
        mul!(cube1, plan.forward, plan.real)
        @cuda threads=256 blocks=min(cld(nk, 256), 1056) ext_predict_kernel!(
            ak, uk, cube1, noisek, dt, two_kappa, m02, c3, inv_gamma,
            Int32(H), Int32(L), nk)
        mul!(plan.real, plan.backward, ak)
        @cuda threads=1024 blocks=min(cld(nr, 1024), 264) etd_cube_kernel!(plan.real, invn, nr)
        mul!(plan.cubek, plan.forward, plan.real)
        @cuda threads=256 blocks=min(cld(nk, 256), 1056) etdrk2_correct_kernel!(
            uk, ak, cube1, plan.cubek, dt, two_kappa, m02, c3, inv_gamma,
            Int32(H), Int32(L), nk)

        if s in report
            ue = etd_state(plan, uk)
            CUDA.synchronize()
            @printf("%8d %14.4e %14.6f %14.6f\n", s,
                    sqrt(mean(abs2, us .- ue)) / sqrt(mean(abs2, us)),
                    sqrt(mean(abs2, us)), sqrt(mean(abs2, ue)))
            flush(stdout)
        end
    end
end

function ext_predict_kernel!(ak, uk, cube1, noisek, dt, two_kappa, m02, c3,
                             inv_gamma, H, L, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        z = linear - Int32(1); i = z % H; j = (z ÷ H) % L
        selfconj, skip, mirror = etd_canonical(i, j, H, L)
        skip && continue
        khat2, λ = etd_mode(i, j, L, two_kappa, m02, inv_gamma)
        x = λ * dt
        old = uk[linear]
        v = old + expm1(-x) * old +
            (-inv_gamma * c3 * khat2 * dt * g1(x)) * cube1[linear] +
            sqrt(g2(x)) * noisek[linear]
        selfconj && (v = Complex(real(v), 0.0f0))
        ak[linear] = v
        mirror && (ak[linear + H * (L - Int32(2) * j)] = conj(v))
    end
    return nothing
end

report = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
lockstep(0.002f0, 5000, report)
lockstep(0.001f0, 5000, report)
