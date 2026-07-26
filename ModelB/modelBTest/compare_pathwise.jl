using CUDA, JLD2, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Pathwise comparison from ONE L=128 kappa=0.2705 HMC configuration, 5000 steps.
# Statistical agreement is not proof; here the two schemes are compared field by
# field and mode by mode.
#   A: T=0.  Both integrate the same ODE, so the difference is purely the drift
#      path (linear symbol, phi^3 term, FFT normalisation).
#   B: T=1, both stepped in lockstep off the SAME link field, so the difference
#      is only the noise handling: ETD applies the exact Ornstein-Uhlenbeck
#      weight sqrt(g2) to the increment while SOSRA uses the plain one and adds
#      a Levy-area proxy dZ that ETD has no counterpart for.  Both are O(dt)
#      differences, so the gap must be bounded and proportional to dt.
# Driving the two schemes in separate loops off a re-seeded generator does NOT
# reliably give them the same field; they must be advanced together.

const L = 128
const κ, λ4, γ = 0.2705f0, 0.022f0, 1.0f0
const CFG = 5000                      # which HMC configuration
const nsteps = 5000
const tfinal = nsteps * 0.002f0

function load_one()
    path = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                             "cfgs_wolff_fahmc_k=$(κ)_l=$(λ4)_$(L)^2.jld2"))
    jldopen(path, "r") do f
        @assert f["N"] == L && isapprox(f["κ"], κ) && isapprox(f["λ"], λ4)
        d = JLD2.readmmap(JLD2.get_dataset(f, "cfgs"))
        CuArray(reshape(Array{Float32}(d[:, :, CFG]), L, L, 1))
    end
end

const khat2 = [4.0f0 * (sinpi(i / L)^2 + sinpi(j / L)^2)
               for i in 0:(L ÷ 2), j in 0:(L - 1)]
const rk = round.(khat2; digits=5)
const groups = let v = sort(unique(rk))
    vcat([("k̂²=$(round(x; digits=4))", rk .== x) for x in v[2:4]],
         [("band 0.5-1", (khat2 .> 0.5) .& (khat2 .<= 1)),
          ("band 3-5", (khat2 .> 3) .& (khat2 .<= 5)),
          ("band 7-8", (khat2 .> 7) .& (khat2 .<= 8))])
end

# ETDRK2 predictor taking the noise spectrum from outside.
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

# Advance both schemes together.  shared=true feeds ETD the same conserved
# increment SOSRA builds from the link fields.
function lockstep(u0, dt, steps, T, rng)
    c = modelb_coefficients(; γ, T, κ, λ=λ4)
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
    rb, kb = min(cld(nr, 1024), 264), min(cld(nk, 256), 1056)
    for s in 1:steps
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
        @cuda threads=1024 blocks=rb etd_cube_kernel!(plan.real, invn, nr)
        mul!(cube1, plan.forward, plan.real)
        @cuda threads=256 blocks=kb ext_predict_kernel!(
            ak, uk, cube1, noisek, dt, two_kappa, m02, c3, inv_gamma,
            Int32(H), Int32(L), nk)
        mul!(plan.real, plan.backward, ak)
        @cuda threads=1024 blocks=rb etd_cube_kernel!(plan.real, invn, nr)
        mul!(plan.cubek, plan.forward, plan.real)
        @cuda threads=256 blocks=kb etdrk2_correct_kernel!(
            uk, ak, cube1, plan.cubek, dt, two_kappa, m02, c3, inv_gamma,
            Int32(H), Int32(L), nk)
    end
    CUDA.synchronize()
    return us, etd_state(plan, uk)
end

function report(label, us, ue)
    d = Array(us .- ue)
    fs = Array(rfft(us, (1, 2)))[:, :, 1]
    fe = Array(rfft(ue, (1, 2)))[:, :, 1]
    @printf("%-20s max|Δφ|=%.2e  rmsΔφ/rmsφ=%.3e\n", label,
            maximum(abs, d), sqrt(mean(abs2, d)) / sqrt(mean(abs2, Array(us))))
    @printf("%-20s", "")
    for (name, mask) in groups
        @printf("  %s: |Δφ_k|/|φ_k|=%.1e (|φ_k|=%.1e)", name,
                mean(abs.(fe[mask] .- fs[mask]) ./ abs.(fs[mask])),
                mean(abs.(fs[mask])))
    end
    println(); flush(stdout)
end

const u0 = load_one()
rng = CUDA.cuRAND.LibraryRNG(CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)
@printf("HMC configuration #%d, L=%d, kappa=%.4f, %d steps to t=%.1f\n",
        CFG, L, κ, nsteps, tfinal)
@printf("rms φ(0)=%.4f  M(0)=%+.6f\n", sqrt(mean(abs2, u0)), mean(u0))
println("\n(the T=0 UV bands are a 0/0 artefact: those modes decay as exp(-32.6 t)")
println(" and sit at the round-off floor by t=10, see the |φ_k| column)\n")

println("--- A: deterministic, T=0 ---")
for dt in (0.002f0, 0.001f0, 0.0005f0)
    report("dt=$dt", lockstep(u0, dt, round(Int, tfinal / dt), 0.0f0, rng)...)
end

println("\n--- B: T=1, both driven by the same conserved increment dW ---")
for dt in (0.002f0, 0.001f0, 0.0005f0)
    report("dt=$dt", lockstep(u0, dt, round(Int, tfinal / dt), 1.0f0, rng)...)
end
