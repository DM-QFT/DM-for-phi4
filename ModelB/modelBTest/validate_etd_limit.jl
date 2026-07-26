using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Do the Fourier and real-space schemes converge to the *same* dt->0 limit in the
# nonlinear regime, and does ETD conserve the zero mode exactly?

const L, Ne = 64, 4096
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0
const tfinal = 10.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)

CUDA.seed!(20260726)
const φ0 = repeat(CUDA.randn(Float32, L, L, 1), 1, 1, Ne)

function run_sosra(dt, seed)
    u = copy(φ0); b = modelb_fixed_buffers(u; seed=UInt64(seed))
    for _ in 1:round(Int, tfinal / dt); modelb_fixed_step!(u, b, dt, coef...); end
    CUDA.synchronize(); rfft(u, (1, 2))
end

function run_etd(dt, seed; rk2=false)
    plan = etd_plan(similar(φ0))
    uk = etd_spectrum(plan, copy(φ0))
    (rk2 ? etdrk2_evolve! : etd_evolve!)(
        uk, plan, round(Int, tfinal / dt), (UInt32(seed), UInt32(0)), dt; γ, T, κ, λ=λ4)
    CUDA.synchronize(); uk
end

function var_k(uk)
    μ = dropdims(mean(uk; dims=3); dims=3)
    Array(dropdims(mean(abs2, uk .- reshape(μ, size(μ)..., 1); dims=3); dims=3))
end

vref = var_k(run_sosra(0.0005f0, 101))
err(uk) = 100mean((var_k(uk) .- vref) ./ vref)

@printf("reference SOSRA dt=0.0005; noise floor %+.4f%%\n", err(run_sosra(0.0005f0, 202)))
@printf("ETD1   dt=0.002 : %+.4f%%\n", err(run_etd(0.002f0, 101)))
@printf("ETDRK2 dt=0.010 : %+.4f%%\n", err(run_etd(0.01f0, 101; rk2=true)))
@printf("ETDRK2 dt=0.020 : %+.4f%%\n", err(run_etd(0.02f0, 101; rk2=true)))

# zero mode: conserved exactly in Fourier space, to roundoff in real space
plan = etd_plan(similar(φ0))
uk = etd_spectrum(plan, copy(φ0))
m0 = Array(uk[1, 1, :])
etd_evolve!(uk, plan, 20000, (UInt32(1), UInt32(0)), 0.05f0; γ, T, κ, λ=λ4)
@printf("ETD1 zero mode after 20000 steps: max|ΔM| = %.3e\n",
        maximum(abs, Array(uk[1, 1, :]) .- m0) / (L * L))
