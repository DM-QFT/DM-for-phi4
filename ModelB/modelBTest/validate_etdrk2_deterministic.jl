using CUDA, Statistics, Printf

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Pointwise check of the drift path.  At T=0 both schemes integrate the same ODE
# and both are second order in the deterministic part, so their difference must
# vanish as dt^2.  This catches any error in khat2, m02, c3, the FFT
# normalisation or the Laplacian discretisation, which statistical tests can
# only see indirectly.

const L, Ne = 64, 16
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 0.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)
const tfinal = 1.0f0

CUDA.seed!(20260727)
const u0 = CUDA.randn(Float32, L, L, Ne)

function sosra(dt)
    u = copy(u0); b = modelb_fixed_buffers(u; seed=UInt64(1))
    for _ in 1:round(Int, tfinal / dt); modelb_fixed_step!(u, b, dt, coef...); end
    CUDA.synchronize(); u
end

function etdrk2(dt)
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, copy(u0))
    etdrk2_evolve!(uk, plan, round(Int, tfinal / dt), (UInt32(1), UInt32(0)), dt;
                   γ, T, κ, λ=λ4)
    CUDA.synchronize(); copy(etd_state(plan, uk))
end

@printf("deterministic (T=0) trajectory to t=%.1f, L=%d Ne=%d, kappa=%.3f\n",
        tfinal, L, Ne, κ)
@printf("%10s %14s %14s %10s\n", "dt", "max|Δφ|", "rms Δφ", "ratio")
prev = NaN
for dt in (0.02f0, 0.01f0, 0.005f0, 0.0025f0, 0.00125f0)
    d = sosra(dt) .- etdrk2(dt)
    m = maximum(abs, d)
    @printf("%10.5f %14.3e %14.3e %10.2f\n", dt, m, sqrt(mean(abs2, d)), prev / m)
    global prev = m
end
@printf("\nreference scale: rms φ(t=%.1f) = %.4f\n", tfinal, sqrt(mean(abs2, sosra(0.00125f0))))
