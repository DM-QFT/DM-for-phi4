using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# At kappa=0.28 the lattice m02 = -0.328 < 0, so the long-wavelength modes are
# linearly unstable (spinodal).  An exponential integrator amplifies those modes
# exactly over the step while the saturating phi^3 term is lagged, so large dt is
# not automatically safe here even though the Gaussian test at kappa=0.2 was.
# Find the largest dt that survives a long run.

const L, Ne = 64, 64
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)
const m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ4) - 8.0f0 * κ

# growth rate of the fastest unstable linear mode
khat2 = range(0.0f0, 8.0f0; length=2001)
rate = maximum(@. -khat2 * (2κ * khat2 + m02) / γ)
@printf("m02 = %+.4f, fastest linear growth rate = %.4f (unstable for khat2 < %.3f)\n\n",
        m02, rate, -m02 / (2κ))

function survive(scheme, dt, tmax)
    CUDA.seed!(7)
    u0 = CUDA.randn(Float32, L, L, Ne)
    steps = round(Int, tmax / dt)
    if scheme == :sosra
        u = copy(u0); b = modelb_fixed_buffers(u; seed=UInt64(3))
        for _ in 1:steps; modelb_fixed_step!(u, b, dt, coef...); end
        CUDA.synchronize(); return mean(abs2, u)
    end
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, u0)
    (scheme == :rk2 ? etdrk2_evolve! : etd_evolve!)(
        uk, plan, steps, (UInt32(3), UInt32(0)), dt; γ, T, κ, λ=λ4)
    CUDA.synchronize()
    return mean(abs2, etd_state(plan, uk))
end

@printf("<phi^2> after the run (NaN/Inf = blown up), L=%d Ne=%d\n", L, Ne)
@printf("%-8s %8s %12s %12s\n", "scheme", "dt", "t=200", "t=2000")
for scheme in (:sosra, :etd1, :rk2), dt in (0.01f0, 0.02f0, 0.05f0, 0.1f0)
    a = survive(scheme, dt, 200.0f0)
    b = survive(scheme, dt, 2000.0f0)
    @printf("%-8s %8.3f %12.4f %12.4f\n", scheme, dt, a, b)
end
