using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# ms/step for every integrator, each measured with only its own buffers resident.

const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)
const dt = 0.01f0
const key = (UInt32(9), UInt32(0))

bench(f) = (for _ in 1:10; f(); end; CUDA.synchronize();
            median([CUDA.@elapsed(f()) for _ in 1:20]))

function measure(scheme, L, Ne)
    u = CUDA.randn(Float32, L, L, Ne)
    t = if scheme == "SOSRA (solver/)"
        b = modelb_fixed_buffers(u; seed=UInt64(5))
        bench(() -> modelb_fixed_step!(u, b, dt, coef...))
    elseif scheme == "SOSRA fused+inline"
        b = fast_buffers(u)
        bench(() -> fastB_step!(u, b, 1, key, dt, coef...))
    elseif scheme == "EM inline"
        v = similar(u)
        bench(() -> em_step!(u, v, 1, key, dt, coef...))
    elseif scheme == "ETD1"
        plan = etd_plan(u); uk = etd_spectrum(plan, u)
        bench(() -> etd_step!(uk, plan, 1, key, dt, coef[1], coef[2], coef[3], coef[4], 2.0f0T / γ))
    elseif scheme == "ETDRK2"
        plan = etd_plan(u); uk = etd_spectrum(plan, u)
        ak = similar(uk); c1 = similar(uk)
        bench(() -> etdrk2_step!(uk, plan, ak, c1, 1, key, dt, coef[1], coef[2], coef[3], coef[4], 2.0f0T / γ))
    end
    CUDA.reclaim()
    return t
end

for (L, Ne) in ((128, 512), (256, 128), (128, 10240))
    @printf("\nL=%d Ne=%d  (%d elements)\n", L, Ne, L * L * Ne)
    @printf("%-22s %10s %10s\n", "scheme", "ms/step", "vs solver")
    base = 0.0
    for scheme in ("SOSRA (solver/)", "SOSRA fused+inline", "EM inline", "ETD1", "ETDRK2")
        t = measure(scheme, L, Ne)
        scheme == "SOSRA (solver/)" && (base = t)
        @printf("%-22s %10.4f %10.2f\n", scheme, 1e3t, base / t)
    end
end
