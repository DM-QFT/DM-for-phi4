using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Is the ETD blowup a dt threshold (lagged phi^3) or something dt-independent?
# Run to t=2000 at several dt and record when <phi^2> first exceeds 5.

const L, Ne = 64, 16
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0

CUDA.seed!(7)
const u0 = CUDA.randn(Float32, L, L, Ne)

function blowup_time(dt; rk2=false, tmax=2000.0f0)
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, copy(u0))
    ev = rk2 ? etdrk2_evolve! : etd_evolve!
    chunk = round(Int, 10.0f0 / dt)
    s = 0
    for block in 1:round(Int, tmax / 10.0f0)
        ev(uk, plan, chunk, (UInt32(3), UInt32(0)), dt; γ, T, κ, λ=λ4, step0=s)
        s += chunk
        v = mean(abs2, etd_state(plan, uk))
        (!isfinite(v) || v > 5.0f0) && return block * 10.0f0, v
    end
    return Inf32, mean(abs2, etd_state(plan, uk))
end

@printf("%-8s %8s %12s %12s\n", "scheme", "dt", "t_blowup", "<φ²> there")
for dt in (0.02f0, 0.01f0, 0.005f0, 0.002f0)
    t, v = blowup_time(dt)
    @printf("%-8s %8.4f %12.1f %12.4f\n", "ETD1", dt, t, v)
end
for dt in (0.02f0, 0.01f0, 0.005f0)
    t, v = blowup_time(dt; rk2=true)
    @printf("%-8s %8.4f %12.1f %12.4f\n", "ETDRK2", dt, t, v)
end
