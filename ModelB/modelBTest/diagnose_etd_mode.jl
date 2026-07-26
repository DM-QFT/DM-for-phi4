using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

const L, Ne = 64, 4
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 0.0f0

CUDA.seed!(7)
u0 = CUDA.randn(Float32, L, L, Ne)
plan = etd_plan(similar(u0))

# 1. does cuFFT's C2R overwrite its input?
uk = etd_spectrum(plan, copy(u0))
before = Array(uk)
mul!(plan.real, plan.backward, uk)
CUDA.synchronize()
@printf("C2R input preserved: max|Δuk| = %.3e\n\n", maximum(abs, Array(uk) .- before))

# 2. where does the deterministic blowup live in k space?
function scan(plan, u0, dt)
    uk = etd_spectrum(plan, copy(u0))
    chunk = round(Int, 10.0f0 / dt)
    s = 0
    khat2 = [4.0f0 * (sinpi(i / L)^2 + sinpi(j / L)^2)
             for i in 0:(L ÷ 2), j in 0:(L - 1)]
    @printf("%6s %10s %10s %14s %10s %8s\n",
            "t", "<φ²>", "max|φ|", "max|uk|²/N", "at khat2", "kx,ky")
    for block in 1:70
        etd_evolve!(uk, plan, chunk, (UInt32(3), UInt32(0)), dt; γ, T, κ, λ=λ4, step0=s)
        s += chunk
        t = block * 10.0
        t < 550 && continue
        φ = etd_state(plan, uk)
        S = real.(Array(dropdims(mean(abs2, uk; dims=3); dims=3))) ./ Float32(L^2)
        S[1, 1] = 0
        v, idx = findmax(S)
        @printf("%6.0f %10.4f %10.3f %14.4e %10.4f %4d,%3d\n",
                t, mean(abs2, φ), maximum(abs, φ), v, khat2[idx], idx[1] - 1, idx[2] - 1)
        isfinite(mean(abs2, φ)) || break
    end
end
scan(plan, u0, 0.01f0)
