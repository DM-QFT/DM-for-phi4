using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Track the ETD1 run that goes NaN: is it a slow drift, a UV pile-up, or a
# single-mode blowup?

const L, Ne = 64, 16
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0
const dt = 0.01f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)

CUDA.seed!(7)
u0 = CUDA.randn(Float32, L, L, Ne)

plan = etd_plan(similar(u0))
uk = etd_spectrum(plan, copy(u0))
usos = copy(u0)
bsos = modelb_fixed_buffers(usos; seed=UInt64(3))

khat2 = CuArray([4.0f0 * (sinpi(i / L)^2 + sinpi(j / L)^2)
                 for i in 0:(L ÷ 2), j in 0:(L - 1)])
uvmask = Array(khat2) .> 6.0f0

@printf("%8s | %10s %10s %10s | %10s %10s\n",
        "t", "ETD <φ²>", "ETD max|φ|", "ETD UV S", "SOS <φ²>", "SOS max|φ|")
chunk = round(Int, 50.0f0 / dt)
for block in 1:20
    etd_evolve!(uk, plan, chunk, (UInt32(3), UInt32(0)), dt; γ, T, κ, λ=λ4,
                step0=(block - 1) * chunk)
    for _ in 1:chunk; modelb_fixed_step!(usos, bsos, dt, coef...); end
    φ = etd_state(plan, uk)
    S = Array(dropdims(mean(abs2, uk; dims=3); dims=3)) ./ Float32(L^2)
    CUDA.synchronize()
    @printf("%8.0f | %10.4f %10.3f %10.3e | %10.4f %10.3f\n",
            block * 50.0, mean(abs2, φ), maximum(abs, φ), mean(S[uvmask]),
            mean(abs2, usos), maximum(abs, usos))
    isfinite(mean(abs2, φ)) || break
end
