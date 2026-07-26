using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Gate test with lambda_phi4 = 0: the lattice theory is Gaussian, so
# S(k) = <|phi_k|^2>/N = T/(m02 + 2 kappa khat2) exactly.  This checks both the
# ETD coefficients and the Hermitian symmetry of the k-space noise.

const L, Ne = 32, 8192
const κ, λ4, γ, T = 0.2f0, 0.0f0, 1.0f0, 1.0f0
const m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ4) - 8.0f0 * κ
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)

khat2 = [4.0f0 * (sinpi(i / L)^2 + sinpi(j / L)^2)
         for i in 0:(L ÷ 2), j in 0:(L - 1)]
exact = T ./ (m02 .+ 2κ .* khat2)
exact[1, 1] = NaN                       # k=0 is conserved, not thermal

spectrum(u) = dropdims(mean(abs2, rfft(u, (1, 2)); dims=3); dims=3) ./ Float32(L^2)
spectrum_k(uk) = dropdims(mean(abs2, uk; dims=3); dims=3) ./ Float32(L^2)

function report(name, S)
    rel = (Array(S) .- exact) ./ exact
    rel[1, 1] = 0.0
    uv = rel[end, L ÷ 2 + 1]            # highest mode, khat2 = 8
    @printf("%-24s  mean %+7.3f%%   max|rel| %6.3f%%   UV mode %+7.3f%%\n",
            name, 100mean(rel), 100maximum(abs, rel), 100uv)
end

# thermalise, then average S(k) over decorrelated snapshots
function measure_etd(dt; therm=500.0f0, snaps=8, gap=50.0f0)
    CUDA.seed!(4242)
    plan = etd_plan(CUDA.zeros(Float32, L, L, Ne))
    uk = etd_spectrum(plan, CUDA.randn(Float32, L, L, Ne))
    key = (UInt32(31), UInt32(0))
    s = 0
    etd_evolve!(uk, plan, round(Int, therm / dt), key, dt; γ, T, κ, λ=λ4, step0=s)
    s += round(Int, therm / dt)
    S = zeros(Float32, L ÷ 2 + 1, L)
    for _ in 1:snaps
        etd_evolve!(uk, plan, round(Int, gap / dt), key, dt; γ, T, κ, λ=λ4, step0=s)
        s += round(Int, gap / dt)
        S .+= Array(spectrum_k(uk))
    end
    return S ./ snaps
end

function measure_sosra(dt; therm=500.0f0, snaps=8, gap=50.0f0)
    CUDA.seed!(4242)
    u = CUDA.randn(Float32, L, L, Ne)
    b = modelb_fixed_buffers(u; seed=UInt64(31))
    for _ in 1:round(Int, therm / dt); modelb_fixed_step!(u, b, dt, coef...); end
    S = zeros(Float32, L ÷ 2 + 1, L)
    for _ in 1:snaps
        for _ in 1:round(Int, gap / dt); modelb_fixed_step!(u, b, dt, coef...); end
        S .+= Array(spectrum(u))
    end
    return S ./ snaps
end

println("Gaussian lattice theory, L=$L Ne=$Ne kappa=$κ lambda=0, m02=$m02")
println("deviation of S(k) from the exact T/(m02+2 kappa khat2):\n")
for dt in (0.002f0, 0.01f0, 0.05f0)
    report("SOSRA dt=$dt", measure_sosra(dt))
end
println()
for dt in (0.01f0, 0.1f0, 0.5f0, 2.0f0)
    report("ETD   dt=$dt", measure_etd(dt))
end
