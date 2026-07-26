using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Step-size error on the quantity the project actually uses: the conditional law
# p(phi(t=10) | phi_0) at the production parameters.  One fixed phi_0 is
# replicated Ne times, so the ensemble spread *is* that conditional law.
# Reference = SOSRA at dt=0.0005; the noise floor is set by rerunning the
# reference with a different seed.

const L, Ne = 64, 4096
const κ, λ4, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0
const tfinal = 10.0f0
const coef = modelb_coefficients(; γ, T, κ, λ=λ4)

CUDA.seed!(20260726)
const φ0 = repeat(CUDA.randn(Float32, L, L, 1), 1, 1, Ne)

function sosra_to_t(dt, seed)
    u = copy(φ0)
    b = modelb_fixed_buffers(u; seed=UInt64(seed))
    for _ in 1:round(Int, tfinal / dt); modelb_fixed_step!(u, b, dt, coef...); end
    CUDA.synchronize()
    return rfft(u, (1, 2))
end

function etd_to_t(dt, seed)
    plan = etd_plan(similar(φ0))
    uk = etd_spectrum(plan, copy(φ0))
    etd_evolve!(uk, plan, round(Int, tfinal / dt), (UInt32(seed), UInt32(0)), dt;
                γ, T, κ, λ=λ4)
    CUDA.synchronize()
    return uk
end

# conditional mean and variance per k mode
function moments(uk)
    μ = dropdims(mean(uk; dims=3); dims=3)
    v = dropdims(mean(abs2, uk .- reshape(μ, size(μ)..., 1); dims=3); dims=3)
    return Array(μ), Array(v)
end

function compare(name, uk, ref)
    μ, v = moments(uk)
    μr, vr = ref
    e_mean = sqrt(sum(abs2, μ .- μr) / sum(abs2, μr))
    e_var = mean((v .- vr) ./ vr)
    @printf("%-22s  ||Δmean||/||mean|| = %8.4f   <Δvar/var> = %+8.4f\n",
            name, e_mean, e_var)
end

ref = moments(sosra_to_t(0.0005f0, 101))
println("conditional law of phi(t=10) given a fixed phi_0, L=$L Ne=$Ne, ",
        "kappa=$κ lambda=$λ4")
println("reference: SOSRA dt=0.0005 (seed 101)\n")
compare("SOSRA dt=0.0005 (seed 2)", sosra_to_t(0.0005f0, 202), ref)   # noise floor
for dt in (0.002f0, 0.005f0, 0.02f0)
    compare("SOSRA dt=$dt", sosra_to_t(dt, 101), ref)
end
println()
for dt in (0.005f0, 0.02f0, 0.05f0, 0.1f0, 0.25f0)
    compare("ETD   dt=$dt", etd_to_t(dt, 101), ref)
end
