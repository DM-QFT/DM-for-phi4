using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Accuracy per unit cost for four integrators of the same lattice Model B.
# Accuracy metric: relative error of the conditional variance of phi(t=10) given
# a fixed phi_0, against SOSRA at dt=0.0005.
# Cost metric: measured ms per step at a production shape.

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

function run_em(dt, seed)
    u = copy(φ0); unew = similar(u); key = (UInt32(seed), UInt32(0))
    for s in 1:round(Int, tfinal / dt)
        em_step!(u, unew, s, key, dt, coef...)
        u, unew = unew, u
    end
    CUDA.synchronize(); rfft(u, (1, 2))
end

function run_etd(dt, seed; rk2=false)
    plan = etd_plan(similar(φ0))
    uk = etd_spectrum(plan, copy(φ0))
    ev = rk2 ? etdrk2_evolve! : etd_evolve!
    ev(uk, plan, round(Int, tfinal / dt), (UInt32(seed), UInt32(0)), dt; γ, T, κ, λ=λ4)
    CUDA.synchronize(); uk
end

function var_k(uk)
    μ = dropdims(mean(uk; dims=3); dims=3)
    Array(dropdims(mean(abs2, uk .- reshape(μ, size(μ)..., 1); dims=3); dims=3))
end

const vref = var_k(run_sosra(0.0005f0, 101))
err(uk) = mean((var_k(uk) .- vref) ./ vref)

@printf("conditional variance of phi(t=10)|phi_0, L=%d Ne=%d, kappa=%.3f\n", L, Ne, κ)
@printf("noise floor (same scheme, other seed): %+.4f%%\n\n", 100err(run_sosra(0.0005f0, 202)))
@printf("%-10s %8s %12s\n", "scheme", "dt", "Δvar/var")
results = Dict{String, Vector{Tuple{Float32, Float64}}}()
for (name, f, dts) in (
        ("EM", (dt) -> run_em(dt, 101), (0.001f0, 0.005f0, 0.01f0, 0.02f0)),
        ("SOSRA", (dt) -> run_sosra(dt, 101), (0.002f0, 0.01f0, 0.02f0, 0.05f0)),
        ("ETD1", (dt) -> run_etd(dt, 101), (0.01f0, 0.05f0, 0.1f0, 0.25f0)),
        ("ETDRK2", (dt) -> run_etd(dt, 101; rk2=true), (0.05f0, 0.1f0, 0.25f0, 0.5f0)))
    rows = Tuple{Float32, Float64}[]
    for dt in dts
        e = err(f(dt))
        push!(rows, (dt, e))
        @printf("%-10s %8.4f %11.3f%%\n", name, dt, 100e)
    end
    results[name] = rows
end

# ------------------------------------------------------------------ cost

bench(f) = (for _ in 1:5; f(); end; CUDA.synchronize();
            median([CUDA.@elapsed(f()) for _ in 1:10]))

println()
@printf("%-10s %10s %12s %12s\n", "scheme", "L", "Ne", "ms/step")
for (Lb, Neb) in ((128, 512), (128, 10240))
    u = CUDA.randn(Float32, Lb, Lb, Neb)
    unew = similar(u)
    b8 = modelb_fixed_buffers(u; seed=UInt64(5))
    bf = fast_buffers(u)
    plan = etd_plan(u)
    uk = etd_spectrum(plan, u)
    ak = similar(uk); c1 = similar(uk)
    key = (UInt32(9), UInt32(0))
    c = coef
    ts = (
        "EM" => bench(() -> em_step!(u, unew, 1, key, 0.01f0, c...)),
        "SOSRA(now)" => bench(() -> modelb_fixed_step!(u, b8, 0.01f0, c...)),
        "SOSRA(fused)" => bench(() -> fastB_step!(u, bf, 1, key, 0.01f0, c...)),
        "ETD1" => bench(() -> etd_step!(uk, plan, 1, key, 0.01f0, c[1], c[2], c[3], c[4], 2.0f0T / γ)),
        "ETDRK2" => bench(() -> etdrk2_step!(uk, plan, ak, c1, 1, key, 0.01f0, c[1], c[2], c[3], c[4], 2.0f0T / γ)),
    )
    for (name, t) in ts
        @printf("%-10s %10d %12d %12.4f\n", name, Lb, Neb, 1e3t)
    end
    println()
end
