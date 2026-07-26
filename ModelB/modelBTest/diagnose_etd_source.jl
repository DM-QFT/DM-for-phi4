using CUDA, Statistics, Printf, Random

include(joinpath(@__DIR__, "modelb_etd.jl"))

# Blowup time is dt-independent, so it is not a step-size stability limit.
# Separate the two remaining suspects: the deterministic drift treatment and the
# k-space noise.

const L, Ne = 64, 16
const κ, λ4, γ = 0.28f0, 0.022f0, 1.0f0

CUDA.seed!(7)
const u0 = CUDA.randn(Float32, L, L, Ne)

function trace_etd(dt, T, key; tmax=1500.0f0, gap=50.0f0)
    plan = etd_plan(similar(u0))
    uk = etd_spectrum(plan, copy(u0))
    chunk = round(Int, gap / dt); s = 0
    out = Float32[]
    for _ in 1:round(Int, tmax / gap)
        etd_evolve!(uk, plan, chunk, key, dt; γ, T, κ, λ=λ4, step0=s)
        s += chunk
        push!(out, mean(abs2, etd_state(plan, uk)))
        isfinite(out[end]) || break
    end
    return out
end

function trace_sosra(dt, T; tmax=1500.0f0, gap=50.0f0)
    u = copy(u0)
    b = modelb_fixed_buffers(u; seed=UInt64(3))
    coef = modelb_coefficients(; γ, T, κ, λ=λ4)
    chunk = round(Int, gap / dt)
    out = Float32[]
    for _ in 1:round(Int, tmax / gap)
        for _ in 1:chunk; modelb_fixed_step!(u, b, dt, coef...); end
        push!(out, mean(abs2, u))
        isfinite(out[end]) || break
    end
    return out
end

show_trace(name, v) = @printf("%-26s %s\n", name,
    join((isfinite(x) ? @sprintf("%7.2f", x) : "    NaN") for x in v[1:min(end, 30)]))

println("<φ²> every t=50, T=0 (deterministic):")
show_trace("ETD1  dt=0.01 T=0", trace_etd(0.01f0, 0.0f0, (UInt32(3), UInt32(0))))
show_trace("SOSRA dt=0.01 T=0", trace_sosra(0.01f0, 0.0f0))

println("\n<φ²> every t=50, T=1, three different noise keys:")
for k in (3, 77, 12345)
    show_trace("ETD1  dt=0.01 key=$k", trace_etd(0.01f0, 1.0f0, (UInt32(k), UInt32(0))))
end
show_trace("SOSRA dt=0.01 T=1", trace_sosra(0.01f0, 1.0f0))
