using CUDA, Printf, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed_sciml.jl"))

const κ = 0.28f0
const λ = 0.022f0
const γ = 1.0f0
const T = 1.0f0
const dt = 0.002f0
const seed = UInt64(20260720)

function wall_seconds(f)
    CUDA.synchronize()
    t0 = time_ns()
    result = f()
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9, result
end

function benchmark_case(L, Nensemble; steps=1000, repetitions=5)
    CUDA.seed!(UInt64(1000L + Nensemble))
    u0 = CUDA.randn(Float32, L, L, Nensemble)
    prob = modelB_2d_fixed_problem(
        u0; γ, T, κ, λ, tspan=(0.0f0, steps * dt),
    )
    buffers = modelb_fixed_buffers(prob.u0; seed)
    alg = ModelBFixedSOSRA(; buffers)

    warm_prob = remake(prob; tspan=(0.0f0, dt))
    solve(warm_prob, alg; dt, seed)

    times = Float64[]
    sol = solve(warm_prob, alg; dt, seed)
    for repetition in 1:repetitions
        seconds, sol = wall_seconds(
            () -> solve(prob, alg; dt, seed=seed + UInt64(repetition)),
        )
        push!(times, seconds)
    end
    @assert all(isfinite, sol.u[end])

    seconds = median(times)
    n = length(u0)
    @printf(
        "%4d\t%5d\t%10d\t%.6f\t%.3f\t%.4f\n",
        L, Nensemble, n, 1.0e3 * seconds / steps,
        1.0e9 * seconds / (steps * n), seconds,
    )
    return seconds / steps
end

CUDA.seed!(20260719)
cold_u0 = CUDA.randn(Float32, 256, 256, 128)
cold_prob = modelB_2d_fixed_problem(
    cold_u0; γ, T, κ, λ, tspan=(0.0f0, dt),
)
cold_seconds, cold_sol = wall_seconds(
    () -> solve(cold_prob, ModelBFixedSOSRA(); dt, seed),
)
@assert all(isfinite, cold_sol.u[end])
@printf("first_one_step_s\t%.6f\n", cold_seconds)

println("equal_element_count")
println("L\tNensemble\telements\tms_per_step\tns_per_element_step\tsolve_s")
for (L, Nensemble) in ((64, 2048), (128, 512), (256, 128))
    benchmark_case(L, Nensemble)
end

println("L256_scaling")
println("L\tNensemble\telements\tms_per_step\tns_per_element_step\tsolve_s")
scaling_seconds = [
    benchmark_case(256, Nensemble; repetitions=3)
    for Nensemble in (8, 16, 32, 64, 128)
]
active_step_seconds = scaling_seconds[end]
@printf(
    "L256_N128_predicted_t100_s\t%.3f\n",
    active_step_seconds * round(Int, 100.0f0 / dt),
)
