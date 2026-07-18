using CUDA, Printf, Statistics

include(joinpath(@__DIR__, "..", "ModelB_fixed_sciml.jl"))

CUDA.seed!(20260718)
u0 = CUDA.randn(Float32, 128, 128, 1024)
u0 .-= mean(u0; dims=(1, 2))
prob = modelB_2d_fixed_problem(u0; tspan=(0.0f0, 2.0f0))
buffers = modelb_fixed_buffers(prob.u0; seed=UInt64(1))
alg = ModelBFixedSOSRA(; buffers)

function direct_solve(prob, buffers, seed)
    u = copy(prob.u0)
    CUDA.synchronize()
    seed_modelb_fixed!(buffers, seed)
    modelB_2d_fixed!(
        u, buffers;
        γ=prob.p.γ, T=prob.p.T, κ=prob.p.κ, λ=prob.p.λ,
        tspan=prob.tspan, dt=0.002f0,
    )
    CUDA.synchronize()
    return u
end

function elapsed_seconds(f)
    CUDA.synchronize()
    start = time_ns()
    result = f()
    CUDA.synchronize()
    return (time_ns() - start) / 1.0e9, result
end

# Compile and warm the complete 1000-step path before measuring it.
reference = direct_solve(prob, buffers, UInt64(314159))
warmup = solve(prob, alg; dt=0.002f0, seed=UInt64(314159))
@assert reference == warmup.u[end]

direct_times = Float64[]
sciml_times = Float64[]
paired_ratios = Float64[]
println("repetition\tdirect_s\tsciml_s\tratio\texact")
for repetition in 1:16
    seed = UInt64(400000 + repetition)
    if isodd(repetition)
        direct_time, direct = elapsed_seconds(
            () -> direct_solve(prob, buffers, seed),
        )
        sciml_time, sol = elapsed_seconds(
            () -> solve(prob, alg; dt=0.002f0, seed),
        )
    else
        sciml_time, sol = elapsed_seconds(
            () -> solve(prob, alg; dt=0.002f0, seed),
        )
        direct_time, direct = elapsed_seconds(
            () -> direct_solve(prob, buffers, seed),
        )
    end
    exact = direct == sol.u[end]
    push!(direct_times, direct_time)
    push!(sciml_times, sciml_time)
    push!(paired_ratios, sciml_time / direct_time)
    @printf(
        "%d\t%.9f\t%.9f\t%.6f\t%s\n",
        repetition, direct_time, sciml_time, sciml_time / direct_time,
        string(exact),
    )
end

@printf(
    "median\t%.9f\t%.9f\t%.6f\n",
    median(direct_times), median(sciml_times),
    median(paired_ratios),
)
@printf(
    "paired_ratio_iqr\t%.6f\t%.6f\n",
    quantile(paired_ratios, 0.25), quantile(paired_ratios, 0.75),
)

# Also expose the allocation-owning public default.  The first call includes
# compilation/allocation; subsequent calls show its hot wall-clock cost.
cold_time, _ = elapsed_seconds(
    () -> solve(prob, ModelBFixedSOSRA(); dt=0.002f0, seed=UInt64(500000)),
)
local_times = Float64[]
for repetition in 1:5
    local_time, _ = elapsed_seconds(
        () -> solve(
            prob, ModelBFixedSOSRA();
            dt=0.002f0, seed=UInt64(500000 + repetition),
        ),
    )
    push!(local_times, local_time)
end
@printf(
    "local_buffers\tfirst=%.9f\thot_median=%.9f\n",
    cold_time, median(local_times),
)
