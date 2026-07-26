using CUDA, Statistics
include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed_sciml.jl"))

L, Nensemble = 256, 128
κ, λ, γ, T = 0.28f0, 0.022f0, 1.0f0, 1.0f0
dt, t = 0.002f0, 100.0f0
initial_seed, noise_seed = UInt64(20260719), UInt64(20260720)

CUDA.seed!(initial_seed)
u0 = CUDA.randn(Float32, L, L, Nensemble)
prob = modelB_2d_fixed_problem(u0; γ, T, κ, λ, tspan=(0.0f0, t))
seconds = @elapsed sol = solve(
    prob, ModelBFixedSOSRA(); dt, save_start=true, seed=noise_seed,
)
@assert all(isfinite, sol.u[end])
drift = maximum(abs, mean(sol.u[end]; dims=(1, 2)) - mean(sol.u[1]; dims=(1, 2)))
println("times=$(sol.t), time=$(round(seconds; digits=3)) s, max|ΔM|=$drift")
