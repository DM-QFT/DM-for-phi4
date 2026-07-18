using CUDA
using DiffEqNoiseProcess
using Random
using Statistics
using StochasticDiffEq

const EXPERIMENT_DIR = @__DIR__
const MODEL_B_PATH = normpath(joinpath(EXPERIMENT_DIR, "..", "ModelB.jl"))
const EXPECTED_MODEL_B_SHA256 =
    "5ade8ac779ab330e483e050e1246cc513a41e57942824a9c7893b3f9328f35d3"

include(MODEL_B_PATH)

const L_DEFAULT = 128
const ENSEMBLES_DEFAULT = 1024
const DT_DEFAULT = 0.002f0
const TEND_DEFAULT = 2.0f0

function make_u0(; L=L_DEFAULT, ensembles=ENSEMBLES_DEFAULT, seed=20260718)
    CUDA.seed!(seed)
    u0 = CUDA.randn(Float32, L, L, ensembles)
    u0 .-= mean(u0; dims=(1, 2))
    return u0
end

function make_modelb_problem(
    u0;
    drift_launch! = launch_modelb_drift!,
    noise_builder = modelb_noise_process,
    synchronize_callbacks = true,
    t_end = TEND_DEFAULT,
)
    u = copy(u0)
    two_kappa = 2.0f0 * 0.2705f0
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * 0.022f0) - 8.0f0 * 0.2705f0
    c3 = 4.0f0 * 0.022f0
    workspace = (similar(u), similar(u), similar(u))

    function drift!(du, u, p, t)
        drift_launch!(du, u, two_kappa, m02, c3, 1.0f0)
        synchronize_callbacks && CUDA.synchronize()
        return nothing
    end

    function diffusion!(du, u, p, t)
        fill!(du, sqrt(2.0f0))
        return nothing
    end

    noise = noise_builder(0.0f0, u)
    problem = SDEProblem(
        drift!, diffusion!, u, (0.0f0, Float32(t_end)), workspace; noise=noise
    )
    return problem
end

function solve_sosra(
    u0;
    drift_launch! = launch_modelb_drift!,
    noise_builder = modelb_noise_process,
    synchronize_callbacks = true,
    dt = DT_DEFAULT,
    t_end = TEND_DEFAULT,
)
    problem = make_modelb_problem(
        u0;
        drift_launch!, noise_builder, synchronize_callbacks, t_end,
    )
    return solve(
        problem, SOSRA();
        adaptive=false,
        dt=Float32(dt),
        saveat=Float32(t_end),
        save_everystep=false,
    )
end

function synchronized_seconds(f)
    CUDA.synchronize()
    elapsed = @elapsed begin
        value = f()
        CUDA.synchronize()
    end
    return elapsed, value
end

function kernel_median_seconds(f; warmup=3, samples=20)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    times = Float64[]
    for _ in 1:samples
        elapsed, _ = synchronized_seconds(f)
        push!(times, elapsed)
    end
    return median(times), minimum(times), maximum(times)
end

max_abs_difference(a, b) = maximum(abs.(a .- b))

