using CUDA
using NPZ
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))
include(joinpath(@__DIR__, "..", "solver", "ModelB.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const INITIAL = joinpath(
    ROOT,
    "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip",
    "samples/hmc_L64_N512_seed2026080201",
    "hmc_ep10000_L64_t0_N512.npz",
)
const OUTPUT = joinpath(ROOT, "results", "gpu_solver_benchmark.tsv")
const LATTICE = 64
const ENSEMBLES = 512
const FINAL_TIME = 100.0f0
const SOSRA_DT = 0.00625f0
const KAPPA = 0.340340f0
const LAMBDA = 1.0f0
const GAMMA = 1.0f0
const TEMPERATURE = 1.0f0
const REPEATS = 1

function load_initial()
    data = npzread(INITIAL)
    initial = permutedims(data["phi_gen"], (3, 2, 1))
    @assert size(initial) == (LATTICE, LATTICE, ENSEMBLES)
    return CuArray{Float32}(initial)
end

function elapsed_seconds(run)
    CUDA.synchronize()
    start = time_ns()
    result = run()
    CUDA.synchronize()
    return (time_ns() - start) / 1.0e9, result
end

function benchmark_fixed(initial)
    state = similar(initial)
    copyto!(state, initial)
    buffers = modelb_fixed_buffers(state; seed=UInt64(1))
    modelB_2d_fixed!(
        state, buffers;
        γ=GAMMA, T=TEMPERATURE, κ=KAPPA, λ=LAMBDA, dt=SOSRA_DT,
        tspan=(0.0f0, SOSRA_DT),
    )
    CUDA.synchronize()

    times = Float64[]
    for repeat in 1:REPEATS
        seconds, _ = elapsed_seconds() do
            copyto!(state, initial)
            seed_modelb_fixed!(buffers, UInt64(10_000 + repeat))
            modelB_2d_fixed!(
                state, buffers;
                γ=GAMMA, T=TEMPERATURE, κ=KAPPA, λ=LAMBDA, dt=SOSRA_DT,
                tspan=(0.0f0, FINAL_TIME),
            )
        end
        push!(times, seconds)
        println("fixed SOSRA repeat $repeat  $seconds s")
        flush(stdout)
    end
    @assert all(isfinite, state)
    return times
end

function run_sciml(initial, alg, dt, final_time, seed)
    solution = modelB_2d_SDE_prob(
        initial;
        γ=GAMMA, T=TEMPERATURE, κ=KAPPA, λ=LAMBDA, dt,
        tspan=(0.0f0, final_time), saveat=final_time,
        seed=UInt64(seed), alg,
    )
    return solution.u[end]
end

function benchmark_sciml(initial, alg, dt, name)
    warm = run_sciml(initial, alg, dt, dt, 20_000)
    CUDA.synchronize()
    @assert all(isfinite, warm)

    times = Float64[]
    result = warm
    for repeat in 1:REPEATS
        seconds, result = elapsed_seconds() do
            run_sciml(initial, alg, dt, FINAL_TIME, 20_000 + repeat)
        end
        push!(times, seconds)
        println("$name repeat $repeat  $seconds s")
        flush(stdout)
    end
    @assert all(isfinite, result)
    return times
end

function write_results(rows)
    mkpath(dirname(OUTPUT))
    open(OUTPUT, "w") do handle
        println(
            handle,
            "method\tbackend\tprecision\tL\ttrajectories\tt_final\tdt\t" *
            "integrator_steps\tdrift_nfe\ttransitions\tnetwork_nfe\tcompile_mode\t" *
            "seconds",
        )
        for (
            method, backend, precision, dt, steps, drift_nfe,
            transitions, network_nfe, mode, times,
        ) in rows
            println(
                handle,
                join((
                    method, backend, precision, LATTICE, ENSEMBLES, FINAL_TIME,
                    dt, steps, drift_nfe, transitions, network_nfe, mode,
                    only(times),
                ), '\t'),
            )
        end
    end
end

initial = load_initial()
fixed = benchmark_fixed(initial)
sciml_sosra = benchmark_sciml(initial, SOSRA(), SOSRA_DT, "SciML SOSRA")
sosra_steps = round(Int, FINAL_TIME / SOSRA_DT)
write_results((
    (
        "SOSRA (This work)", "Julia CUDA.jl", "Float32",
        SOSRA_DT, sosra_steps, 3 * sosra_steps, 1, 0, "custom kernels", fixed,
    ),
    (
        "SOSRA (StochasticDiffEq.jl)", "Julia StochasticDiffEq.jl", "Float32",
        SOSRA_DT, sosra_steps, 3 * sosra_steps, 1, 0, "none", sciml_sosra,
    ),
))
println("saved $OUTPUT")
