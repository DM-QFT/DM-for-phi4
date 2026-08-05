# Fixed-step GPU throughput/stability benchmark near the Model-B critical point.
# Accuracy is measured separately by critical_paired_benchmark.jl; this script
# deliberately times only solve(), after method-specific JIT warmup.
#
# Usage:
#   julia critical_gpu_benchmark.jl [outfile] [L] [T] [tend] [nseeds]

include(joinpath(@__DIR__, "modelB_runner.jl"))
using Random
using Printf
using Statistics

const ROCK_RHO = 256.0f0
const ROCK_EIGEN_EST = integ -> (integ.eigen_est = ROCK_RHO)

function make_gpu_problem(u0, temperature; with_z = false)
    u = CuArray(u0)
    function potential_prime_gpu(x)
        -x + x^3 / 6
    end
    function drift_gpu!(du, state, p, t)
        CUDA.@sync modelB_2d_loop_GPU(du, state, potential_prime_gpu, 1.0f0)
        return nothing
    end
    function diffusion_gpu!(du, state, p, t)
        du .= sqrt(2temperature)
        return nothing
    end
    params = (similar(u), similar(u), similar(u))
    W0 = zero(u)
    Z0 = with_z ? zero(u) : nothing
    noise = RealModelBProcess!(0.0f0, W0, Z0; save_everystep = false)
    return SDEProblem(drift_gpu!, diffusion_gpu!, u, (0.0f0, 1.0f0), params; noise)
end

function specs()
    return vcat(
        [("EM", EM(), false, dt) for dt in (0.0025f0, 0.005f0, 0.01f0, 0.02f0)],
        [("SOSRA", SOSRA(), true, dt) for dt in (0.005f0, 0.01f0, 0.02f0, 0.04f0)],
        [("SOSRA2", SOSRA2(), true, dt) for dt in (0.01f0, 0.02f0)],
        [("SROCK2", SROCK2(eigen_est = ROCK_EIGEN_EST), false, dt)
         for dt in (0.005f0, 0.01f0, 0.02f0, 0.04f0)],
        [("LambaEM", LambaEM(), false, 0.005f0)],
        [("DRI1NM-invalid", DRI1NM(), false, 0.005f0)],
    )
end

function run_once(u0, temperature, tend, alg, with_z, dt, seed)
    CUDA.seed!(seed)
    baseprob = make_gpu_problem(u0, temperature; with_z)
    prob = remake(baseprob; tspan = (0.0f0, Float32(tend)))
    CUDA.synchronize()
    sol = nothing
    wall = @elapsed begin
        sol = solve(
            prob, alg; dt, adaptive = false, save_start = false,
            save_everystep = false, save_end = true, dense = false,
            maxiters = typemax(Int)
        )
        CUDA.synchronize()
    end
    final = Array(sol.u[end])
    return (;
        wall,
        retcode = string(sol.retcode),
        finite = all(isfinite, final),
        masserr = abs(mean(final) - mean(u0)),
        phi2 = mean(abs2, final),
        naccept = sol.stats.naccept,
        nreject = sol.stats.nreject,
    )
end

function warmup!(temperature, allspecs)
    rng = Xoshiro(991)
    u0 = randn(rng, Float32, 8, 8)
    u0 .-= mean(u0)
    for (_, alg, with_z, dt) in allspecs
        try
            run_once(u0, temperature, max(2dt, 0.01f0), alg, with_z, dt, 1234)
            run_once(u0, temperature, max(2dt, 0.01f0), alg, with_z, dt, 1235)
        catch err
            @warn "warmup failed" alg dt exception = (err, catch_backtrace())
        end
    end
    CUDA.reclaim()
    return nothing
end

function summarize(rows, L, tend)
    keys_sorted = sort(unique((r.method, r.dt) for r in rows); by = x -> (x[1], x[2]))
    println("\n# GH200 fixed-step summary (post-JIT median across seeds)")
    @printf("%-18s %9s %10s %12s %12s %9s %9s\n",
        "method", "dt", "wall_s", "wall/tphys", "mass_err_max", "failures", "steps")
    for key in keys_sorted
        rs = filter(r -> (r.method, r.dt) == key, rows)
        good = filter(r -> r.finite && r.retcode == "Success", rs)
        wall = isempty(good) ? Inf : median(getproperty.(good, :wall))
        massmax = isempty(good) ? Inf : maximum(getproperty.(good, :masserr))
        failures = length(rs) - length(good)
        steps = isempty(good) ? 0 : round(Int, median(getproperty.(good, :naccept)))
        @printf("%-18s %9.5f %10.4f %12.5f %12.4e %4d/%-4d %9d\n",
            key[1], key[2], wall, wall / tend, massmax, failures, length(rs), steps)
    end
    @printf("lattice=%dx%d, physical horizon=%.3f\n", L, L, tend)
end

function main(args = ARGS)
    outfile = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "critical_gpu_results.tsv")
    L = length(args) >= 2 ? parse(Int, args[2]) : 256
    temperature = length(args) >= 3 ? parse(Float32, args[3]) : 4.45f0
    tend = length(args) >= 4 ? parse(Float32, args[4]) : 10.0f0
    nseeds = length(args) >= 5 ? parse(Int, args[5]) : 3
    allspecs = specs()

    @printf("GPU benchmark: L=%d T=%.4f tend=%.2f seeds=%d device=%s\n",
        L, temperature, tend, nseeds, CUDA.name(CUDA.device()))
    warmup!(temperature, allspecs)

    rows = NamedTuple[]
    for seed_index in 1:nseeds
        seed = 7200 + seed_index
        rng = Xoshiro(seed)
        u0 = randn(rng, Float32, L, L)
        u0 .-= mean(u0)
        for (name, alg, with_z, dt) in allspecs
            result = try
                run_once(u0, temperature, tend, alg, with_z, dt, seed)
            catch err
                @warn "solver run failed" name dt seed exception = (err, catch_backtrace())
                (wall = Inf, retcode = "Exception", finite = false, masserr = Inf,
                 phi2 = NaN, naccept = 0, nreject = 0)
            end
            push!(rows, (; seed, method = name, dt = Float64(dt), result...))
            @printf("seed=%d %-16s dt=%.5f wall=%.3fs ret=%s finite=%s mass=%.2e\n",
                seed, name, dt, result.wall, result.retcode, result.finite, result.masserr)
            flush(stdout)
        end
    end

    open(outfile, "w") do io
        println(io, "seed\tmethod\tdt\twall\tretcode\tfinite\tmasserr\tphi2\tnaccept\tnreject")
        for r in rows
            println(io, join((getproperty(r, k) for k in propertynames(r)), '\t'))
        end
    end
    summarize(rows, L, tend)
    println("wrote ", abspath(outfile))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

