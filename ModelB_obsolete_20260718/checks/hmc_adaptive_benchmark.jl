# Adaptive-SOSRA check on the HMC-matched Model-B problem.
# Usage: julia hmc_adaptive_benchmark.jl [outfile] [L] [Ncfg] [tend]

include(joinpath(@__DIR__, "hmc_equilibrium_benchmark.jl"))

function run_adaptive(u0, kappa, lambda, tend, tol, seed;
                      dt0=FT(0.001), dtmax=FT(0.02))
    CUDA.seed!(seed)
    prob = make_problem(u0, kappa, lambda, tend; with_z=true)
    CUDA.synchronize()
    sol = nothing
    wall = @elapsed begin
        sol = solve(
            prob, SOSRA(); dt=dt0, adaptive=true,
            abstol=FT(tol), reltol=FT(tol), dtmax,
            save_start=false, save_everystep=false, save_end=true,
            dense=false, maxiters=typemax(Int)
        )
        CUDA.synchronize()
    end
    final = Array(sol.u[end])
    return final, (;
        wall, retcode=string(sol.retcode), finite=all(isfinite, final),
        naccept=sol.stats.naccept, nreject=sol.stats.nreject,
        avgdt=Float64(tend) / max(sol.stats.naccept, 1)
    )
end

function adaptive_main(args=ARGS)
    outfile = length(args) >= 1 ? args[1] :
        joinpath(@__DIR__, "hmc_adaptive_results.tsv")
    L = length(args) >= 2 ? parse(Int, args[2]) : 16
    ncfg = length(args) >= 3 ? parse(Int, args[3]) : 2048
    tend = length(args) >= 4 ? parse(FT, args[4]) : FT(5)
    kappa, lambda = 0.2705, 0.022
    data = joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                    "cfgs_wolff_fahmc_k=0.2705_l=0.022_$(L)^2.jld2")
    hmc = jldopen(data, "r") do f
        Array(f["cfgs"])
    end
    indices = round.(Int, range(1, size(hmc, 3); length=ncfg))
    u0 = FT.(hmc[:, :, indices])
    initial_obs = observables(u0, kappa, lambda)
    tols = FT.((0.5, 0.2, 0.1, 0.05, 0.02))

    @printf("adaptive SOSRA HMC stationarity: L=%d N=%d tend=%.2f dtmax=.02 device=%s\n",
            L, ncfg, tend, CUDA.name(CUDA.device()))
    rows = NamedTuple[]
    for (i, tol) in enumerate(tols)
        # Same-shape preflight, excluded from timing.
        run_adaptive(u0, kappa, lambda, FT(0.02), tol, 7000 + i)
        final, meta = run_adaptive(u0, kappa, lambda, tend, tol, 8000 + i)
        final_obs = observables(final, kappa, lambda)
        massmax = maximum(abs.(final_obs[:mass] .- initial_obs[:mass]))
        summaries = paired_summary(initial_obs, final_obs)
        for s in summaries
            push!(rows, (; tol=Float64(tol), dtmax=0.02, tend=Float64(tend), ncfg,
                meta.wall, meta.avgdt, meta.naccept, meta.nreject, meta.retcode,
                meta.finite, massmax, s...))
        end
        a = only(filter(r -> r.observable == "action", summaries))
        @printf("tol=%5.3f avgdt=%.6f steps=%5d reject=%3d wall=%.3fs dS/S=%+.3e z=%+.1f mass=%.2e\n",
                tol, meta.avgdt, meta.naccept, meta.nreject, meta.wall,
                a.relative_delta, a.z, massmax)
    end

    open(outfile, "w") do io
        println(io, "tol\tdtmax\ttend\tncfg\twall\tavgdt\tnaccept\tnreject\tretcode\tfinite\tmassmax\tobservable\tinitial\tfinal\tdelta\tpaired_se\tz\trelative_delta")
        for r in rows
            println(io, join((getproperty(r, key) for key in propertynames(r)), '\t'))
        end
    end
    println("wrote ", abspath(outfile))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    adaptive_main()
end
