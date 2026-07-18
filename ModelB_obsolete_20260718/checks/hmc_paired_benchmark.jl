# Common-Brownian-path strong work/precision benchmark matched to the 2Dphi4
# HMC action.  This complements hmc_equilibrium_benchmark.jl: HMC supplies
# equilibrium initial states, while a very fine SOSRA path supplies the dynamic
# reference (HMC itself contains no real-time dynamics).

include(joinpath(@__DIR__, "critical_paired_benchmark.jl"))
using JLD2

const HMC_KAPPA = 0.2705
const HMC_LAMBDA = 0.022
const HMC_M02 = 2 * (1 - 2HMC_LAMBDA) - 8HMC_KAPPA
const HMC_C3 = 4HMC_LAMBDA
const HMC_TWO_KAPPA = 2HMC_KAPPA

"""Model-B drift whose invariant action exactly matches the HMC files."""
function hmc_drift!(du, u, p, t)
    L = size(u, 1)
    @inbounds for y in 1:L, x in 1:L
        xm1, xp1 = prev(x, L), next(x, L)
        ym1, yp1 = prev(y, L), next(y, L)
        xm2, xp2 = prev2(x, L), next2(x, L)
        ym2, yp2 = prev2(y, L), next2(y, L)
        minus_bih = -20u[x, y] + 8u[x, ym1] - u[x, ym2] +
                    8u[x, yp1] - u[x, yp2] +
                    8u[xm1, y] - 2u[xm1, ym1] - 2u[xm1, yp1] - u[xm2, y] +
                    8u[xp1, y] - 2u[xp1, ym1] - 2u[xp1, yp1] - u[xp2, y]
        f(v) = HMC_M02 * v + HMC_C3 * v^3
        lap_uprime = f(u[xm1, y]) + f(u[xp1, y]) +
                     f(u[x, ym1]) + f(u[x, yp1]) - 4f(u[x, y])
        du[x, y] = HMC_TWO_KAPPA * minus_bih + lap_uprime
    end
    return nothing
end

function solve_hmc_grid(u0, alg, dt, times, Wvals, Zvals)
    noise = NoiseGrid(times, Wvals, Zvals)
    function diffusion!(du, u, p, t)
        fill!(du, sqrt(2.0))
        return nothing
    end
    prob = SDEProblem(hmc_drift!, diffusion!, copy(u0),
                      (times[1], times[end]); noise)
    elapsed = @elapsed sol = solve(
        prob, alg; dt, adaptive=false, save_start=false,
        save_everystep=false, save_end=true, dense=false
    )
    return Array(sol.u[end]), elapsed, sol.retcode
end

function hmc_action_density(u)
    L = size(u, 1)
    total = 0.0
    @inbounds for y in 1:L, x in 1:L
        v = u[x, y]
        nbr = u[next(x, L), y] + u[x, next(y, L)]
        total += -2HMC_KAPPA * v * nbr +
                 (1 - 2HMC_LAMBDA) * v^2 + HMC_LAMBDA * v^4
    end
    return total / length(u)
end

function hmc_error_metrics(u, ref, mean0)
    L = size(u, 1)
    err = u .- ref
    rmse = sqrt(mean(abs2, err))
    relrmse = rmse / sqrt(mean(abs2, ref))
    F = fft(err)
    q = [4sinpi((i - 1) / L)^2 + 4sinpi((j - 1) / L)^2
         for i in 1:L, j in 1:L]
    q[1, 1] = Inf
    hminus1 = sqrt(sum(abs2.(F) ./ q)) / L^2
    lowmode = sqrt((abs2(F[2, 1]) + abs2(F[1, 2])) / (2L^2))
    masserr = abs(mean(u) - mean0)
    p2rel = abs(mean(abs2, u) - mean(abs2, ref)) / mean(abs2, ref)
    erel = abs(hmc_action_density(u) - hmc_action_density(ref)) /
           max(abs(hmc_action_density(ref)), eps())
    skrel = abs(structure_min(u) - structure_min(ref)) /
            max(structure_min(ref), eps())
    return (; rmse, relrmse, hminus1, lowmode, masserr, p2rel, erel, skrel)
end

function run_hmc_path(path, u0, tend, href, specs)
    nsteps = round(Int, tend / href)
    isapprox(nsteps * href, tend; atol=100eps(tend)) ||
        error("tend/href must be integral")
    rng = Xoshiro(0x484d435f4d42 + path)
    dWfine, dZfine = fine_noise(size(u0, 1), nsteps, href, rng)

    tref, Wref, Zref = aggregate_grids(dWfine, dZfine, href, 1; need_z=true)
    reference, _, refcode = solve_hmc_grid(u0, SOSRA(), href, tref, Wref, Zref)
    refcode == ReturnCode.Success || error("reference solve failed: $refcode")

    rows = NamedTuple[]
    for (name, alg, need_z, dt) in specs
        m = round(Int, dt / href)
        isapprox(m * href, dt; atol=100eps(dt)) || continue
        nsteps % m == 0 || continue
        times, Wvals, Zvals = aggregate_grids(dWfine, dZfine, href, m; need_z)
        u, elapsed, retcode = solve_hmc_grid(u0, alg, dt, times, Wvals, Zvals)
        finite = all(isfinite, u)
        metrics = finite ? hmc_error_metrics(u, reference, mean(u0)) :
            (rmse=Inf, relrmse=Inf, hminus1=Inf, lowmode=Inf,
             masserr=Inf, p2rel=Inf, erel=Inf, skrel=Inf)
        push!(rows, (; path, method=name, dt, elapsed,
                     retcode=string(retcode), finite, metrics...))
    end

    times, Wvals, Zvals = aggregate_grids(dWfine, dZfine, href, 2; need_z=true)
    u, elapsed, retcode = solve_hmc_grid(u0, SOSRA(), 2href, times, Wvals, Zvals)
    metrics = hmc_error_metrics(u, reference, mean(u0))
    push!(rows, (; path, method="SOSRA-refcheck", dt=2href, elapsed,
                 retcode=string(retcode), finite=all(isfinite, u), metrics...))
    return rows
end

function hmc_main(args=ARGS)
    outfile = length(args) >= 1 ? args[1] :
        joinpath(@__DIR__, "hmc_paired_results.tsv")
    L = length(args) >= 2 ? parse(Int, args[2]) : 16
    tend = length(args) >= 3 ? parse(Float64, args[3]) : 2.0
    npaths = length(args) >= 4 ? parse(Int, args[4]) : 16
    href = length(args) >= 5 ? parse(Float64, args[5]) : 0.000625
    data = length(args) >= 6 ? args[6] : joinpath(
        @__DIR__, "..", "..", "2Dphi4", "trainingdata",
        "cfgs_wolff_fahmc_k=0.2705_l=0.022_$(L)^2.jld2"
    )
    specs = method_specs(href)
    if length(args) >= 8 && args[8] == "l128_equal_cost"
        specs = vcat(
            [("EM", EM(), false, dt) for dt in (0.0025, 0.005, 0.01, 0.0125)],
            [("SOSRA", SOSRA(), true, dt) for dt in (0.004, 0.008, 0.016, 0.02)],
        )
    end
    if length(args) >= 7
        requested = Set(split(args[7], ','))
        specs = filter(s -> s[1] in requested, specs)
        isempty(specs) && error("method filter selected no methods")
    end

    hmc = jldopen(data, "r") do f
        Array(f["cfgs"])
    end
    npaths <= size(hmc, 3) || error("not enough HMC configurations")
    sample_indices = round.(Int, range(1, size(hmc, 3); length=npaths))

    @printf("HMC-matched paired benchmark: L=%d kappa=%.4f lambda=%.3f T=1 tend=%.3f paths=%d href=%.7f threads=%d\n",
        L, HMC_KAPPA, HMC_LAMBDA, tend, npaths, href, Threads.nthreads())
    @printf("HMC initial states: %s (means retained)\n", abspath(data))

    # Method-specific compilation, excluded from path timings.
    wrng = Xoshiro(77)
    warmW, warmZ = fine_noise(4, 256, href, wrng)
    warmu = zeros(4, 4)
    for (_, alg, need_z, dt) in specs
        m = max(1, round(Int, dt / href))
        m > size(warmW, 3) && continue
        nusable = size(warmW, 3) - size(warmW, 3) % m
        t, W, Z = aggregate_grids(@view(warmW[:, :, 1:nusable]),
                                  @view(warmZ[:, :, 1:nusable]), href, m; need_z)
        try
            solve_hmc_grid(warmu, alg, m * href, t, W, Z)
        catch
        end
    end

    per_path = Vector{Vector{NamedTuple}}(undef, npaths)
    Threads.@threads :static for path in 1:npaths
        u0 = copy(@view hmc[:, :, sample_indices[path]])
        per_path[path] = run_hmc_path(path, u0, tend, href, specs)
        @printf("path %d/%d complete (m=%+.4f)\n", path, npaths, mean(u0))
    end
    rows = reduce(vcat, per_path)

    open(outfile, "w") do io
        println(io, "path\tmethod\tdt\telapsed\tretcode\tfinite\trmse\trelrmse\thminus1\tlowmode\tmasserr\tp2rel\terel\tskrel")
        for r in rows
            println(io, join((getproperty(r, k) for k in propertynames(r)), '\t'))
        end
    end
    summarize(rows)
    println("wrote ", abspath(outfile))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    hmc_main()
end
