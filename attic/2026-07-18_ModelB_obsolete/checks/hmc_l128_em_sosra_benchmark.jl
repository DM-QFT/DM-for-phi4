# Dedicated L=128 work/precision benchmark for fixed-step EM versus SOSRA.
#
# It performs two complementary tests on the HMC-matched critical action:
#   1. batched HMC-stationarity accuracy with independent noise repetitions;
#   2. post-JIT single-trajectory throughput with repeated timings.
#
# Usage:
#   julia --startup-file=no -t 16 checks/hmc_l128_em_sosra_benchmark.jl \
#       PREFIX NCFG ACC_REPS ACC_TEND SPEED_TEND SPEED_REPS PROFILE
#
# PROFILE is full, core, or equal. Set SPEED_REPS=0 to skip throughput.
#
# Outputs:
#   <prefix>_accuracy.tsv
#   <prefix>_throughput.tsv

include(joinpath(@__DIR__, "hmc_equilibrium_benchmark.jl"))

const L128_KAPPA = 0.2705
const L128_LAMBDA = 0.022

function l128_specs(profile="full")
    specs = vcat(
        [(method="EM", alg=EM(), with_z=false, dt=FT(dt))
         for dt in (0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04)],
        [(method="SOSRA", alg=SOSRA(), with_z=true, dt=FT(dt))
         for dt in (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04)],
    )
    if profile == "full"
        return specs
    elseif profile == "core"
        return filter(specs) do s
            (s.method == "EM" && s.dt <= FT(0.01)) ||
            (s.method == "SOSRA" && s.dt in FT.((0.0025, 0.005, 0.01, 0.02)))
        end
    elseif profile == "equal"
        return vcat(
            [(method="EM", alg=EM(), with_z=false, dt=FT(dt))
             for dt in (0.0025, 0.005, 0.01, 0.0125)],
            [(method="SOSRA", alg=SOSRA(), with_z=true, dt=FT(dt))
             for dt in (0.004, 0.008, 0.016, 0.02)],
        )
    end
    error("profile must be full, core, or equal")
end

function l128_block_se(delta, nblocks)
    n = length(delta)
    nb = min(nblocks, n)
    blocksize = n ÷ nb
    means = [mean(@view delta[(i - 1) * blocksize + 1:i * blocksize])
             for i in 1:nb]
    return nb > 1 ? std(means) / sqrt(nb) : NaN
end

function l128_paired_summary(initial_obs, final_obs)
    rows = NamedTuple[]
    for name in sort!(collect(keys(initial_obs)))
        before = initial_obs[name]
        after = final_obs[name]
        delta = after .- before
        dmean = mean(delta)
        se32 = l128_block_se(delta, 32)
        se64 = l128_block_se(delta, 64)
        se128 = l128_block_se(delta, 128)
        base = mean(before)
        push!(rows, (;
            observable=String(name), initial=base, final=mean(after), delta=dmean,
            paired_se=se64, paired_se32=se32, paired_se128=se128,
            z=dmean / max(se64, eps()),
            relative_delta=dmean / max(abs(base), eps())
        ))
    end
    return rows
end

function load_l128_hmc(ncfg)
    data = joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                    "cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2")
    hmc = jldopen(data, "r") do f
        Int(f["N"]) == 128 || error("HMC lattice metadata mismatch")
        k = haskey(f, "kappa") ? f["kappa"] : f["\u03ba"]
        l = haskey(f, "lambda") ? f["lambda"] : f["\u03bb"]
        isapprox(Float64(k), L128_KAPPA; rtol=0, atol=1e-12) ||
            error("HMC kappa mismatch")
        isapprox(Float64(l), L128_LAMBDA; rtol=0, atol=1e-12) ||
            error("HMC lambda mismatch")
        Array(f["cfgs"])
    end
    ncfg <= size(hmc, 3) || error("requested N=$ncfg > HMC sample count")
    # Keep Markov-chain ordering while separating selected states by ~N_total/N.
    indices = round.(Int, range(1, size(hmc, 3); length=ncfg))
    return FT.(hmc[:, :, indices]), data, indices
end

function accuracy_test(u0, specs, tend, nreps, outfile)
    initial_obs = observables(u0, L128_KAPPA, L128_LAMBDA)
    ncfg = size(u0, 3)
    rows = NamedTuple[]

    @printf("\n# L=128 HMC-stationarity accuracy: N=%d reps=%d tend=%.3f\n",
            ncfg, nreps, tend)
    for (ispec, spec) in enumerate(specs)
        # Same-shape preflight removes lazy compilation from every measured run.
        run_once(u0, L128_KAPPA, L128_LAMBDA,
                 max(2spec.dt, FT(0.01)), spec.alg, spec.with_z,
                 spec.dt, 20000 + ispec)
        GC.gc(false)

        delta_sum = Dict(name => zeros(Float64, ncfg) for name in keys(initial_obs))
        replicate_means = Dict(name => Float64[] for name in keys(initial_obs))
        walls = Float64[]
        naccepts = Int[]
        nrejects = Int[]
        retcodes = String[]
        finite_all = true
        mass_max = 0.0

        for rep in 1:nreps
            seed = 30000 + 100ispec + rep
            final, meta = run_once(u0, L128_KAPPA, L128_LAMBDA, tend,
                                   spec.alg, spec.with_z, spec.dt, seed)
            push!(walls, meta.wall)
            push!(naccepts, meta.naccept)
            push!(nrejects, meta.nreject)
            push!(retcodes, meta.retcode)
            finite_all &= meta.finite && meta.retcode == "Success"
            if meta.finite
                final_obs = observables(final, L128_KAPPA, L128_LAMBDA)
                mass_max = max(mass_max,
                               maximum(abs.(final_obs[:mass] .- initial_obs[:mass])))
                for name in keys(initial_obs)
                    delta = final_obs[name] .- initial_obs[name]
                    delta_sum[name] .+= delta
                    push!(replicate_means[name], mean(delta))
                end
            end
            GC.gc(false)
            @printf("  %-5s dt=%7.5f rep=%d/%d wall=%7.3fs ret=%s\n",
                    spec.method, spec.dt, rep, nreps, meta.wall, meta.retcode)
            flush(stdout)
        end

        if finite_all
            averaged_obs = Dict(name => initial_obs[name] .+
                                delta_sum[name] ./ nreps for name in keys(initial_obs))
            summaries = l128_paired_summary(initial_obs, averaged_obs)
            for s in summaries
                rmeans = replicate_means[Symbol(s.observable)]
                replicate_se = length(rmeans) > 1 ? std(rmeans) / sqrt(length(rmeans)) : NaN
                push!(rows, (;
                    method=spec.method, dt=Float64(spec.dt), tend=Float64(tend),
                    ncfg, nreps, wall_median=median(walls), wall_q25=quantile(walls, 0.25),
                    wall_q75=quantile(walls, 0.75), wall_min=minimum(walls),
                    wall_max=maximum(walls), naccept=round(Int, median(naccepts)),
                    nreject=round(Int, median(nrejects)), retcode=join(unique(retcodes), ","),
                    finite=finite_all, mass_max, replicate_se, s...
                ))
            end
            a = only(filter(s -> s.observable == "action", summaries))
            p2 = only(filter(s -> s.observable == "phi2", summaries))
            sk = only(filter(s -> s.observable == "skmin", summaries))
            @printf("  => %-5s dt=%7.5f median=%7.3fs dS/S=%+.4f%% (z=%+.1f) dphi2=%+.4f%% Skmin=%+.3f%% mass=%.2e\n",
                    spec.method, spec.dt, median(walls), 100a.relative_delta, a.z,
                    100p2.relative_delta, 100sk.relative_delta, mass_max)
        else
            @warn "one or more accuracy repetitions failed" spec retcodes
        end
        CUDA.reclaim()
    end

    open(outfile, "w") do io
        println(io, "method\tdt\ttend\tncfg\tnreps\twall_median\twall_q25\twall_q75\twall_min\twall_max\tnaccept\tnreject\tretcode\tfinite\tmass_max\treplicate_se\tobservable\tinitial\tfinal\tdelta\tpaired_se\tpaired_se32\tpaired_se128\tz\trelative_delta")
        for r in rows
            println(io, join((getproperty(r, key) for key in propertynames(r)), '\t'))
        end
    end
    println("wrote ", abspath(outfile))
    return rows
end

function throughput_test(u0single, specs, tend, nreps, outfile)
    rows = NamedTuple[]
    @printf("\n# L=128 single-trajectory post-JIT throughput: reps=%d tend=%.3f\n",
            nreps, tend)
    for (ispec, spec) in enumerate(specs)
        run_once(u0single, L128_KAPPA, L128_LAMBDA,
                 max(2spec.dt, FT(0.01)), spec.alg, spec.with_z,
                 spec.dt, 40000 + ispec)
        GC.gc(false)

        walls = Float64[]
        masserrs = Float64[]
        phi2s = Float64[]
        naccepts = Int[]
        nrejects = Int[]
        failures = 0
        for rep in 1:nreps
            final, meta = run_once(u0single, L128_KAPPA, L128_LAMBDA, tend,
                                   spec.alg, spec.with_z, spec.dt,
                                   50000 + 100ispec + rep)
            if meta.finite && meta.retcode == "Success"
                push!(walls, meta.wall)
                push!(naccepts, meta.naccept)
                push!(nrejects, meta.nreject)
                push!(phi2s, mean(abs2, final))
                m0 = mean(u0single)
                push!(masserrs, maximum(abs.(mean(final; dims=(1, 2)) .- m0)))
            else
                failures += 1
            end
            GC.gc(false)
        end
        isempty(walls) && error("all throughput repetitions failed for $(spec.method) dt=$(spec.dt)")
        push!(rows, (;
            method=spec.method, dt=Float64(spec.dt), tend=Float64(tend), nreps,
            wall_median=median(walls), wall_q25=quantile(walls, 0.25),
            wall_q75=quantile(walls, 0.75), wall_min=minimum(walls), wall_max=maximum(walls),
            wall_per_t=median(walls) / tend,
            steps=round(Int, median(naccepts)), rejects=round(Int, median(nrejects)),
            failures, mass_max=maximum(masserrs), phi2_mean=mean(phi2s)
        ))
        @printf("  %-5s dt=%7.5f wall=%8.4fs wall/t=%9.6f steps=%6d fail=%d mass=%.2e\n",
                spec.method, spec.dt, median(walls), median(walls) / tend,
                round(Int, median(naccepts)), failures, maximum(masserrs))
        CUDA.reclaim()
    end

    open(outfile, "w") do io
        println(io, "method\tdt\ttend\tnreps\twall_median\twall_q25\twall_q75\twall_min\twall_max\twall_per_t\tsteps\trejects\tfailures\tmass_max\tphi2_mean")
        for r in rows
            println(io, join((getproperty(r, key) for key in propertynames(r)), '\t'))
        end
    end
    println("wrote ", abspath(outfile))
    return rows
end

function l128_main(args=ARGS)
    prefix = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "hmc_l128_em_sosra")
    ncfg = length(args) >= 2 ? parse(Int, args[2]) : 512
    accuracy_reps = length(args) >= 3 ? parse(Int, args[3]) : 3
    accuracy_tend = length(args) >= 4 ? parse(FT, args[4]) : FT(5)
    speed_tend = length(args) >= 5 ? parse(FT, args[5]) : FT(20)
    speed_reps = length(args) >= 6 ? parse(Int, args[6]) : 7
    profile = length(args) >= 7 ? args[7] : "full"

    u0, data, indices = load_l128_hmc(ncfg)
    specs = l128_specs(profile)
    @printf("L=128 EM/SOSRA benchmark on %s\n", CUDA.name(CUDA.device()))
    @printf("HMC=%s N=%d selected stride~%.2f first=%d last=%d\n",
            abspath(data), ncfg, 10240 / ncfg, first(indices), last(indices))
    @printf("action match: T=1 kappa=%.4f lambda=%.3f dx=gamma=1 Float32 evolution\n",
            L128_KAPPA, L128_LAMBDA)
    @printf("profile=%s methods=%d\n", profile, length(specs))

    accuracy_file = prefix * "_accuracy.tsv"
    throughput_file = prefix * "_throughput.tsv"
    accuracy_test(u0, specs, accuracy_tend, accuracy_reps, accuracy_file)
    if speed_reps > 0
        # One untouched HMC state; retain its magnetization.
        u0single = copy(@view u0[:, :, 1:1])
        throughput_test(u0single, specs, speed_tend, speed_reps, throughput_file)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    l128_main()
end
