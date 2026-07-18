using CUDA
using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "solver_kernel_variants.jl"))
include(joinpath(@__DIR__, "rng_variants.jl"))
include(joinpath(@__DIR__, "fused_fixed_sosra.jl"))

using .FusedFixedSOSRA128
using .H100RNGVariants

const HMC_KAPPA = 0.2705
const HMC_LAMBDA = 0.022
const HMC_DT = 0.002f0
const HMC_TEND = 5.0f0
const HMC_NCFG = 512
const HMC_REPETITIONS = 2
const HMC_RSWM1 = DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1)

function load_hmc(ncfg=HMC_NCFG)
    path = normpath(joinpath(
        @__DIR__, "..", "..", "2Dphi4", "trainingdata",
        "cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2",
    ))
    configurations = jldopen(path, "r") do file
        Int(file["N"]) == 128 || error("HMC lattice-size mismatch")
        kappa = haskey(file, "kappa") ? file["kappa"] : file["κ"]
        lambda = haskey(file, "lambda") ? file["lambda"] : file["λ"]
        isapprox(Float64(kappa), HMC_KAPPA; rtol=0, atol=1e-12) ||
            error("HMC kappa mismatch")
        isapprox(Float64(lambda), HMC_LAMBDA; rtol=0, atol=1e-12) ||
            error("HMC lambda mismatch")
        Array(file["cfgs"])
    end
    ncfg <= size(configurations, 3) || error("too few HMC configurations")
    indices = round.(Int, range(1, size(configurations, 3); length=ncfg))
    # Preserve every configuration's magnetization.  Model B samples fixed
    # magnetization sectors; subtracting means would no longer be the HMC
    # canonical mixture used as the stationarity ground truth.
    selected = Float32.(configurations[:, :, indices])
    return selected, path, indices
end

function local_observables(field)
    L, L2, batch = size(field)
    L == L2 || error("square lattice required")
    volume = L^2
    action = zeros(Float64, batch)
    phi2 = zeros(Float64, batch)
    phi4 = zeros(Float64, batch)
    nearest = zeros(Float64, batch)
    mass = zeros(Float64, batch)

    Threads.@threads for ensemble in 1:batch
        action_sum = 0.0
        phi2_sum = 0.0
        phi4_sum = 0.0
        nearest_sum = 0.0
        mass_sum = 0.0
        @inbounds for y in 1:L, x in 1:L
            xp1 = x == L ? 1 : x + 1
            yp1 = y == L ? 1 : y + 1
            value = Float64(field[x, y, ensemble])
            value2 = value * value
            value4 = value2 * value2
            forward = Float64(field[xp1, y, ensemble]) +
                      Float64(field[x, yp1, ensemble])
            action_sum += -2HMC_KAPPA * value * forward +
                          (1 - 2HMC_LAMBDA) * value2 +
                          HMC_LAMBDA * value4
            nearest_sum += 0.5 * value * forward
            phi2_sum += value2
            phi4_sum += value4
            mass_sum += value
        end
        action[ensemble] = action_sum / volume
        phi2[ensemble] = phi2_sum / volume
        phi4[ensemble] = phi4_sum / volume
        nearest[ensemble] = nearest_sum / volume
        mass[ensemble] = mass_sum / volume
    end
    return (; action, phi2, phi4, nearest, mass)
end

function block_standard_error(values; blocks=64)
    nblocks = min(blocks, length(values))
    blocksize = length(values) ÷ nblocks
    blockmeans = [
        mean(@view values[(block - 1) * blocksize + 1:block * blocksize])
        for block in 1:nblocks
    ]
    return std(blockmeans) / sqrt(nblocks)
end

function custom_run(u0, seed)
    u = CuArray(u0)
    workspace = allocate_workspace(u; rng_kind=:library_philox, seed)
    params = ModelBParameters()
    CUDA.synchronize()
    start_ns = time_ns()
    solve_fixed_sosra!(
        u, workspace, params;
        dt=HMC_DT, t_end=HMC_TEND,
        threads=1024, blocks=264,
        noise_threads=512, noise_blocks=528,
    )
    CUDA.synchronize()
    seconds = (time_ns() - start_ns) / 1e9
    return Array(u), seconds
end

function sciml_run(u0, seed)
    u0_gpu = CuArray(u0)
    builder = (t, u) -> packed_noise_process128(
        t, u;
        rng_kind=:library_philox,
        seed,
        synchronize_callbacks=false,
        rswm=HMC_RSWM1,
    )
    start_ns = time_ns()
    solution = solve_sosra(
        u0_gpu;
        drift_launch! = drift_linear128_best!,
        noise_builder=builder,
        synchronize_callbacks=false,
        dt=HMC_DT,
        t_end=HMC_TEND,
    )
    CUDA.synchronize()
    seconds = (time_ns() - start_ns) / 1e9
    return Array(solution.u[end]), seconds
end

function summarize_method(name, run!, u0, initial, repetitions)
    accumulated = Dict(
        observable => zeros(Float64, size(u0, 3))
        for observable in (:action, :phi2, :phi4, :nearest, :mass)
    )
    times = Float64[]
    max_mass_drift = 0.0
    for repetition in 1:repetitions
        final, seconds = run!(u0, 41000 + repetition)
        all(isfinite, final) || error("$name produced a non-finite field")
        final_obs = local_observables(final)
        for observable in keys(accumulated)
            accumulated[observable] .+= getproperty(final_obs, observable)
        end
        max_mass_drift = max(
            max_mass_drift,
            maximum(abs.(final_obs.mass .- initial.mass)),
        )
        push!(times, seconds)
        @printf(
            "%s repetition=%d seconds=%.6f max_mass_density_drift=%.9g\n",
            name, repetition, seconds,
            maximum(abs.(final_obs.mass .- initial.mass)),
        )
    end

    rows = NamedTuple[]
    for observable in (:action, :phi2, :phi4, :nearest, :mass)
        before = getproperty(initial, observable)
        after = accumulated[observable] ./ repetitions
        delta = after .- before
        initial_mean = mean(before)
        delta_mean = mean(delta)
        paired_se = block_standard_error(delta)
        push!(rows, (;
            method=name,
            observable=String(observable),
            initial=initial_mean,
            final=mean(after),
            delta=delta_mean,
            paired_se,
            z=delta_mean / paired_se,
            relative_delta=delta_mean / max(abs(initial_mean), eps()),
            wall_median=median(times),
            max_mass_density_drift=max_mass_drift,
        ))
    end
    return rows
end

function main()
    ncfg = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : HMC_NCFG
    repetitions = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : HMC_REPETITIONS
    outfile = length(ARGS) >= 3 ? ARGS[3] :
        joinpath(@__DIR__, "fused_vs_hmc.tsv")

    u0, path, indices = load_hmc(ncfg)
    initial = local_observables(u0)
    @printf(
        "HMC source=%s ncfg=%d first=%d last=%d initial_action=%.9g\n",
        path, ncfg, first(indices), last(indices), mean(initial.action),
    )

    # Same-shape short preflights remove compilation from reported full runs.
    custom_run(@view(u0[:, :, 1:2]), 40001)
    sciml_run(@view(u0[:, :, 1:2]), 40002)

    rows = vcat(
        summarize_method("custom_fused_philox", custom_run, u0, initial, repetitions),
        summarize_method("sciml_packed_philox", sciml_run, u0, initial, repetitions),
    )

    open(outfile, "w") do io
        println(io, "method\tobservable\tinitial\tfinal\tdelta\tpaired_se\tz\trelative_delta\twall_median\tmax_mass_density_drift")
        for row in rows
            println(io, join((getproperty(row, key) for key in propertynames(row)), '\t'))
        end
    end
    for row in rows
        if row.observable in ("action", "phi2", "phi4")
            @printf(
                "%s %s relative_delta=%+.6f%% paired_z=%+.3f\n",
                row.method, row.observable, 100row.relative_delta, row.z,
            )
        end
    end
    println("wrote ", abspath(outfile))
end

main()
