using CUDA, JLD2, LinearAlgebra, NPZ, Printf, Random, Statistics

module RealspaceStats
include(joinpath(
    @__DIR__,
    "..",
    "realspace_distribution",
    "realspace_methods.jl",
))
end

include(joinpath(@__DIR__, "fused_philox.jl"))

function record_stats!(stats, index, u, gpu_obs, p)
    RealspaceStats.realspace_observables!(gpu_obs, u, p)
    @views stats[index, :, :] .= Float32.(Array(gpu_obs))
    return nothing
end

function run_tiled(u0, times, dt, seed, p)
    u = copy(u0)
    buffers = fused_philox_buffers(u; seed)
    gpu_obs = CUDA.zeros(Float64, 6, size(u, 3))
    stats = Array{Float32}(undef, length(times), 6, size(u, 3))
    steps = round(Int, (times[2] - times[1]) / dt)
    seconds = 0.0
    record_stats!(stats, 1, u, gpu_obs, p)
    for index in 2:length(times)
        CUDA.synchronize()
        seconds += @elapsed begin
            tiled_philox_steps!(
                u,
                buffers,
                steps,
                dt,
                p.two_kappa,
                p.m02,
                p.c3,
                p.inv_gamma,
                p.noise_amplitude,
            )
            CUDA.synchronize()
        end
        record_stats!(stats, index, u, gpu_obs, p)
    end
    M0 = @view stats[1, 1, :]
    M_drift = @views stats[:, 1, :] .- reshape(M0, 1, :)
    @assert all(isfinite, stats)
    return (
        stats=stats,
        seconds=seconds,
        max_M_drift=maximum(abs, M_drift),
    )
end

make_gpuarrays_buffers(u, seed) = gpuarrays_buffers(u; seed)
make_direct_buffers(u, seed) = modelb_fixed_buffers(u; seed)

function run_buffered(
    u0, times, dt, seed, p, make_buffers, step!,
)
    u = copy(u0)
    buffers = make_buffers(u, seed)
    gpu_obs = CUDA.zeros(Float64, 6, size(u, 3))
    stats = Array{Float32}(undef, length(times), 6, size(u, 3))
    steps = round(Int, (times[2] - times[1]) / dt)
    seconds = 0.0
    record_stats!(stats, 1, u, gpu_obs, p)
    for index in 2:length(times)
        CUDA.synchronize()
        seconds += @elapsed begin
            for _ in 1:steps
                step!(
                    u,
                    buffers,
                    dt,
                    p.two_kappa,
                    p.m02,
                    p.c3,
                    p.inv_gamma,
                    p.noise_amplitude,
                )
            end
            CUDA.synchronize()
        end
        record_stats!(stats, index, u, gpu_obs, p)
    end
    M0 = @view stats[1, 1, :]
    M_drift = @views stats[:, 1, :] .- reshape(M0, 1, :)
    @assert all(isfinite, stats)
    return (
        stats=stats,
        seconds=seconds,
        max_M_drift=maximum(abs, M_drift),
    )
end

function distribution_metrics(x, reference)
    x64 = Float64.(x)
    reference64 = Float64.(reference)
    scale = std(reference64)
    sort!(x64)
    sort!(reference64)
    return (
        w1=mean(abs, x64 .- reference64) / scale,
        ks=ks_distance_sorted(x64, reference64),
    )
end

function ks_distance_sorted(x, y)
    nx, ny = length(x), length(y)
    i = 1
    j = 1
    distance = 0.0
    while i <= nx || j <= ny
        value =
            j > ny || (i <= nx && x[i] <= y[j]) ? x[i] : y[j]
        while i <= nx && x[i] <= value
            i += 1
        end
        while j <= ny && y[j] <= value
            j += 1
        end
        distance = max(
            distance,
            abs((i - 1) / nx - (j - 1) / ny),
        )
    end
    return distance
end

function sliced_w1(candidate, reference, baseline, directions)
    reference64 = Float64.(reference)
    candidate64 = Float64.(candidate)
    baseline64 = Float64.(baseline)
    μ = mean(reference64; dims=2)
    σ = std(reference64; dims=2)
    reference64 .= (reference64 .- μ) ./ σ
    candidate64 .= (candidate64 .- μ) ./ σ
    baseline64 .= (baseline64 .- μ) ./ σ
    distance = 0.0
    sampling_floor = 0.0
    for projection in axes(directions, 2)
        direction = @view directions[:, projection]
        projected_reference =
            vec(transpose(direction) * reference64)
        projected_candidate =
            vec(transpose(direction) * candidate64)
        projected_baseline =
            vec(transpose(direction) * baseline64)
        scale = std(projected_reference)
        sort!(projected_reference)
        sort!(projected_candidate)
        sort!(projected_baseline)
        distance += mean(
            abs,
            projected_candidate .- projected_reference,
        ) / scale
        sampling_floor += mean(
            abs,
            projected_baseline .- projected_reference,
        ) / scale
    end
    return (
        distance=distance / size(directions, 2),
        sampling_floor=sampling_floor / size(directions, 2),
    )
end

function summarize(
    initial, name, candidate, reference, baseline, times, directions,
    seconds, max_M_drift,
)
    joint = Float64[]
    sampling_joint = Float64[]
    action = Float64[]
    sampling_action = Float64[]
    action_ks = Float64[]
    for index in 2:length(times)
        joint_metric = sliced_w1(
            @view(candidate[index, :, :]),
            @view(reference[index, :, :]),
            @view(baseline[index, :, :]),
            directions,
        )
        action_metric = distribution_metrics(
            @view(candidate[index, 6, :]),
            @view(reference[index, 6, :]),
        )
        action_floor = distribution_metrics(
            @view(baseline[index, 6, :]),
            @view(reference[index, 6, :]),
        )
        push!(joint, joint_metric.distance)
        push!(sampling_joint, joint_metric.sampling_floor)
        push!(action, action_metric.w1)
        push!(sampling_action, action_floor.w1)
        push!(action_ks, action_metric.ks)
    end
    return (
        initial=initial,
        run=name,
        seconds=seconds,
        ms_per_step=1.0e3 * seconds / 5000,
        max_M_drift=max_M_drift,
        mean_joint=mean(joint),
        max_joint=maximum(joint),
        final_joint=joint[end],
        mean_sampling_joint=mean(sampling_joint),
        max_sampling_joint=maximum(sampling_joint),
        mean_action_w1=mean(action),
        max_action_w1=maximum(action),
        final_action_w1=action[end],
        mean_sampling_action_w1=mean(sampling_action),
        max_action_ks=maximum(action_ks),
    )
end

function write_summary(path, rows)
    open(path, "w") do io
        println(
            io,
            "initial\trun\tseconds\tms_per_step\tmax_M_drift\t",
            "mean_joint\tmax_joint\tfinal_joint\t",
            "mean_sampling_joint\tmax_sampling_joint\t",
            "mean_action_w1\tmax_action_w1\tfinal_action_w1\t",
            "mean_sampling_action_w1\tmax_action_ks",
        )
        for row in rows
            @printf(
                io,
                "%s\t%s\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n",
                row.initial,
                row.run,
                row.seconds,
                row.ms_per_step,
                row.max_M_drift,
                row.mean_joint,
                row.max_joint,
                row.final_joint,
                row.mean_sampling_joint,
                row.max_sampling_joint,
                row.mean_action_w1,
                row.max_action_w1,
                row.final_action_w1,
                row.mean_sampling_action_w1,
                row.max_action_ks,
            )
        end
    end
    return nothing
end

function main()
    L, Nensemble = 32, 5120
    dt = 0.002f0
    times = collect(0.0f0:0.1f0:10.0f0)
    p = RealspaceStats.realspace_parameters(
        κ=0.28f0,
        λ=0.022f0,
        γ=1.0f0,
        T=1.0f0,
    )
    root = normpath(joinpath(@__DIR__, "..", ".."))
    sources = (
        (
            initial="hmc",
            path=joinpath(
                root,
                "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip",
                "samples/hmc_chain_ep10000_N5120_seed20260901",
                "hmc_ep10000_t0_N5120.npz",
            ),
            reference=joinpath(
                @__DIR__,
                "..",
                "realspace_distribution",
                "output",
                "stats_hmc.jld2",
            ),
            seed=UInt64(202607280001),
        ),
        (
            initial="gaussian",
            path=joinpath(
                root,
                "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip",
                "samples/gaussian_chain_ep10000_N5120_seed20261001",
                "gaussian_ep10000_t0_N5120_seed20260721.npz",
            ),
            reference=joinpath(
                @__DIR__,
                "..",
                "realspace_distribution",
                "output",
                "stats_gaussian.jld2",
            ),
            seed=UInt64(202607280002),
        ),
    )
    direction_rng = Xoshiro(20260726)
    directions = randn(direction_rng, 6, 16)
    directions ./= sqrt.(sum(abs2, directions; dims=1))
    rows = NamedTuple[]
    rng_stats = Dict{String, Array{Float32, 3}}()

    warmed = false
    for source in sources
        data = npzread(source.path)
        cfgs = permutedims(Float32.(data["phi_gen"]), (2, 3, 1))
        @assert size(cfgs) == (L, L, Nensemble)
        u0 = CuArray(cfgs)
        if !warmed
            warm_u = copy(u0)
            warm_buffers =
                fused_philox_buffers(warm_u; seed=source.seed)
            tiled_philox_step!(
                warm_u,
                warm_buffers,
                dt,
                p.two_kappa,
                p.m02,
                p.c3,
                p.inv_gamma,
                p.noise_amplitude,
            )
            for (make_buffers, step!) in (
                (make_gpuarrays_buffers, modelb_fixed_step!),
                (make_direct_buffers, direct_curand_step!),
            )
                buffered_u = copy(u0)
                buffered_buffers =
                    make_buffers(buffered_u, source.seed)
                step!(
                    buffered_u,
                    buffered_buffers,
                    dt,
                    p.two_kappa,
                    p.m02,
                    p.c3,
                    p.inv_gamma,
                    p.noise_amplitude,
                )
            end
            gpu_obs = CUDA.zeros(Float64, 6, Nensemble)
            RealspaceStats.realspace_observables!(
                gpu_obs,
                warm_u,
                p,
            )
            CUDA.synchronize()
            warmed = true
        end

        reference, baseline, standard, standard_seconds, standard_M_drift =
            jldopen(source.reference, "r") do file
                (
                    file["runs/reference_a/stats"],
                    file["runs/reference_b/stats"],
                    file["runs/sosra_dt0p002/stats"],
                    Float64(file["runs/sosra_dt0p002/seconds"]),
                    Float64(file["runs/sosra_dt0p002/max_M_drift"]),
                )
            end
        push!(
            rows,
            summarize(
                source.initial,
                "standard_curand",
                standard,
                reference,
                baseline,
                times,
                directions,
                standard_seconds,
                standard_M_drift,
            ),
        )

        for (replica, seed_offset) in (("a", UInt64(0)), ("b", UInt64(100)))
            tiled = run_tiled(
                u0,
                times,
                dt,
                source.seed + seed_offset,
                p,
            )
            key = "$(source.initial)_$replica"
            rng_stats["tiled_$key"] = tiled.stats
            push!(
                rows,
                summarize(
                    source.initial,
                    "tiled_philox_$replica",
                    tiled.stats,
                    reference,
                    baseline,
                    times,
                    directions,
                    tiled.seconds,
                    tiled.max_M_drift,
                ),
            )
            println(
                "$(source.initial) $replica: tiled $(round(tiled.seconds; digits=4)) s",
            )
            flush(stdout)
        end

        for (name, make_buffers, step!) in (
            (
                "gpuarrays_public",
                make_gpuarrays_buffers,
                modelb_fixed_step!,
            ),
            (
                "curand_direct_even",
                make_direct_buffers,
                direct_curand_step!,
            ),
        )
            buffered = run_buffered(
                u0,
                times,
                dt,
                source.seed,
                p,
                make_buffers,
                step!,
            )
            key = "$(name)_$(source.initial)"
            rng_stats[key] = buffered.stats
            push!(
                rows,
                summarize(
                    source.initial,
                    name,
                    buffered.stats,
                    reference,
                    baseline,
                    times,
                    directions,
                    buffered.seconds,
                    buffered.max_M_drift,
                ),
            )
            println(
                "$(source.initial) $name: $(round(buffered.seconds; digits=4)) s",
            )
            flush(stdout)
        end
    end

    output = joinpath(@__DIR__, "output")
    mkpath(output)
    jldopen(joinpath(output, "l32_rng_stats.jld2"), "w") do file
        file["times"] = times
        for (key, stats) in rng_stats
            file["runs/$key/stats"] = stats
        end
    end
    write_summary(
        joinpath(output, "l32_distribution_summary.tsv"),
        rows,
    )
end

main()
