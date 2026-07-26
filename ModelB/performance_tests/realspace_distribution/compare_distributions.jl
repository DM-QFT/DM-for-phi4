using CUDA, JLD2, LinearAlgebra, NPZ, Printf, Random, Statistics

include(joinpath(@__DIR__, "realspace_methods.jl"))

const OBSERVABLES = (
    "M",
    "phi_site",
    "mean_phi2",
    "mean_phi4",
    "nearest_neighbor",
    "action_density",
)

function interval_steps(Δt, dt)
    steps = round(Int, Δt / dt)
    @assert isapprox(steps * dt, Δt; atol=8 * eps(Float32))
    return steps
end

function record_observables!(stats, index, u, gpu_obs, p)
    realspace_observables!(gpu_obs, u, p)
    @views stats[index, :, :] .= Float32.(Array(gpu_obs))
    return nothing
end

function run_sosra(u0, times, dt, seed, p)
    u = copy(u0)
    buffers = modelb_fixed_buffers(u; seed)
    gpu_obs = CUDA.zeros(Float64, 6, size(u, 3))
    stats = Array{Float32}(undef, length(times), 6, size(u, 3))
    steps = interval_steps(times[2] - times[1], dt)
    seconds = 0.0
    record_observables!(stats, 1, u, gpu_obs, p)
    for index in 2:length(times)
        CUDA.synchronize()
        seconds += @elapsed begin
            sosra_steps!(u, buffers, steps, dt, p)
            CUDA.synchronize()
        end
        record_observables!(stats, index, u, gpu_obs, p)
    end
    M0 = @view stats[1, 1, :]
    M_drift = @views stats[:, 1, :] .- reshape(M0, 1, :)
    max_M_drift = maximum(abs, M_drift)
    @assert all(isfinite, stats)
    return (
        method="sosra",
        dt=Float32(dt),
        seconds=seconds,
        max_M_drift=max_M_drift,
        stats=stats,
    )
end

function run_heun(u0, times, dt, seed, p)
    u = copy(u0)
    buffers = heun_buffers(u; seed)
    gpu_obs = CUDA.zeros(Float64, 6, size(u, 3))
    stats = Array{Float32}(undef, length(times), 6, size(u, 3))
    steps = interval_steps(times[2] - times[1], dt)
    seconds = 0.0
    record_observables!(stats, 1, u, gpu_obs, p)
    for index in 2:length(times)
        CUDA.synchronize()
        seconds += @elapsed begin
            heun_steps!(u, buffers, steps, dt, p)
            CUDA.synchronize()
        end
        record_observables!(stats, index, u, gpu_obs, p)
    end
    M0 = @view stats[1, 1, :]
    M_drift = @views stats[:, 1, :] .- reshape(M0, 1, :)
    max_M_drift = maximum(abs, M_drift)
    @assert all(isfinite, stats)
    return (
        method="heun",
        dt=Float32(dt),
        seconds=seconds,
        max_M_drift=max_M_drift,
        stats=stats,
    )
end

function run_euler(u0, times, dt, seed, p)
    u = copy(u0)
    v = similar(u)
    buffers = euler_buffers(u; seed)
    gpu_obs = CUDA.zeros(Float64, 6, size(u, 3))
    stats = Array{Float32}(undef, length(times), 6, size(u, 3))
    steps = interval_steps(times[2] - times[1], dt)
    seconds = 0.0
    record_observables!(stats, 1, u, gpu_obs, p)
    for index in 2:length(times)
        CUDA.synchronize()
        seconds += @elapsed begin
            u, v = euler_steps!(u, v, buffers, steps, dt, p)
            CUDA.synchronize()
        end
        record_observables!(stats, index, u, gpu_obs, p)
    end
    M0 = @view stats[1, 1, :]
    M_drift = @views stats[:, 1, :] .- reshape(M0, 1, :)
    max_M_drift = maximum(abs, M_drift)
    @assert all(isfinite, stats)
    return (
        method="euler",
        dt=Float32(dt),
        seconds=seconds,
        max_M_drift=max_M_drift,
        stats=stats,
    )
end

function warmup(u0, p)
    u = copy(u0)
    v = similar(u)
    eb = euler_buffers(u; seed=UInt64(10))
    u, v = euler_steps!(u, v, eb, 1, 0.002f0, p)

    u = copy(u0)
    hb = heun_buffers(u; seed=UInt64(11))
    heun_steps!(u, hb, 1, 0.002f0, p)

    u = copy(u0)
    sb = modelb_fixed_buffers(u; seed=UInt64(12))
    sosra_steps!(u, sb, 1, 0.002f0, p)

    gpu_obs = CUDA.zeros(Float64, 6, size(u, 3))
    realspace_observables!(gpu_obs, u, p)
    Array(gpu_obs)
    return nothing
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

function distribution_metrics(x, reference)
    x64 = Float64.(x)
    reference64 = Float64.(reference)
    scale = std(reference64)
    @assert scale > 0.0
    sorted_x = sort!(x64)
    sorted_reference = sort!(reference64)
    return (
        mean_shift_sigma=(mean(x) - mean(reference)) / scale,
        w1_sigma=mean(abs, sorted_x .- sorted_reference) / scale,
        ks=ks_distance_sorted(sorted_x, sorted_reference),
    )
end

function sliced_w1(candidate, reference, baseline, directions)
    reference64 = Float64.(reference)
    candidate64 = Float64.(candidate)
    baseline64 = Float64.(baseline)
    μ = mean(reference64; dims=2)
    σ = std(reference64; dims=2)
    @assert minimum(σ) > 0.0
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
        @assert scale > 0.0
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

function save_raw_stats(path, times, source, runs, order)
    jldopen(path, "w") do file
        file["times"] = Float32.(times)
        file["observable_names"] = collect(OBSERVABLES)
        file["source_path"] = source
        for key in order
            run = runs[key]
            file["runs/$key/method"] = run.method
            file["runs/$key/dt"] = run.dt
            file["runs/$key/seconds"] = run.seconds
            file["runs/$key/max_M_drift"] = run.max_M_drift
            file["runs/$key/stats"] = run.stats
        end
    end
    return nothing
end

function write_metrics(path, rows)
    open(path, "w") do io
        println(
            io,
            "initial\trun\tmethod\tdt\ttime\tobservable\t",
            "mean_shift_sigma\tw1_sigma\tks\t",
            "sampling_w1_sigma\tsampling_ks",
        )
        for row in rows
            @printf(
                io,
                "%s\t%s\t%s\t%.7g\t%.7g\t%s\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n",
                row.initial, row.run, row.method, row.dt, row.time,
                row.observable, row.mean_shift_sigma, row.w1_sigma,
                row.ks, row.sampling_w1_sigma, row.sampling_ks,
            )
        end
    end
    return nothing
end

function write_aggregate(path, rows)
    open(path, "w") do io
        println(
            io,
            "initial\trun\tmethod\tdt\tobservable\t",
            "max_abs_mean_shift_sigma\tmean_w1_sigma\tmax_w1_sigma\t",
            "final_w1_sigma\tmax_ks\tfinal_ks\t",
            "mean_sampling_w1_sigma\tmax_sampling_ks",
        )
        for row in rows
            @printf(
                io,
                "%s\t%s\t%s\t%.7g\t%s\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n",
                row.initial, row.run, row.method, row.dt, row.observable,
                row.max_abs_mean_shift_sigma, row.mean_w1_sigma,
                row.max_w1_sigma, row.final_w1_sigma, row.max_ks,
                row.final_ks, row.mean_sampling_w1_sigma,
                row.max_sampling_ks,
            )
        end
    end
    return nothing
end

function write_joint_metrics(path, rows)
    open(path, "w") do io
        println(
            io,
            "initial\trun\tmethod\tdt\ttime\t",
            "sliced_w1\tsampling_sliced_w1",
        )
        for row in rows
            @printf(
                io,
                "%s\t%s\t%s\t%.7g\t%.7g\t%.8g\t%.8g\n",
                row.initial, row.run, row.method, row.dt, row.time,
                row.sliced_w1, row.sampling_sliced_w1,
            )
        end
    end
    return nothing
end

function write_timings(path, rows)
    open(path, "w") do io
        println(
            io,
            "initial\trun\tmethod\tdt\tsteps\tseconds\t",
            "ms_per_step\tmax_M_drift",
        )
        for row in rows
            @printf(
                io,
                "%s\t%s\t%s\t%.7g\t%d\t%.8g\t%.8g\t%.8g\n",
                row.initial, row.run, row.method, row.dt, row.steps,
                row.seconds, row.ms_per_step, row.max_M_drift,
            )
        end
    end
    return nothing
end

function analyze!(
    full_rows, aggregate_rows, joint_rows, initial, times, runs, order,
    directions,
)
    reference = runs["reference_a"].stats
    baseline = runs["reference_b"].stats
    for key in order[2:end]
        run = runs[key]
        for observable_index in eachindex(OBSERVABLES)
            metrics = NamedTuple[]
            sampling = NamedTuple[]
            for time_index in eachindex(times)
                metric = distribution_metrics(
                    @view(run.stats[time_index, observable_index, :]),
                    @view(reference[time_index, observable_index, :]),
                )
                floor = distribution_metrics(
                    @view(baseline[time_index, observable_index, :]),
                    @view(reference[time_index, observable_index, :]),
                )
                push!(metrics, metric)
                push!(sampling, floor)
                push!(
                    full_rows,
                    (
                        initial=initial,
                        run=key,
                        method=run.method,
                        dt=run.dt,
                        time=times[time_index],
                        observable=OBSERVABLES[observable_index],
                        mean_shift_sigma=metric.mean_shift_sigma,
                        w1_sigma=metric.w1_sigma,
                        ks=metric.ks,
                        sampling_w1_sigma=floor.w1_sigma,
                        sampling_ks=floor.ks,
                    ),
                )
            end
            dynamic_indices = 2:length(times)
            push!(
                aggregate_rows,
                (
                    initial=initial,
                    run=key,
                    method=run.method,
                    dt=run.dt,
                    observable=OBSERVABLES[observable_index],
                    max_abs_mean_shift_sigma=maximum(
                        abs(metric.mean_shift_sigma)
                        for metric in metrics[dynamic_indices]
                    ),
                    mean_w1_sigma=mean(
                        metric.w1_sigma
                        for metric in metrics[dynamic_indices]
                    ),
                    max_w1_sigma=maximum(
                        metric.w1_sigma
                        for metric in metrics[dynamic_indices]
                    ),
                    final_w1_sigma=metrics[end].w1_sigma,
                    max_ks=maximum(
                        metric.ks for metric in metrics[dynamic_indices]
                    ),
                    final_ks=metrics[end].ks,
                    mean_sampling_w1_sigma=mean(
                        floor.w1_sigma
                        for floor in sampling[dynamic_indices]
                    ),
                    max_sampling_ks=maximum(
                        floor.ks for floor in sampling[dynamic_indices]
                    ),
                ),
            )
        end

        for time_index in eachindex(times)
            metric = sliced_w1(
                @view(run.stats[time_index, :, :]),
                @view(reference[time_index, :, :]),
                @view(baseline[time_index, :, :]),
                directions,
            )
            push!(
                joint_rows,
                (
                    initial=initial,
                    run=key,
                    method=run.method,
                    dt=run.dt,
                    time=times[time_index],
                    sliced_w1=metric.distance,
                    sampling_sliced_w1=metric.sampling_floor,
                ),
            )
        end
    end
    return nothing
end

function main()
    L, Nensemble = 32, 5120
    tmax, save_dt = 10.0f0, 0.1f0
    reference_dt = 0.0005f0
    dts = (0.002f0, 0.005f0, 0.01f0, 0.02f0)
    p = realspace_parameters(κ=0.28f0, λ=0.022f0, γ=1.0f0, T=1.0f0)
    times = collect(0.0f0:save_dt:tmax)
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
            seed=UInt64(202607260000),
        ),
        (
            initial="gaussian",
            path=joinpath(
                root,
                "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip",
                "samples/gaussian_chain_ep10000_N5120_seed20261001",
                "gaussian_ep10000_t0_N5120_seed20260721.npz",
            ),
            seed=UInt64(202607270000),
        ),
    )
    output = joinpath(@__DIR__, "output")
    mkpath(output)
    full_rows = NamedTuple[]
    aggregate_rows = NamedTuple[]
    joint_rows = NamedTuple[]
    timing_rows = NamedTuple[]
    direction_rng = Xoshiro(20260726)
    directions = randn(direction_rng, length(OBSERVABLES), 16)
    directions ./= sqrt.(sum(abs2, directions; dims=1))

    warmed = false
    for source in sources
        data = npzread(source.path)
        cfgs = permutedims(Float32.(data["phi_gen"]), (2, 3, 1))
        @assert size(cfgs) == (L, L, Nensemble)
        u0 = CuArray(cfgs)
        if !warmed
            warmup(u0, p)
            warmed = true
        end

        runs = Dict{String, Any}()
        order = String[]
        println("$(source.initial): reference_a")
        flush(stdout)
        runs["reference_a"] =
            run_sosra(u0, times, reference_dt, source.seed + 1, p)
        push!(order, "reference_a")
        println("$(source.initial): reference_b")
        flush(stdout)
        runs["reference_b"] =
            run_sosra(u0, times, reference_dt, source.seed + 2, p)
        push!(order, "reference_b")

        run_index = UInt64(10)
        for method in ("sosra", "heun", "euler"), dt in dts
            key = "$(method)_dt$(replace(string(dt), "." => "p"))"
            println("$(source.initial): $key")
            flush(stdout)
            run =
                method == "sosra" ?
                run_sosra(
                    u0, times, dt, source.seed + run_index, p,
                ) :
                method == "heun" ?
                run_heun(
                    u0, times, dt, source.seed + run_index, p,
                ) :
                run_euler(
                    u0, times, dt, source.seed + run_index, p,
                )
            runs[key] = run
            push!(order, key)
            run_index += 1
        end

        for key in order
            run = runs[key]
            steps = round(Int, tmax / run.dt)
            push!(
                timing_rows,
                (
                    initial=source.initial,
                    run=key,
                    method=run.method,
                    dt=run.dt,
                    steps=steps,
                    seconds=run.seconds,
                    ms_per_step=1.0e3 * run.seconds / steps,
                    max_M_drift=run.max_M_drift,
                ),
            )
        end
        save_raw_stats(
            joinpath(output, "stats_$(source.initial).jld2"),
            times,
            source.path,
            runs,
            order,
        )
        analyze!(
            full_rows,
            aggregate_rows,
            joint_rows,
            source.initial,
            times,
            runs,
            order,
            directions,
        )
    end

    write_metrics(joinpath(output, "time_metrics.tsv"), full_rows)
    write_aggregate(joinpath(output, "aggregate_metrics.tsv"), aggregate_rows)
    write_joint_metrics(joinpath(output, "joint_metrics.tsv"), joint_rows)
    write_timings(joinpath(output, "timings.tsv"), timing_rows)
end

main()
