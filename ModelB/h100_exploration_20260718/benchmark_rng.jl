include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "solver_kernel_variants.jl"))
include(joinpath(@__DIR__, "rng_variants.jl"))

using .H100RNGVariants
using DiffEqNoiseProcess
using Random
using Printf

const RSWM1 = DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1)

function bench_ms(f; warmup=3, samples=15)
    for _ in 1:warmup
        f()
    end
    CUDA.device_synchronize()
    timings = Float64[]
    for _ in 1:samples
        CUDA.device_synchronize()
        t = @elapsed begin
            f()
            CUDA.device_synchronize()
        end
        push!(timings, 1e3t)
    end
    return (median=median(timings), minimum=minimum(timings), maximum=maximum(timings))
end

function print_timing(name, timing)
    @printf("TIMING\t%s\t%.6f\t%.6f\t%.6f\n",
            name, timing.median, timing.minimum, timing.maximum)
end

function raw_statistics!(rng, packed)
    Random.randn!(rng, packed)
    CUDA.device_synchronize()
    x = @view packed[:, :, :, 1]
    y = @view packed[:, :, :, 2]
    mx, my = Float64(mean(x)), Float64(mean(y))
    vx, vy = Float64(var(x)), Float64(var(y))
    covariance = Float64(mean((x .- Float32(mx)) .* (y .- Float32(my))))
    correlation = covariance / sqrt(vx * vy)
    return (; mx, my, vx, vy, correlation)
end

function noise_statistics!(rng, packed, out)
    Random.randn!(rng, packed)
    H100RNGVariants.launch_packed_scaled128!(
        out, packed, 1.0f0; threads=1024, blocks=264,
    )
    CUDA.device_synchronize()
    ensemble_sum = sum(out; dims=(1, 2))
    return (
        mean=Float64(mean(out)),
        variance=Float64(var(out)),
        max_abs_ensemble_sum=Float64(maximum(abs, ensemble_sum)),
        rms_ensemble_sum=Float64(sqrt(mean(abs2, ensemble_sum))),
    )
end

function run_micro(; ensembles=1024)
    dims = (128, 128, ensembles)
    xi_x = CUDA.zeros(Float32, dims)
    xi_y = similar(xi_x)
    out = similar(xi_x)
    packed = CUDA.zeros(Float32, dims..., 2)

    CUDA.seed!(20260718)
    default_rng = CUDA.default_rng()
    current_two = () -> begin
        DiffEqNoiseProcess.wiener_randn!(default_rng, xi_x)
        DiffEqNoiseProcess.wiener_randn!(default_rng, xi_y)
    end
    current_four = () -> begin
        current_two()
        current_two()
    end
    current_two_fused = () -> begin
        current_two()
        launch_noise_scaled128!(
            out, xi_x, xi_y, 1.0f0; threads=1024, blocks=264,
        )
    end
    current_four_fused = () -> begin
        current_two_fused()
        current_two_fused()
    end
    print_timing("current_two_split_fills", bench_ms(current_two))
    print_timing("current_four_split_fills", bench_ms(current_four))
    print_timing("current_one_callback_split_fused", bench_ms(current_two_fused))
    print_timing("current_two_callbacks_split_fused", bench_ms(current_four_fused))

    for kind in H100RNGVariants.RNG_CANDIDATES
        try
            rng = H100RNGVariants.make_gpu_rng(kind, 20260718)
            split_one = () -> begin
                Random.randn!(rng, xi_x)
                Random.randn!(rng, xi_y)
            end
            packed_one = () -> Random.randn!(rng, packed)
            packed_two_callbacks = () -> begin
                Random.randn!(rng, packed)
                H100RNGVariants.launch_packed_scaled128!(
                    out, packed, 1.0f0; threads=1024, blocks=264,
                )
                Random.randn!(rng, packed)
                H100RNGVariants.launch_packed_scaled128!(
                    out, packed, 1.0f0; threads=1024, blocks=264,
                )
            end
            print_timing("$(kind)_split_two_fills", bench_ms(split_one))
            print_timing("$(kind)_packed_one_fill", bench_ms(packed_one))
            print_timing("$(kind)_two_callbacks_packed_fused", bench_ms(packed_two_callbacks))
            raw = raw_statistics!(rng, packed)
            conserved = noise_statistics!(rng, packed, out)
            @printf("STATS\t%s\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\t%.8g\n",
                    kind, raw.mx, raw.my, raw.vx, raw.vy, raw.correlation,
                    conserved.mean, conserved.variance,
                    conserved.max_abs_ensemble_sum,
                    conserved.rms_ensemble_sum)
        catch error
            println("UNSUPPORTED\t", kind, "\t", typeof(error), "\t", sprint(showerror, error))
            CUDA.device_synchronize()
        end
    end

    # cuRAND ordering can materially change generation layout and speed.  Test
    # every ordering exposed by CUDA.jl for the fastest generator above.
    for ordering in keys(H100RNGVariants.RNG_ORDERINGS)
        try
            rng = H100RNGVariants.make_gpu_rng(
                :library_philox, 20260718; ordering,
            )
            packed_one = () -> Random.randn!(rng, packed)
            print_timing(
                "library_philox_order_$(ordering)_packed_one_fill",
                bench_ms(packed_one),
            )
        catch error
            println("UNSUPPORTED\tlibrary_philox_order_", ordering, "\t",
                    typeof(error), "\t", sprint(showerror, error))
            CUDA.device_synchronize()
        end
    end

    # Measure stream/event designs through their actual callback objects.  Two
    # dist calls emulate the two independent increments required by fixed SOSRA.
    for (name, builder) in (
        ("dual_stream_default", (t, u) -> H100RNGVariants.dual_stream_noise_process128(
            t, u; seed=20260718, rswm=RSWM1,
        )),
        ("prefetch_packed_default", (t, u) -> H100RNGVariants.prefetch_packed_noise_process128(
            t, u; seed=20260718, rswm=RSWM1,
        )),
    )
        try
            template = CUDA.zeros(Float32, dims)
            process = builder(0.0f0, template)
            callback = () -> begin
                process.dist(out, process.W, 0.002f0, template, nothing, 0.0f0, process.rng)
                process.dist(out, process.W, 0.002f0, template, nothing, 0.0f0, process.rng)
            end
            print_timing("$(name)_two_callbacks", bench_ms(callback; warmup=3, samples=15))
        catch error
            println("UNSUPPORTED\t", name, "\t", typeof(error), "\t", sprint(showerror, error))
            CUDA.device_synchronize()
        end
    end
end

function noise_builder_for(mode::String)
    if mode == "current_split_default"
        return rswm1_linear_fused_noise_process
    elseif startswith(mode, "packed_")
        kind = Symbol(mode[8:end])
        return (t, u) -> H100RNGVariants.packed_noise_process128(
            t, u; rng_kind=kind, seed=314159, rswm=RSWM1,
        )
    elseif startswith(mode, "prefetch_")
        kind = Symbol(mode[10:end])
        return (t, u) -> H100RNGVariants.prefetch_packed_noise_process128(
            t, u; rng_kind=kind, seed=314159, rswm=RSWM1,
        )
    elseif startswith(mode, "dual_")
        kind = Symbol(mode[6:end])
        return (t, u) -> H100RNGVariants.dual_stream_noise_process128(
            t, u; rng_kind=kind, seed=314159, rswm=RSWM1,
        )
    end
    error("unknown RNG full-solver mode: $mode")
end

function run_solver(mode; t_end=2.0f0, repetitions=3)
    noise_builder = noise_builder_for(mode)

    # Compile every kernel and callback on a tiny ensemble before timing.
    warm = make_u0(L=128, ensembles=2, seed=11)
    solve_sosra(
        warm;
        drift_launch! = drift_linear128_best!,
        noise_builder,
        synchronize_callbacks=false,
        dt=DT_DEFAULT,
        t_end=DT_DEFAULT,
    )
    CUDA.device_synchronize()

    u0 = make_u0()
    timings = Float64[]
    for repetition in 1:repetitions
        CUDA.seed!(314159 + repetition)
        elapsed, solution = synchronized_seconds() do
            solve_sosra(
                u0;
                drift_launch! = drift_linear128_best!,
                noise_builder,
                synchronize_callbacks=false,
                dt=DT_DEFAULT,
                t_end,
            )
        end
        final = solution.u[end]
        max_mass = Float64(maximum(abs.(sum(final; dims=(1, 2)))))
        @printf("SOLVER\t%s\t%d\t%.9f\t%s\t%d\t%d\t%s\t%.8g\t%.8g\n",
                mode, repetition, elapsed, solution.retcode,
                solution.stats.naccept, solution.stats.nreject,
                all(isfinite, final), Float64(maximum(abs, final)), max_mass)
        push!(timings, elapsed)
        solution = nothing
        final = nothing
        GC.gc(false)
    end
    warm_median = repetitions > 1 ? median(timings[2:end]) : timings[1]
    @printf("SOLVER_SUMMARY\t%s\t%.9f\t%.9f\n", mode, median(timings), warm_median)
end

function main()
    action = isempty(ARGS) ? "micro" : ARGS[1]
    if action == "micro"
        ensembles = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1024
        run_micro(; ensembles)
    elseif action == "solver"
        mode = ARGS[2]
        t_end = length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : 2.0f0
        repetitions = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
        run_solver(mode; t_end, repetitions)
    else
        error("usage: benchmark_rng.jl [micro [ensembles] | solver MODE [t_end] [repetitions]]")
    end
end

main()
