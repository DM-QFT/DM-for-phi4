include(joinpath(@__DIR__, "library_variants.jl"))

epoch_seconds() = time()

function marker(label)
    println(label, " epoch=", epoch_seconds())
    flush(stdout)
end

function run_batched_fft!(du, u, workspace, batches, batch)
    start_ns = time_ns()
    for _ in 1:batches
        for _ in 1:batch
            spectral_drift!(du, u, workspace)
        end
        CUDA.synchronize()
    end
    return Float64(time_ns() - start_ns) * 1.0e-9
end

function main()
    window_seconds = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 20.0
    batch = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32

    u = make_u0()
    du = similar(u)
    workspace = SpectralDriftWorkspace(u)
    for _ in 1:3
        spectral_drift!(du, u, workspace)
    end
    CUDA.synchronize()
    marker("POWER_WINDOW_START")

    # The spectral drift is ~0.78 ms on this device.  A fixed number of batches
    # avoids Julia 1.12's observed corruption of a loop-local wall-clock scalar
    # across repeated GPU calls in this process.
    batches = max(1, ceil(Int, window_seconds / (batch * 0.00078)))
    iterations = batches * batch
    elapsed = run_batched_fft!(du, u, workspace, batches, batch)
    marker("POWER_WINDOW_END")
    checksum = Array(@view du[1:1])[1]
    println(
        "POWER_SUMMARY mode=fft_drift_only",
        " iterations=", iterations,
        " window_seconds=", elapsed,
        " seconds_per_drift=", elapsed / iterations,
        " checksum=", checksum,
    )
end

main()
