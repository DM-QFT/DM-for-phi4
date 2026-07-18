include(joinpath(@__DIR__, "library_variants.jl"))

function main()
    u = make_u0()
    reference = similar(u)
    spectral = similar(u)
    two_kappa = 0.541f0
    m02 = -0.252f0
    c3 = 0.088f0

    workspace = SpectralDriftWorkspace(
        u; two_kappa, m02, c3, inv_gamma=1.0f0
    )
    launch_modelb_drift!(reference, u, two_kappa, m02, c3, 1.0f0)
    spectral_drift!(spectral, u, workspace)
    CUDA.synchronize()

    difference = spectral .- reference
    println("spectral_max_abs=", maximum(abs, difference))
    println("spectral_rmse=", sqrt(mean(abs2, difference)))
    println("reference_rms=", sqrt(mean(abs2, reference)))

    baseline_time = kernel_median_seconds(
        () -> launch_modelb_drift!(reference, u, two_kappa, m02, c3, 1.0f0)
    )
    spectral_time = kernel_median_seconds(() -> spectral_drift!(spectral, u, workspace))
    println("baseline_drift_seconds=", baseline_time)
    println("spectral_drift_seconds=", spectral_time)
end

main()
