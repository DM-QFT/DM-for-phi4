include(joinpath(@__DIR__, "solver_kernel_variants.jl"))

using Printf

const VALIDATION_TEND = 0.020f0
const VALIDATION_SEED = 271828

function run_variant(
    name,
    u0;
    drift_launch!,
    noise_builder,
    synchronize_callbacks,
)
    CUDA.seed!(VALIDATION_SEED)
    elapsed, sol = synchronized_seconds() do
        solve_sosra(
            u0;
            drift_launch!,
            noise_builder,
            synchronize_callbacks,
            dt=DT_DEFAULT,
            t_end=VALIDATION_TEND,
        )
    end
    final = copy(sol.u[end])
    @printf(
        "%s seconds=%.6f retcode=%s naccept=%d finite=%s maxabs=%.9g max_mass=%.9g\n",
        name,
        elapsed,
        string(sol.retcode),
        sol.stats.naccept,
        string(all(isfinite, final)),
        maximum(abs, final),
        maximum(abs.(sum(final; dims=(1, 2)))),
    )
    return final
end

function compare(name, candidate, reference)
    difference = candidate .- reference
    max_abs = maximum(abs, difference)
    rms = sqrt(sum(abs2, difference) / length(difference))
    exact = all(candidate .== reference)
    @printf(
        "%s exact=%s max_abs_difference=%.9g rms_difference=%.9g\n",
        name, string(exact), max_abs, rms,
    )
end

function main()
    # Compile every kernel before the measured/equality runs.  In particular,
    # this avoids the observed first-launch anomaly of the unchanged 896-thread
    # production drift kernel when its output is freshly allocated.
    warm = make_u0(L=128, ensembles=2, seed=17)
    warm_du = similar(warm)
    fill!(warm_du, 0.0f0)
    launch_modelb_drift!(warm_du, warm, 0.541f0, -0.344f0, 0.088f0, 1.0f0)
    drift_linear128_best!(warm_du, warm, 0.541f0, -0.344f0, 0.088f0, 1.0f0)
    CUDA.synchronize()

    u0 = make_u0(seed=20260718)
    reference = run_variant(
        "production_rswm3_sync", u0;
        drift_launch! = launch_modelb_drift!,
        noise_builder=modelb_noise_process,
        synchronize_callbacks=true,
    )

    async_rswm3 = run_variant(
        "production_rswm3_async", u0;
        drift_launch! = launch_modelb_drift!,
        noise_builder=async_noise_process,
        synchronize_callbacks=false,
    )
    compare("production_async_vs_sync", async_rswm3, reference)

    linear_rswm3 = run_variant(
        "linear_fused_rswm3_async", u0;
        drift_launch! = drift_linear128_best!,
        noise_builder=linear_fused_noise_process,
        synchronize_callbacks=false,
    )
    compare("linear_fused_rswm3_vs_production", linear_rswm3, reference)

    production_rswm1 = run_variant(
        "production_rswm1_async", u0;
        drift_launch! = launch_modelb_drift!,
        noise_builder=rswm1_async_noise_process,
        synchronize_callbacks=false,
    )
    compare("production_rswm1_vs_rswm3", production_rswm1, reference)

    linear_rswm1 = run_variant(
        "linear_fused_rswm1_async", u0;
        drift_launch! = drift_linear128_best!,
        noise_builder=rswm1_linear_fused_noise_process,
        synchronize_callbacks=false,
    )
    compare("linear_fused_rswm1_vs_production_rswm1", linear_rswm1, production_rswm1)
    compare("linear_fused_rswm1_vs_production_rswm3", linear_rswm1, reference)
end

main()
