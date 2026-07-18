include(joinpath(@__DIR__, "solver_variants.jl"))
include(joinpath(@__DIR__, "kernels_variants.jl"))
using .H100KernelVariants

function drift_linear128_best!(du, u, two_kappa, m02, c3, inv_gamma)
    launch_drift_linear128!(
        du, u, two_kappa, m02, c3, inv_gamma;
        threads=1024, blocks=528,
    )
end

function drift_shared128_best!(du, u, two_kappa, m02, c3, inv_gamma)
    launch_drift_shared128_32x8!(du, u, two_kappa, m02, c3, inv_gamma)
end

function noise_dist_linear_fused_async!(rand_vec, W, dt, u, p, t, rng)
    _, xi_x, xi_y = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_y)
    launch_noise_scaled128!(
        rand_vec, xi_x, xi_y, sqrt(abs(dt));
        threads=1024, blocks=264,
    )
    return nothing
end

function noise_bridge_linear_fused_async!(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    _, xi_x, xi_y = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_y)
    launch_noise_bridge128!(
        rand_vec, xi_x, xi_y, sqrt((1 - q) * q * abs(h)), q, Wh;
        threads=1024, blocks=264,
    )
    return nothing
end

function linear_fused_noise_process(t0, template)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        noise_dist_linear_fused_async!,
        noise_bridge_linear_fused_async!;
        save_everystep=false,
    )
end

function rswm1_linear_fused_noise_process(t0, template)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        noise_dist_linear_fused_async!,
        noise_bridge_linear_fused_async!;
        save_everystep=false,
        rswm=DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1),
    )
end

