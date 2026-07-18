include(joinpath(@__DIR__, "common.jl"))

function noise_dist_async!(rand_vec, W, dt, u, p, t, rng)
    noise_work, xi_x, xi_y = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_y)
    launch_conserved_noise!(noise_work, xi_x, xi_y)
    @. rand_vec = sqrt(abs(dt)) * noise_work
    return nothing
end

function noise_bridge_async!(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    noise_work, xi_x, xi_y = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_x)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), xi_y)
    launch_conserved_noise!(noise_work, xi_x, xi_y)
    @. rand_vec = sqrt((1 - q) * q * abs(h)) * noise_work + q * Wh
    return nothing
end

function async_noise_process(t0, template)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        noise_dist_async!,
        noise_bridge_async!;
        save_everystep=false,
    )
end

function rswm1_noise_process(t0, template)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        modelb_noise_dist!,
        modelb_noise_bridge!;
        save_everystep=false,
        rswm=DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1),
    )
end

function rswm1_async_noise_process(t0, template)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        noise_dist_async!,
        noise_bridge_async!;
        save_everystep=false,
        rswm=DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1),
    )
end
