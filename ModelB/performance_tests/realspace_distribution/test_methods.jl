using CUDA, Random, Statistics, Test

include(joinpath(@__DIR__, "realspace_methods.jl"))

function cpu_drift(u, p)
    L, _, Nensemble = size(u)
    drift = similar(u)
    @inbounds for e in 1:Nensemble, y in 1:L, x in 1:L
        xm1 = mod1(x - 1, L)
        xp1 = mod1(x + 1, L)
        xm2 = mod1(x - 2, L)
        xp2 = mod1(x + 2, L)
        ym1 = mod1(y - 1, L)
        yp1 = mod1(y + 1, L)
        ym2 = mod1(y - 2, L)
        yp2 = mod1(y + 2, L)
        center = u[x, y, e]
        nearest =
            u[xm1, y, e] + u[xp1, y, e] +
            u[x, ym1, e] + u[x, yp1, e]
        diagonal =
            u[xm1, ym1, e] + u[xm1, yp1, e] +
            u[xp1, ym1, e] + u[xp1, yp1, e]
        second =
            u[xm2, y, e] + u[xp2, y, e] +
            u[x, ym2, e] + u[x, yp2, e]
        minus_biharmonic =
            -20.0f0 * center + 8.0f0 * nearest -
            2.0f0 * diagonal - second
        laplacian = nearest - 4.0f0 * center
        laplacian3 =
            u[xm1, y, e]^3 + u[xp1, y, e]^3 +
            u[x, ym1, e]^3 + u[x, yp1, e]^3 - 4.0f0 * center^3
        drift[x, y, e] = p.inv_gamma * (
            p.two_kappa * minus_biharmonic +
            p.m02 * laplacian + p.c3 * laplacian3
        )
    end
    return drift
end

function cpu_observables(u, p)
    L, _, Nensemble = size(u)
    obs = zeros(Float64, 6, Nensemble)
    @inbounds for e in 1:Nensemble
        for y in 1:L, x in 1:L
            right = Float64(u[mod1(x + 1, L), y, e])
            up = Float64(u[x, mod1(y + 1, L), e])
            ϕ = Float64(u[x, y, e])
            ϕ2 = ϕ * ϕ
            obs[1, e] += ϕ
            obs[3, e] += ϕ2
            obs[4, e] += ϕ2 * ϕ2
            obs[5, e] += 0.5 * ϕ * (right + up)
            obs[6, e] +=
                Float64(p.κ) * ((ϕ - right)^2 + (ϕ - up)^2) +
                0.5 * Float64(p.m02) * ϕ2 + Float64(p.λ) * ϕ2 * ϕ2
        end
        obs[1, e] /= L * L
        obs[2, e] = u[1, 1, e]
        obs[3:6, e] ./= L * L
    end
    return obs
end

function main()
    L, Nensemble = 8, 7
    dt = 0.001f0
    p0 = realspace_parameters(T=0.0f0)
    rng = Xoshiro(20260726)
    u0 = randn(rng, Float32, L, L, Nensemble)
    k1 = cpu_drift(u0, p0)
    euler_reference = u0 .+ dt .* k1
    predictor = euler_reference
    heun_reference =
        u0 .+ 0.5f0 .* dt .* (k1 .+ cpu_drift(predictor, p0))

    u = CuArray(u0)
    v = similar(u)
    eb = euler_buffers(u; seed=UInt64(1))
    u, v = euler_steps!(u, v, eb, 1, dt, p0)
    @test Array(u) ≈ euler_reference rtol=2.0e-6 atol=2.0e-6

    u = CuArray(u0)
    hb = heun_buffers(u; seed=UInt64(2))
    heun_steps!(u, hb, 1, dt, p0)
    @test Array(u) ≈ heun_reference rtol=3.0e-6 atol=3.0e-6

    p = realspace_parameters()
    u = CuArray(u0)
    gpu_obs = CUDA.zeros(Float64, 6, Nensemble)
    realspace_observables!(gpu_obs, u, p)
    @test Array(gpu_obs) ≈ cpu_observables(u0, p) rtol=2.0e-12 atol=2.0e-12

    M0 = vec(mean(Float64, u0; dims=(1, 2)))
    u = CuArray(u0)
    v = similar(u)
    eb = euler_buffers(u; seed=UInt64(3))
    u, v = euler_steps!(u, v, eb, 100, dt, p)
    @test maximum(abs, vec(mean(Float64, Array(u); dims=(1, 2))) - M0) < 2.0e-6

    u = CuArray(u0)
    hb = heun_buffers(u; seed=UInt64(4))
    heun_steps!(u, hb, 100, dt, p)
    @test maximum(abs, vec(mean(Float64, Array(u); dims=(1, 2))) - M0) < 2.0e-6
end

main()
