include(joinpath(@__DIR__, "common.jl"))

using CUDA.CUFFT
using LinearAlgebra

"""Batched finite-difference drift evaluated through cuFFT."""
struct SpectralDriftWorkspace
    phi3::CuArray{Float32, 3}
    uhat::CuArray{ComplexF32, 3}
    phi3hat::CuArray{ComplexF32, 3}
    outhat::CuArray{ComplexF32, 3}
    linear_symbol::CuArray{Float32, 3}
    nonlinear_symbol::CuArray{Float32, 3}
    forward_plan
    inverse_plan
end

function SpectralDriftWorkspace(template::CuArray{Float32, 3};
                                two_kappa=0.541f0,
                                m02=-0.252f0,
                                c3=0.088f0,
                                inv_gamma=1.0f0)
    L = size(template, 1)
    @assert size(template, 2) == L
    phi3 = similar(template)
    forward_plan = plan_rfft(template, (1, 2))
    uhat = forward_plan * template
    phi3hat = similar(uhat)
    outhat = similar(uhat)
    inverse_plan = plan_irfft(uhat, L, (1, 2))

    linear = Array{Float32}(undef, div(L, 2) + 1, L, 1)
    nonlinear = similar(linear)
    for j in 1:L, i in 1:(div(L, 2) + 1)
        kx = 2.0f0 * Float32(pi) * Float32(i - 1) / Float32(L)
        ky = 2.0f0 * Float32(pi) * Float32(j - 1) / Float32(L)
        lap = 2.0f0 * cos(kx) + 2.0f0 * cos(ky) - 4.0f0
        linear[i, j, 1] = inv_gamma * (two_kappa * (-lap * lap) + m02 * lap)
        nonlinear[i, j, 1] = inv_gamma * c3 * lap
    end

    return SpectralDriftWorkspace(
        phi3, uhat, phi3hat, outhat,
        CuArray(linear), CuArray(nonlinear),
        forward_plan, inverse_plan,
    )
end

function spectral_drift!(du, u, ws::SpectralDriftWorkspace)
    @. ws.phi3 = u * u * u
    mul!(ws.uhat, ws.forward_plan, u)
    mul!(ws.phi3hat, ws.forward_plan, ws.phi3)
    @. ws.outhat = ws.linear_symbol * ws.uhat +
        ws.nonlinear_symbol * ws.phi3hat
    mul!(du, ws.inverse_plan, ws.outhat)
    return nothing
end

