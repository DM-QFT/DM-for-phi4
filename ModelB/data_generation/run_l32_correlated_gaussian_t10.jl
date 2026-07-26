using CUDA, FFTW, JLD2, Random, Statistics
include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))
L, Nensemble = 32, 51200
κ, λ, γ, T, dt, horizon = 0.28f0, 0.022f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
initial_seed, noise_seed = UInt64(20260802), UInt64(20260803)
dir = normpath(joinpath(@__DIR__, "..", "runs", "L32_Ne51200_dt0p002_corrgauss_uniformM_t10"))
condition = joinpath(dir, "modelB_L32_Ne51200_corrgauss_t0_seed20260802.jld2")
target = joinpath(dir, "modelB_L32_Ne51200_corrgauss_kappa_0.28_t10_dt0.002_seed20260803.jld2")
mkpath(dir)
rng = Xoshiro(initial_seed)
Z = 0.2f0 .+ 0.8f0 .* rand(rng, Float32, Nensemble)
m2 = 0.5f0 .* rand(rng, Float32, Nensemble)
M = 3.6f0 .* rand(rng, Float32, Nensemble) .- 1.8f0
q = 4.0f0 .* sinpi.(Float32.(0:L-1) ./ L) .^ 2
q2 = reshape(q, L, 1) .+ reshape(q, 1, L)
# q²(0)=∞ removes the zero mode before adding the independently sampled M.
q2[1, 1] = Inf32
CUDA.seed!(initial_seed)
F = fft(CUDA.randn(Float32, L, L, Nensemble), (1, 2))
F ./= sqrt.(reshape(CuArray(q2), L, L, 1) .* reshape(CuArray(Z), 1, 1, :) .+
            reshape(CuArray(m2), 1, 1, :))
u = real.(ifft(F, (1, 2)))
u .+= reshape(CuArray(M), 1, 1, :)
cfgs = Array(u)
M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
@assert maximum(abs, M0 - M) < 1.0e-5
jldsave(condition; cfgs, L, Nensemble, kappa=κ, lambda=λ, propagator_Z=Z,
        propagator_m2=m2, constant_mean=M, initial_seed)
buffers = modelb_fixed_buffers(u; seed=noise_seed)
seconds = @elapsed begin
    modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, horizon))
    CUDA.synchronize()
end
@assert all(isfinite, u)
cfgs = Array(u)
drift = vec(mean(Float64, cfgs; dims=(1, 2))) - M0
jldsave(target; cfgs, L, Nensemble, time=horizon, kappa=κ, lambda=λ, gamma=γ,
        temperature=T, dt, noise_seed, source_path=condition,
        max_abs_conservation_drift=maximum(abs, drift))
println("time=$(round(seconds; digits=1)) s, max|ΔM|=$(maximum(abs, drift))")
