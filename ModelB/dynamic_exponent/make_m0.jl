using JLD2, Random, Statistics, Printf

include(joinpath(@__DIR__, "..", "..", "julia", "WolffFAHMC_ND.jl"))

function hmc_m0(L, n, kappa, lambda, warmup, thin, nstep, seed)
    rng = MersenneTwister(seed)
    lat = Lattice{2}(L)
    phi = randn(rng, L, L)
    phi .-= mean(phi)
    pi = similar(phi)
    old = similar(phi)
    fm = FourierMassMatrix{2}(L, 0.0)
    # At the 2D Ising fixed point G(k)⁻¹ ∝ k^(2-η), η=1/4.
    @. fm.M_k = fm.M_k^0.875
    @. fm.sqrtM_k = sqrt(fm.M_k)
    @. fm.invM_k = 1 / fm.M_k
    fm.sqrtM_k[1] = 0
    fm.invM_k[1] = 0

    da = DualAveraging(0.05)
    epsilon = 0.05
    for _ in 1:warmup
        steps = rand(rng, round(Int, 0.8nstep):round(Int, 1.2nstep))
        _, delta_h = fahmc_step!(
            rng, phi, pi, fm, epsilon, steps, kappa, lambda, old, lat,
        )
        probability = isfinite(delta_h) ? min(1.0, exp(-delta_h)) : 0.0
        epsilon = da_update!(da, probability)
    end
    epsilon = da_final_ε(da)

    cfgs = Array{Float32}(undef, L, L, n)
    modes = Vector{ComplexF64}(undef, n)
    accepted = 0
    for sample in 1:n
        for _ in 1:thin
            steps = rand(rng, round(Int, 0.8nstep):round(Int, 1.2nstep))
            ok, _ = fahmc_step!(
                rng, phi, pi, fm, epsilon, steps, kappa, lambda, old, lat,
            )
            accepted += ok
        end
        mul!(fm.cbuf, fm.rfft_plan, phi)
        modes[sample] = fm.cbuf[2, 1] / L^2
        cfg = @view cfgs[:, :, sample]
        cfg .= phi
        cfg .-= mean(cfg)
    end
    centered = modes .- mean(modes)
    rho1 = real(dot(@view(centered[1:end-1]), @view(centered[2:end]))) /
           sum(abs2, @view(centered[1:end-1]))
    return cfgs, accepted / (n * thin), epsilon, rho1
end

@assert length(ARGS) == 9
L = parse(Int, ARGS[1])
n = parse(Int, ARGS[2])
kappa = parse(Float64, ARGS[3])
lambda = parse(Float64, ARGS[4])
warmup = parse(Int, ARGS[5])
thin = parse(Int, ARGS[6])
nstep = parse(Int, ARGS[7])
seed = parse(Int, ARGS[8])
output = abspath(ARGS[9])
@assert !ispath(output)

seconds = @elapsed cfgs, acceptance, epsilon, rho1 =
    hmc_m0(L, n, kappa, lambda, warmup, thin, nstep, seed)
magnetization = vec(mean(cfgs; dims=(1, 2)))
@assert maximum(abs, magnetization) < 1e-6
mkpath(dirname(output))
jldsave(
    output;
    cfgs, L, kappa, lambda, fixed_magnetization=0.0f0,
    acceptance, epsilon, rho1, warmup, thin, nstep, seed, seconds,
    max_abs_magnetization=maximum(abs, magnetization),
)
@printf(
    "saved %s  acceptance=%.3f  rho1=%.3f  max|M|=%.3e  %.1fs\n",
    output, acceptance, rho1, maximum(abs, magnetization), seconds,
)
