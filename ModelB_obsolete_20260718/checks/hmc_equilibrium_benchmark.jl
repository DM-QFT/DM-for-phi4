# HMC-stationarity work/precision benchmark for the conserved 2D phi^4 SDE.
#
# The input configurations are distributed as exp(-S), where
#   S = sum_x[-2kappa sum_mu phi_x phi_(x+mu)
#             + (1-2lambda)phi_x^2 + lambda phi_x^4].
# At T=1 the exact Model-B semigroup leaves this distribution invariant.  Each
# trajectory keeps the magnetization of its own HMC initial state, so the full
# canonical HMC ensemble is the correct mixture of conserved sectors.  Never
# subtract the per-configuration mean in this test.
#
# Usage (post-JIT GPU timings, default L=16, N=2048, t=2):
#   julia --startup-file=no checks/hmc_equilibrium_benchmark.jl \
#       checks/hmc_equilibrium_L16.tsv 16 2048 2.0

using CUDA
using DifferentialEquations
using DiffEqNoiseProcess
using JLD2
using Printf
using Random
using Statistics
using StochasticDiffEq

const FT = Float32

@inline periodic_minus_one(i, L) = i == 1 ? L : i - 1
@inline periodic_plus_one(i, L) = i == L ? 1 : i + 1
@inline periodic_minus_two(i, L) = i <= 2 ? L + i - 2 : i - 2
@inline periodic_plus_two(i, L) = i >= L - 1 ? i - L + 2 : i + 2

"""Batched (L,L,B) drift: Delta[-2kappa Delta phi + m02 phi + 4lambda phi^3]."""
function hmc_modelb_drift_kernel!(du, u, two_kappa, m02, c3)
    L = size(u, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    inds = CartesianIndices(u)
    @inbounds for linear in id:stride:length(u)
        x, y, b = Tuple(inds[linear])
        xm1, xp1 = periodic_minus_one(x, L), periodic_plus_one(x, L)
        ym1, yp1 = periodic_minus_one(y, L), periodic_plus_one(y, L)
        xm2, xp2 = periodic_minus_two(x, L), periodic_plus_two(x, L)
        ym2, yp2 = periodic_minus_two(y, L), periodic_plus_two(y, L)

        # -Delta^2 phi (13-point periodic stencil)
        minus_bih = -20u[x, y, b] +
            8u[x, ym1, b] - u[x, ym2, b] +
            8u[x, yp1, b] - u[x, yp2, b] +
            8u[xm1, y, b] - 2u[xm1, ym1, b] - 2u[xm1, yp1, b] - u[xm2, y, b] +
            8u[xp1, y, b] - 2u[xp1, ym1, b] - 2u[xp1, yp1, b] - u[xp2, y, b]

        uc = u[x, y, b]
        fxm = m02 * u[xm1, y, b] + c3 * u[xm1, y, b]^3
        fxp = m02 * u[xp1, y, b] + c3 * u[xp1, y, b]^3
        fym = m02 * u[x, ym1, b] + c3 * u[x, ym1, b]^3
        fyp = m02 * u[x, yp1, b] + c3 * u[x, yp1, b]^3
        fc = m02 * uc + c3 * uc^3
        du[x, y, b] = two_kappa * minus_bih + fxm + fxp + fym + fyp - 4fc
    end
    return nothing
end

"""Batched link divergence; its covariance is -Delta independently in each replica."""
function conserved_noise_kernel!(out, edge_x, edge_y)
    L = size(out, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    inds = CartesianIndices(out)
    @inbounds for linear in id:stride:length(out)
        x, y, b = Tuple(inds[linear])
        xm1, ym1 = periodic_minus_one(x, L), periodic_minus_one(y, L)
        out[x, y, b] = edge_x[x, y, b] - edge_x[xm1, y, b] +
                       edge_y[x, y, b] - edge_y[x, ym1, b]
    end
    return nothing
end

function launch_drift!(du, u, two_kappa, m02, c3)
    threads = 256
    blocks = min(256, cld(length(u), threads))
    @cuda threads=threads blocks=blocks hmc_modelb_drift_kernel!(du, u, two_kappa, m02, c3)
    return nothing
end

function launch_noise!(out, edge_x, edge_y)
    threads = 256
    blocks = min(256, cld(length(out), threads))
    @cuda threads=threads blocks=blocks conserved_noise_kernel!(out, edge_x, edge_y)
    return nothing
end

function noise_distribution!(rand_vec, W, dt, u, p, t, rng)
    work, edge_x, edge_y = p.noise_work
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), edge_x)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), edge_y)
    launch_noise!(work, edge_x, edge_y)
    sqrth = sqrt(abs(dt))
    @. rand_vec = sqrth * work
    return nothing
end

# Conditional Gaussian bridge for the same conserved spatial covariance.
function noise_bridge!(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    work, edge_x, edge_y = p.noise_work
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), edge_x)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), edge_y)
    launch_noise!(work, edge_x, edge_y)
    bridge_sigma = sqrt((1 - q) * q * abs(h))
    @. rand_vec = bridge_sigma * work + q * Wh
    return nothing
end

function conserved_process(t0, W0, Z0=nothing; kwargs...)
    return NoiseProcess{true}(
        t0, W0, Z0, noise_distribution!, noise_bridge!; kwargs...
    )
end

function make_problem(u0, kappa, lambda, tend; with_z=false)
    u = CuArray(FT.(u0))
    m02 = FT(2 * (1 - 2lambda) - 8kappa)
    c3 = FT(4lambda)
    two_kappa = FT(2kappa)
    noise_work = (similar(u), similar(u), similar(u))
    params = (; two_kappa, m02, c3, noise_work)

    function drift!(du, state, p, t)
        launch_drift!(du, state, p.two_kappa, p.m02, p.c3)
        return nothing
    end
    function diffusion!(du, state, p, t)
        fill!(du, sqrt(FT(2)))
        return nothing
    end

    W0 = zero(u)
    Z0 = with_z ? zero(u) : nothing
    noise = conserved_process(FT(0), W0, Z0; save_everystep=false)
    return SDEProblem(drift!, diffusion!, u, (FT(0), FT(tend)), params; noise)
end

const ROCK_RHO = 128.0
const ROCK_ESTIMATE = integ -> (integ.eigen_est = ROCK_RHO)

function method_specs()
    return vcat(
        [("EM", EM(), false, dt) for dt in FT.((0.0025, 0.005, 0.01, 0.02, 0.04))],
        [("SOSRA", SOSRA(), true, dt)
         for dt in FT.((0.005, 0.0075, 0.01, 0.015, 0.02, 0.04))],
        [("SOSRA2", SOSRA2(), true, dt) for dt in FT.((0.01, 0.02, 0.04))],
        [("SROCK2", SROCK2(eigen_est=ROCK_ESTIMATE), false, dt)
         for dt in FT.((0.01, 0.02, 0.04))],
        [("LambaEM", LambaEM(), false, dt) for dt in FT.((0.005, 0.01))],
        [("DRI1NM-invalid", DRI1NM(), false, FT(0.005))],
    )
end

function run_once(u0, kappa, lambda, tend, alg, with_z, dt, seed)
    CUDA.seed!(seed)
    prob = make_problem(u0, kappa, lambda, tend; with_z)
    CUDA.synchronize()
    sol = nothing
    wall = @elapsed begin
        sol = solve(
            prob, alg; dt, adaptive=false, save_start=false,
            save_everystep=false, save_end=true, dense=false,
            maxiters=typemax(Int)
        )
        CUDA.synchronize()
    end
    final = Array(sol.u[end])
    return final, (;
        wall,
        retcode=string(sol.retcode),
        finite=all(isfinite, final),
        naccept=sol.stats.naccept,
        nreject=sol.stats.nreject,
    )
end

"""Per-configuration observables; zero-mode observables are diagnostic only."""
function observables(x, kappa, lambda)
    L, L2, batch = size(x)
    L == L2 || error("square lattice required")
    V = L^2
    names = (:action, :phi2, :phi4, :nn, :sd, :mu2,
             :skmin, :sk2, :skdiag, :skmid, :mass, :maxabs)
    out = Dict(name => zeros(Float64, batch) for name in names)
    phase_min = cis.(-2pi .* (0:L-1) ./ L)
    phase_2 = cis.(-4pi .* (0:L-1) ./ L)
    nmid = max(1, L ÷ 4)
    phase_mid = cis.(-2pi * nmid .* (0:L-1) ./ L)

    Threads.@threads for b in 1:batch
        action = 0.0
        p2 = 0.0
        p4 = 0.0
        nn = 0.0
        sd = 0.0
        mu2 = 0.0
        mass = 0.0
        maxabs = 0.0
        fx_min = 0.0im
        fy_min = 0.0im
        fx_2 = 0.0im
        fy_2 = 0.0im
        fdiag_pp = 0.0im
        fdiag_pm = 0.0im
        fx_mid = 0.0im
        fy_mid = 0.0im
        @inbounds for y in 1:L, xx in 1:L
            v = Float64(x[xx, y, b])
            xp = xx == L ? 1 : xx + 1
            xm = xx == 1 ? L : xx - 1
            yp = y == L ? 1 : y + 1
            ym = y == 1 ? L : y - 1
            nbr_fwd = Float64(x[xp, y, b]) + Float64(x[xx, yp, b])
            nbr_all = nbr_fwd + Float64(x[xm, y, b]) + Float64(x[xx, ym, b])
            v2 = v * v
            v4 = v2 * v2
            force = -2kappa * nbr_all + 2(1 - 2lambda) * v + 4lambda * v^3
            action += -2kappa * v * nbr_fwd + (1 - 2lambda) * v2 + lambda * v4
            p2 += v2
            p4 += v4
            nn += 0.5 * v * nbr_fwd
            sd += v * force
            mu2 += force * force
            mass += v
            maxabs = max(maxabs, abs(v))
            fx_min += v * phase_min[xx]
            fy_min += v * phase_min[y]
            fx_2 += v * phase_2[xx]
            fy_2 += v * phase_2[y]
            fdiag_pp += v * phase_min[xx] * phase_min[y]
            fdiag_pm += v * phase_min[xx] * conj(phase_min[y])
            fx_mid += v * phase_mid[xx]
            fy_mid += v * phase_mid[y]
        end
        out[:action][b] = action / V
        out[:phi2][b] = p2 / V
        out[:phi4][b] = p4 / V
        out[:nn][b] = nn / V
        out[:sd][b] = sd / V
        out[:mu2][b] = mu2 / V
        out[:skmin][b] = (abs2(fx_min) + abs2(fy_min)) / (2V)
        out[:sk2][b] = (abs2(fx_2) + abs2(fy_2)) / (2V)
        out[:skdiag][b] = (abs2(fdiag_pp) + abs2(fdiag_pm)) / (2V)
        out[:skmid][b] = (abs2(fx_mid) + abs2(fy_mid)) / (2V)
        out[:mass][b] = mass / V
        out[:maxabs][b] = maxabs
    end
    return out
end

function paired_summary(initial_obs, final_obs)
    rows = NamedTuple[]
    for name in sort!(collect(keys(initial_obs)))
        before = initial_obs[name]
        after = final_obs[name]
        delta = after .- before
        n = length(delta)
        dmean = mean(delta)
        # Preserve HMC chain order and estimate uncertainty from block means.
        # This remains conservative when consecutive source configurations are
        # autocorrelated; it also avoids treating Z2 partners as extra samples.
        nblocks = min(64, n)
        blocksize = n ÷ nblocks
        blockmeans = [mean(@view delta[(i - 1) * blocksize + 1:i * blocksize])
                      for i in 1:nblocks]
        dse = nblocks > 1 ? std(blockmeans) / sqrt(nblocks) : NaN
        base = mean(before)
        rel = dmean / max(abs(base), eps())
        z = dmean / max(dse, eps())
        push!(rows, (; observable=String(name), initial=base, final=mean(after),
                     delta=dmean, paired_se=dse, z, relative_delta=rel))
    end
    return rows
end

function warmup!(specs, kappa, lambda)
    u = zeros(FT, 8, 8, 4)
    for (_, alg, with_z, dt) in specs
        try
            run_once(u, kappa, lambda, max(2dt, FT(0.01)), alg, with_z, dt, 9001)
        catch err
            @warn "warmup failed" alg dt exception=(err, catch_backtrace())
        end
    end
    CUDA.reclaim()
    return nothing
end

function main(args=ARGS)
    outfile = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "hmc_equilibrium_results.tsv")
    L = length(args) >= 2 ? parse(Int, args[2]) : 16
    ncfg = length(args) >= 3 ? parse(Int, args[3]) : 2048
    tend = length(args) >= 4 ? parse(FT, args[4]) : FT(2)
    kappa = length(args) >= 5 ? parse(Float64, args[5]) : 0.2705
    lambda = length(args) >= 6 ? parse(Float64, args[6]) : 0.022
    data = length(args) >= 7 ? args[7] : joinpath(
        @__DIR__, "..", "..", "2Dphi4", "trainingdata",
        "cfgs_wolff_fahmc_k=$(kappa)_l=$(lambda)_$(L)^2.jld2"
    )

    hmc = jldopen(data, "r") do f
        Int(f["N"]) == L || error("HMC L metadata mismatch")
        file_kappa = haskey(f, "kappa") ? f["kappa"] : f["\u03ba"]
        file_lambda = haskey(f, "lambda") ? f["lambda"] : f["\u03bb"]
        isapprox(Float64(file_kappa), kappa; atol=0, rtol=1e-12) || error("kappa mismatch")
        isapprox(Float64(file_lambda), lambda; atol=0, rtol=1e-12) || error("lambda mismatch")
        Array(f["cfgs"])
    end
    ncfg <= size(hmc, 3) || error("requested more configurations than HMC file contains")
    # Even coverage of the Markov-chain order; importantly, do not remove the zero mode.
    indices = round.(Int, range(1, size(hmc, 3); length=ncfg))
    u0 = FT.(hmc[:, :, indices])
    initial_obs = observables(u0, kappa, lambda)
    specs = method_specs()

    @printf("HMC stationarity: L=%d N=%d kappa=%.6f lambda=%.6f T=1 tend=%.3f device=%s\n",
        L, ncfg, kappa, lambda, tend, CUDA.name(CUDA.device()))
    @printf("HMC source: %s\n", abspath(data))
    warmup!(specs, kappa, lambda)

    allrows = NamedTuple[]
    for (runid, (name, alg, with_z, dt)) in enumerate(specs)
        result = try
            # A same-shape preflight prevents lazy GPU/solver compilation from
            # contaminating the measured production solve.
            run_once(u0, kappa, lambda, max(2dt, FT(0.01)), alg, with_z, dt,
                     5000 + runid)
            final, meta = run_once(u0, kappa, lambda, tend, alg, with_z, dt, 10000 + runid)
            final_obs = meta.finite ? observables(final, kappa, lambda) : initial_obs
            mass_max = meta.finite ? maximum(abs.(final_obs[:mass] .- initial_obs[:mass])) : Inf
            summaries = meta.finite ? paired_summary(initial_obs, final_obs) : NamedTuple[]
            for s in summaries
                push!(allrows, (; method=name, dt=Float64(dt), tend=Float64(tend), ncfg,
                    wall=meta.wall, retcode=meta.retcode, finite=meta.finite,
                    mass_max, naccept=meta.naccept, nreject=meta.nreject, s...))
            end
            @printf("%-16s dt=%7.4f wall=%7.3fs ret=%-10s mass=%.2e",
                name, dt, meta.wall, meta.retcode, mass_max)
            if meta.finite
                arow = only(filter(r -> r.observable == "action", summaries))
                p2row = only(filter(r -> r.observable == "phi2", summaries))
                @printf("  dS/S=%+.3e (z=%+.1f) dphi2/phi2=%+.3e (z=%+.1f)\n",
                    arow.relative_delta, arow.z, p2row.relative_delta, p2row.z)
            else
                println()
            end
            (; final, meta)
        catch err
            @warn "solver run failed" name dt exception=(err, catch_backtrace())
            @printf("%-16s dt=%7.4f EXCEPTION\n", name, dt)
            nothing
        end
        flush(stdout)
        result === nothing || nothing
        CUDA.reclaim()
    end

    open(outfile, "w") do io
        println(io, "method\tdt\ttend\tncfg\twall\tretcode\tfinite\tmass_max\tnaccept\tnreject\tobservable\tinitial\tfinal\tdelta\tpaired_se\tz\trelative_delta")
        for r in allrows
            println(io, join((getproperty(r, key) for key in propertynames(r)), '\t'))
        end
    end
    println("wrote ", abspath(outfile))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
