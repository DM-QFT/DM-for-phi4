# Paired work-precision benchmark for the 2D Model-B SPDE near criticality.
#
# The fine link-noise increments are shared by every method.  Coarse W increments
# are exact sums.  For SRA methods, the auxiliary increment is reconstructed so
# that the space-time integral I_(1,0) is also identical across refinements.
# This makes the field error a genuine strong error instead of a difference of
# unrelated stochastic trajectories.
#
# Usage:
#   julia -t auto critical_paired_benchmark.jl [outfile] [L] [T] [tend] [npaths] [href]

using Random
using Statistics
using Printf
using FFTW
using StochasticDiffEq
using DiffEqNoiseProcess

const SQRT3 = sqrt(3.0)

@inline prev(i, L) = i == 1 ? L : i - 1
@inline next(i, L) = i == L ? 1 : i + 1
@inline prev2(i, L) = i <= 2 ? L + i - 2 : i - 2
@inline next2(i, L) = i >= L - 1 ? i - L + 2 : i + 2
@inline potential_prime(x) = -x + x^3 / 6

"""CPU version of the production 13-point drift stencil (dx=gamma=1)."""
function drift!(du, u, p, t)
    L = size(u, 1)
    @inbounds for y in 1:L, x in 1:L
        xm1, xp1 = prev(x, L), next(x, L)
        ym1, yp1 = prev(y, L), next(y, L)
        xm2, xp2 = prev2(x, L), next2(x, L)
        ym2, yp2 = prev2(y, L), next2(y, L)
        bih = -20u[x, y] + 8u[x, ym1] - u[x, ym2] + 8u[x, yp1] - u[x, yp2] +
              8u[xm1, y] - 2u[xm1, ym1] - 2u[xm1, yp1] - u[xm2, y] +
              8u[xp1, y] - 2u[xp1, ym1] - 2u[xp1, yp1] - u[xp2, y]
        lapV = -4potential_prime(u[x, y]) + potential_prime(u[xm1, y]) +
               potential_prime(u[xp1, y]) + potential_prime(u[x, ym1]) +
               potential_prime(u[x, yp1])
        du[x, y] = bih + lapV
    end
    return nothing
end

function conserved_increment!(out, edge_x, edge_y, sqrth)
    L = size(out, 1)
    @inbounds for y in 1:L, x in 1:L
        out[x, y] = sqrth * (
            edge_x[x, y] - edge_x[prev(x, L), y] +
            edge_y[x, y] - edge_y[x, prev(y, L)]
        )
    end
    return out
end

"""Generate fine increments for W and the independent SRA auxiliary Gaussian."""
function fine_noise(L, nsteps, h, rng)
    dW = Array{Float64}(undef, L, L, nsteps)
    dZ = similar(dW)
    ex = Matrix{Float64}(undef, L, L)
    ey = similar(ex)
    sqrth = sqrt(h)
    for n in 1:nsteps
        randn!(rng, ex); randn!(rng, ey)
        conserved_increment!(@view(dW[:, :, n]), ex, ey, sqrth)
        randn!(rng, ex); randn!(rng, ey)
        conserved_increment!(@view(dZ[:, :, n]), ex, ey, sqrth)
    end
    return dW, dZ
end

"""
Build cumulative W/Z grids at step H=m*h.

For a fine substep j, I_j = h/2 (dW_j + dZ_j/sqrt(3)).  Translation of
the iterated integral to the coarse interval gives
I_H = sum_j [I_j + (m-1-j) h dW_j], hence
dZ_H = sqrt(3) (2 I_H/H - dW_H).
"""
function aggregate_grids(dWfine, dZfine, h, m; need_z)
    L = size(dWfine, 1)
    nfine = size(dWfine, 3)
    nfine % m == 0 || error("coarse step must divide the fine grid")
    ncoarse = nfine ÷ m
    H = m * h
    times = collect(range(0.0; step = H, length = ncoarse + 1))
    Wvals = Vector{Matrix{Float64}}(undef, ncoarse + 1)
    Zvals = need_z ? Vector{Matrix{Float64}}(undef, ncoarse + 1) : nothing
    curW = zeros(Float64, L, L)
    curZ = need_z ? zeros(Float64, L, L) : nothing
    Wvals[1] = copy(curW)
    need_z && (Zvals[1] = copy(curZ))
    dW = similar(curW)
    integ = similar(curW)
    dZ = need_z ? similar(curW) : nothing
    for n in 1:ncoarse
        fill!(dW, 0.0)
        need_z && fill!(integ, 0.0)
        firstj = (n - 1) * m + 1
        @inbounds for ell in 0:(m - 1)
            j = firstj + ell
            @views dW .+= dWfine[:, :, j]
            if need_z
                @views @. integ += (h / 2) * (
                    dWfine[:, :, j] + dZfine[:, :, j] / SQRT3
                ) + (m - 1 - ell) * h * dWfine[:, :, j]
            end
        end
        curW .+= dW
        Wvals[n + 1] = copy(curW)
        if need_z
            @. dZ = SQRT3 * (2integ / H - dW)
            curZ .+= dZ
            Zvals[n + 1] = copy(curZ)
        end
    end
    return times, Wvals, Zvals
end

function solve_on_grid(u0, temperature, alg, dt, times, Wvals, Zvals)
    noise = NoiseGrid(times, Wvals, Zvals)
    sigma = sqrt(2temperature)
    function diffusion!(du, u, p, t)
        fill!(du, sigma)
        return nothing
    end
    prob = SDEProblem(drift!, diffusion!, copy(u0), (times[1], times[end]); noise)
    elapsed = @elapsed sol = solve(
        prob, alg; dt, adaptive = false, save_start = false,
        save_everystep = false, save_end = true, dense = false
    )
    return Array(sol.u[end]), elapsed, sol.retcode
end

function energy_density(u)
    L = size(u, 1)
    total = 0.0
    @inbounds for y in 1:L, x in 1:L
        ux = u[next(x, L), y] - u[x, y]
        uy = u[x, next(y, L)] - u[x, y]
        v = u[x, y]
        total += 0.5 * (ux^2 + uy^2) - 0.5v^2 + v^4 / 24
    end
    return total / length(u)
end

function structure_min(u)
    L = size(u, 1)
    F = fft(u)
    return (abs2(F[2, 1]) + abs2(F[1, 2])) / (2L^2)
end

function error_metrics(u, ref, mean0)
    L = size(u, 1)
    err = u .- ref
    rmse = sqrt(mean(abs2, err))
    relrmse = rmse / sqrt(mean(abs2, ref))
    F = fft(err)
    q = [4sinpi((i - 1) / L)^2 + 4sinpi((j - 1) / L)^2 for i in 1:L, j in 1:L]
    q[1, 1] = Inf
    hminus1 = sqrt(sum(abs2.(F) ./ q)) / L^2
    lowmode = sqrt((abs2(F[2, 1]) + abs2(F[1, 2])) / (2L^2))
    masserr = abs(mean(u) - mean0)
    p2rel = abs(mean(abs2, u) - mean(abs2, ref)) / mean(abs2, ref)
    erel = abs(energy_density(u) - energy_density(ref)) /
           max(abs(energy_density(ref)), eps())
    skrel = abs(structure_min(u) - structure_min(ref)) /
            max(structure_min(ref), eps())
    return (; rmse, relrmse, hminus1, lowmode, masserr, p2rel, erel, skrel)
end

function method_specs(href)
    fixed_rho = integ -> (integ.eigen_est = 256.0)
    return vcat(
        [("EM", EM(), false, dt) for dt in (0.0025, 0.005, 0.01)],
        [("SOSRA", SOSRA(), true, dt) for dt in (0.0025, 0.005, 0.01, 0.02)],
        [("SOSRA2", SOSRA2(), true, dt) for dt in (0.005, 0.01, 0.02)],
        [("SROCK2", SROCK2(eigen_est = fixed_rho), false, dt) for dt in (0.01, 0.02, 0.04, 0.08)],
        [("DRI1NM-invalid", DRI1NM(), false, 0.005)],
    )
end

function run_path(path, L, temperature, tend, href, specs)
    nsteps = round(Int, tend / href)
    isapprox(nsteps * href, tend; atol = 100eps(tend)) || error("tend/href must be integral")
    rng = Xoshiro(0x4d6f64656c42 + path)
    u0 = randn(rng, L, L)
    u0 .-= mean(u0)
    dWfine, dZfine = fine_noise(L, nsteps, href, rng)

    tref, Wref, Zref = aggregate_grids(dWfine, dZfine, href, 1; need_z = true)
    reference, _, refcode = solve_on_grid(u0, temperature, SOSRA(), href, tref, Wref, Zref)
    refcode == ReturnCode.Success || error("reference solve failed: $refcode")

    rows = NamedTuple[]
    for (name, alg, need_z, dt) in specs
        m = round(Int, dt / href)
        isapprox(m * href, dt; atol = 100eps(dt)) || continue
        times, Wvals, Zvals = aggregate_grids(dWfine, dZfine, href, m; need_z)
        u, elapsed, retcode = solve_on_grid(u0, temperature, alg, dt, times, Wvals, Zvals)
        finite = all(isfinite, u)
        metrics = finite ? error_metrics(u, reference, mean(u0)) :
            (rmse = Inf, relrmse = Inf, hminus1 = Inf, lowmode = Inf,
             masserr = Inf, p2rel = Inf, erel = Inf, skrel = Inf)
        push!(rows, (; path, method = name, dt, elapsed, retcode = string(retcode),
                     finite, metrics...))
    end

    # A direct refinement check on the reference method.
    if 2href <= tend
        times, Wvals, Zvals = aggregate_grids(dWfine, dZfine, href, 2; need_z = true)
        u, elapsed, retcode = solve_on_grid(u0, temperature, SOSRA(), 2href, times, Wvals, Zvals)
        metrics = error_metrics(u, reference, mean(u0))
        push!(rows, (; path, method = "SOSRA-refcheck", dt = 2href, elapsed,
                     retcode = string(retcode), finite = all(isfinite, u), metrics...))
    end
    return rows
end

function summarize(rows)
    keys_sorted = sort(unique((r.method, r.dt) for r in rows); by = x -> (x[1], x[2]))
    println("\n# paired strong-error summary (mean +/- standard error across paths)")
    @printf("%-18s %9s %12s %10s %12s %12s %12s %10s %12s\n",
        "method", "dt", "rel_RMSE", "rel_SE", "H-1_RMSE", "low-k_RMSE",
        "rel_phi2", "phi2_SE", "mass_max")
    for key in keys_sorted
        rs = filter(r -> (r.method, r.dt) == key, rows)
        vals(field) = Float64[getproperty(r, field) for r in rs]
        pm(v) = (mean(v), length(v) > 1 ? std(v) / sqrt(length(v)) : NaN)
        rr, rrse = pm(vals(:relrmse)); hm, _ = pm(vals(:hminus1))
        lm, _ = pm(vals(:lowmode)); p2, p2se = pm(vals(:p2rel))
        massmax = maximum(vals(:masserr))
        @printf("%-18s %9.6f %12.4e %10.2e %12.4e %12.4e %12.4e %10.2e %12.4e\n",
            key[1], key[2], rr, rrse, hm, lm, p2, p2se, massmax)
    end
end

function main(args = ARGS)
    outfile = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "critical_paired_results.tsv")
    L = length(args) >= 2 ? parse(Int, args[2]) : 16
    temperature = length(args) >= 3 ? parse(Float64, args[3]) : 4.45
    tend = length(args) >= 4 ? parse(Float64, args[4]) : 2.0
    npaths = length(args) >= 5 ? parse(Int, args[5]) : 16
    href = length(args) >= 6 ? parse(Float64, args[6]) : 0.000625
    specs = method_specs(href)

    @printf("paired benchmark: L=%d T=%.4f tend=%.3f paths=%d href=%.7f threads=%d\n",
        L, temperature, tend, npaths, href, Threads.nthreads())

    # Compile all solver paths before the measured path ensemble.
    warm_rng = Xoshiro(7)
    warmW, warmZ = fine_noise(4, 256, href, warm_rng)
    warmu = zeros(4, 4)
    for (_, alg, need_z, dt) in specs
        m = max(1, round(Int, dt / href))
        m > size(warmW, 3) && continue
        nusable = size(warmW, 3) - size(warmW, 3) % m
        t, W, Z = aggregate_grids(@view(warmW[:, :, 1:nusable]),
                                  @view(warmZ[:, :, 1:nusable]), href, m; need_z)
        try
            solve_on_grid(warmu, temperature, alg, m * href, t, W, Z)
        catch
            # The real run records method failures; warmup must not mask them.
        end
    end

    per_path = Vector{Vector{NamedTuple}}(undef, npaths)
    Threads.@threads :static for path in 1:npaths
        per_path[path] = run_path(path, L, temperature, tend, href, specs)
        @printf("path %d/%d complete\n", path, npaths)
    end
    rows = reduce(vcat, per_path)

    open(outfile, "w") do io
        println(io, "path\tmethod\tdt\telapsed\tretcode\tfinite\trmse\trelrmse\thminus1\tlowmode\tmasserr\tp2rel\terel\tskrel")
        for r in rows
            println(io, join((getproperty(r, k) for k in propertynames(r)), '\t'))
        end
    end
    summarize(rows)
    println("wrote ", abspath(outfile))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
