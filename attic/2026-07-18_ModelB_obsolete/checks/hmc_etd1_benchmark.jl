# HMC-stationarity benchmark for the spectral stochastic exponential-Euler
# (ETD1) candidate.  The cubic term is deliberately NOT dealiased: a 2/3 mask
# changes the finite-lattice HMC action and therefore has a different invariant
# distribution.

include(joinpath(@__DIR__, "hmc_equilibrium_benchmark.jl"))
using CUDA.CUFFT
using LinearAlgebra

function spectral_coefficients(L, kappa, lambda, dt)
    kvals = FT.(2pi .* (0:L-1) ./ L)
    kh1 = @. 2 * (1 - cos(kvals))
    kh2_host = [kh1[i] + kh1[j] for i in 1:L, j in 1:L]
    kh2 = reshape(CuArray(kh2_host), L, L, 1)
    m02 = FT(2 * (1 - 2lambda) - 8kappa)
    lop = @. -kh2 * (FT(2kappa) * kh2 + m02)
    ldt = lop .* FT(dt)
    lsafe = @. ifelse(abs(lop) < FT(1e-12), one(FT), lop)
    E = exp.(ldt)
    phi1 = @. ifelse(abs(ldt) < FT(1e-7), FT(dt), expm1(ldt) / lsafe)
    var = @. ifelse(abs(2ldt) < FT(1e-7),
                    FT(2) * kh2 * FT(dt),
                    FT(2) * kh2 * expm1(2ldt) / (2lsafe))
    noisec = sqrt.(max.(var, zero(FT)))
    nonlinear_k = FT(4lambda) .* kh2
    return E, phi1, noisec, nonlinear_k
end

function run_etd1(u0, kappa, lambda, tend, dt, seed)
    nsteps = round(Int, tend / dt)
    isapprox(nsteps * dt, tend; rtol=1e-5, atol=1e-6) ||
        error("ETD1 benchmark requires tend/dt to be integral")
    CUDA.seed!(seed)
    phi = CuArray(FT.(u0))
    L = size(phi, 1)
    E, phi1, noisec, nonlinear_k = spectral_coefficients(L, kappa, lambda, dt)

    real_work = similar(phi)
    eta = similar(phi)
    complex_in = similar(phi, ComplexF32)
    ph = similar(complex_in)
    cubeh = similar(complex_in)
    etah = similar(complex_in)
    pf = plan_fft(complex_in, (1, 2))
    pif = plan_ifft(complex_in, (1, 2))
    rng = CUDA.default_rng()

    CUDA.synchronize()
    wall = @elapsed begin
        for _ in 1:nsteps
            @. complex_in = ComplexF32(phi)
            mul!(ph, pf, complex_in)
            @. real_work = phi^3
            @. complex_in = ComplexF32(real_work)
            mul!(cubeh, pf, complex_in)
            randn!(rng, eta)
            @. complex_in = ComplexF32(eta)
            mul!(etah, pf, complex_in)
            @. ph = E * ph - phi1 * nonlinear_k * cubeh + noisec * etah
            mul!(complex_in, pif, ph)
            @. phi = real(complex_in)
        end
        CUDA.synchronize()
    end
    final = Array(phi)
    return final, (; wall, nsteps, finite=all(isfinite, final))
end

function etd_main(args=ARGS)
    outfile = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "hmc_etd1_results.tsv")
    L = length(args) >= 2 ? parse(Int, args[2]) : 16
    ncfg = length(args) >= 3 ? parse(Int, args[3]) : 2048
    tend = length(args) >= 4 ? parse(FT, args[4]) : FT(5)
    kappa, lambda = 0.2705, 0.022
    data = joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata",
                    "cfgs_wolff_fahmc_k=0.2705_l=0.022_$(L)^2.jld2")
    hmc = jldopen(data, "r") do f
        Array(f["cfgs"])
    end
    indices = round.(Int, range(1, size(hmc, 3); length=ncfg))
    u0 = FT.(hmc[:, :, indices])
    initial_obs = observables(u0, kappa, lambda)
    dts = FT.((0.005, 0.01, 0.02, 0.04, 0.05, 0.1))

    @printf("spectral ETD1 HMC stationarity: L=%d N=%d tend=%.2f dealias=false device=%s\n",
            L, ncfg, tend, CUDA.name(CUDA.device()))
    rows = NamedTuple[]
    for (i, dt) in enumerate(dts)
        # Same-shape preflight excludes FFT plan/kernel JIT from timing.
        run_etd1(u0, kappa, lambda, FT(2dt), dt, 12000 + i)
        result = try
            final, meta = run_etd1(u0, kappa, lambda, tend, dt, 13000 + i)
            final_obs = meta.finite ? observables(final, kappa, lambda) : initial_obs
            massmax = meta.finite ? maximum(abs.(final_obs[:mass] .- initial_obs[:mass])) : Inf
            summaries = meta.finite ? paired_summary(initial_obs, final_obs) : NamedTuple[]
            for s in summaries
                push!(rows, (; method="ETD1", dt=Float64(dt), tend=Float64(tend), ncfg,
                    meta.wall, meta.nsteps, meta.finite, massmax, s...))
            end
            a = only(filter(r -> r.observable == "action", summaries))
            @printf("dt=%6.3f steps=%4d wall=%.3fs finite=%s dS/S=%+.3e z=%+.1f mass=%.2e\n",
                    dt, meta.nsteps, meta.wall, meta.finite,
                    a.relative_delta, a.z, massmax)
            true
        catch err
            @warn "ETD1 failed" dt exception=(err, catch_backtrace())
            @printf("dt=%6.3f FAILED\n", dt)
            false
        end
        result || nothing
        CUDA.reclaim()
    end

    open(outfile, "w") do io
        println(io, "method\tdt\ttend\tncfg\twall\tnsteps\tfinite\tmassmax\tobservable\tinitial\tfinal\tdelta\tpaired_se\tz\trelative_delta")
        for r in rows
            println(io, join((getproperty(r, key) for key in propertynames(r)), '\t'))
        end
    end
    println("wrote ", abspath(outfile))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    etd_main()
end
