# Runtime verification of ModelB.jl: kernels/noise/SDE-setup copied verbatim from ../ModelB.jl
# (only changes: drop meta-package `DifferentialEquations`; CURAND.default_rng() -> CUDA.default_rng(),
#  since CURAND.default_rng no longer exists in this CUDA.jl version).
#
# Usage: julia modelB_runner.jl <outprefix> <L> <tend> <T1,T2,...>
# For each T: runs Model B from randn IC (mean-subtracted), saves per-frame observables
# (t, mean(u), <phi^2>, S(kmin)) and the final configuration.
using CUDA, StochasticDiffEq, DiffEqNoiseProcess, Statistics, Printf, DelimitedFiles, FFTW
const myT = Float32

limitbound(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end

function limitbound2(a, n)
    if a == n + 1
        1
    elseif a == 0
        n
    elseif a == n + 2
        2
    elseif a == -1
        n - 1
    else
        a
    end
end

function update_2d_modelB!(dσ, σ, fun, dx)
    N = size(σ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y = Tuple(cind[i])
        xp1, xm1 = limitbound2(x + 1, N), limitbound2(x - 1, N)
        yp1, ym1 = limitbound2(y + 1, N), limitbound2(y - 1, N)
        xp2, xm2 = limitbound2(x + 2, N), limitbound2(x - 2, N)
        yp2, ym2 = limitbound2(y + 2, N), limitbound2(y - 2, N)
        @inbounds dσ[x, y] =
            (
                -20 * σ[x, y] + 8 * σ[x, ym1] - σ[x, ym2] + 8 * σ[x, yp1] - σ[x, yp2] +
                8 * σ[xm1, y] - 2 * σ[xm1, ym1] - 2 * σ[xm1, yp1] - σ[xm2, y] -
                2 * (-4 * σ[xp1, y] + σ[xp1, ym1] + σ[xp1, yp1]) - σ[xp2, y]
            ) / dx^4 -
            (
                4 * fun(σ[x, y]) - fun(σ[xm1, y]) - fun(σ[xp1, y]) - fun(σ[x, yp1]) -
                fun(σ[x, ym1])
            ) / dx^2
    end
    return nothing
end

function update_conserve_noise_kernel!(
    dσ::AbstractArray{T,2},
    ξx::AbstractArray{T,2},
    ξy::AbstractArray{T,2},
    dx::T,
) where {T}
    N = size(dσ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(dσ)
    for i = id:blockDim().x*gridDim().x:prod(size(dσ))
        x, y = Tuple(cind[i])
        xm1 = limitbound(x - 1, N)
        ym1 = limitbound(y - 1, N)
        @inbounds dσ[x, y] =
            ((ξx[x, y] - ξx[xm1, y]) + (ξy[x, y] - ξy[x, ym1])) / dx
    end
    return nothing
end

function modelB_2d_loop_GPU(dσ, σ, fun, dx)
    threads = 512
    blocks = 2^5
    @cuda blocks = blocks threads = threads maxregs = 4 update_2d_modelB!(dσ, σ, fun, dx)
end

function conserve_noise_loop_GPU(dσ, ξx, ξy, dx)
    threads = 512
    blocks = 2^5
    @cuda blocks = blocks threads = threads maxregs = 4 update_conserve_noise_kernel!(dσ, ξx, ξy, dx)
end

function REAL_INPLACE_ModelB_NOISE_DIST(rand_vec, W, dt, u, p, t, rng)
    sqabsdt = @fastmath sqrt(abs(dt))
    p1, ξx, ξy = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξx)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξy)
    CUDA.@sync conserve_noise_loop_GPU(p1, ξx, ξy, 1.0f0)
    CUDA.@sync @. rand_vec = sqabsdt * p1
end

# Exact bridge for the conserved Gaussian noise: same covariance -∇², standard bridge formula
# with the white-noise sample replaced by a conserved-noise sample.
function REAL_INPLACE_ModelB_NOISE_BRIDGE(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    p1, ξx, ξy = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξx)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξy)
    CUDA.@sync conserve_noise_loop_GPU(p1, ξx, ξy, 1.0f0)
    CUDA.@sync @. rand_vec = sqrt((1 - q) * q * abs(h)) * p1 + q * Wh
end

function RealModelBProcess!(t0, W0, Z0 = nothing; kwargs...)
    NoiseProcess{true}(
        t0,
        W0,
        Z0,
        REAL_INPLACE_ModelB_NOISE_DIST,
        REAL_INPLACE_ModelB_NOISE_BRIDGE;
        kwargs...,
    )
end

function modelB_2d_SDE_prob(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    dx = 1.0f0,
    solver = EM(),
    save_start = false,
    save_everystep = false,
    save_end = false,
    abstol = 5e-2,
    reltol = 5e-2,
    with_Z = false,
    kwargs...,
)
    function Ufun(x)
        -x + x^3 / 6
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync modelB_2d_loop_GPU(dσ, σ, Ufun, dx)
        CUDA.@sync @. dσ = (1 / γ) * dσ
    end

    function g(du, u, p, t)
        du .= sqrt(2 * T / γ)
    end
    myp = (similar(u0_GPU), similar(u0_GPU), similar(u0_GPU))
    W_test = zero(u0_GPU)
    Z0 = with_Z ? zero(u0_GPU) : nothing
    W = RealModelBProcess!(0.0f0, W_test, Z0, save_everystep = false)
    sdeprob = SDEProblem(ODEfun_tex, g, u0_GPU, tspan, myp; noise = W)
    GC.gc(true)
    CUDA.reclaim()
    solve(
        sdeprob,
        solver;
        save_start = save_start,
        save_everystep = save_everystep,
        save_end = save_end,
        abstol = abstol,
        reltol = reltol,
        saveat = tspan[1]:(1f0):tspan[2],
        kwargs...,
    )
end

# ---------------- driver ----------------
function main(args = ARGS)
    outprefix = args[1]
    L = parse(Int, args[2])
    tend = parse(Float32, args[3])
    Ts = parse.(Float32, split(args[4], ","))
    dtval = length(args) >= 5 ? (args[5] in ("auto", "0") ? 0.0f0 : parse(Float32, args[5])) : 0.005f0
    solvername = length(args) >= 6 ? args[6] : "EM"
    solvers = Dict("EM" => EM(), "EMZ" => EM(), "LambaEM" => LambaEM(), "DRI1NM" => DRI1NM(),
        "DRI1" => DRI1(), "SOSRA" => SOSRA(), "SRA3" => SRA3(), "SRA1" => SRA1(),
        "EulerHeun" => EulerHeun())
    needs_Z = solvername in ("SOSRA", "SRA3", "SRA1", "EMZ")
    tolval = length(args) >= 7 ? parse(Float32, args[7]) : 1.0f-2

    for T in Ts
        uini = randn(Float32, L, L)
        uini .-= mean(uini)   # exactly zero conserved magnetization
        dtkw = dtval > 0 ? (dt = dtval,) : (dtmin = 1.0f-9, force_dtmin = true)
        t_run = @elapsed sol = modelB_2d_SDE_prob(;
            u0 = uini, γ = 1.0f0, tspan = (0.0f0, tend), T = T,
            solver = solvers[solvername], save_start = true,
            with_Z = needs_Z, abstol = tolval, reltol = tolval, dtkw...,
        )
        frames = [Array(u) for u in sol.u]
        ts = sol.t
        tag = @sprintf("%sT%.3f_L%d", outprefix, T, L)
        open(tag * "_obs.dat", "w") do io
            println(io, "# t  mean(u)  <phi^2>  S(kmin)")
            for (t, fr) in zip(ts, frames)
                F = fft(Float64.(fr))
                Skmin = (abs2(F[2, 1]) + abs2(F[1, 2])) / (2 * L^2)
                @printf(io, "%.2f  %.6e  %.6f  %.6f\n", t, mean(fr), mean(abs2, fr), Skmin)
            end
        end
        writedlm(tag * "_final.dat", frames[end])
        # axis-averaged structure factor S(k) over the second half of the trajectory
        nh = length(frames) ÷ 2
        Sacc = zeros(L ÷ 2)
        for fr in frames[nh+1:end]
            F = fft(Float64.(fr))
            for n in 1:L÷2
                Sacc[n] += (abs2(F[n+1, 1]) + abs2(F[1, n+1])) / (2 * L^2)
            end
        end
        Sacc ./= (length(frames) - nh)
        open(tag * "_Sk.dat", "w") do io
            println(io, "# k  S(k)   (axis-averaged, time-averaged over second half)")
            for n in 1:L÷2
                @printf(io, "%.6f  %.6f\n", 2pi * n / L, Sacc[n])
            end
        end
        consdrift = maximum(abs.([mean(fr) for fr in frames] .- mean(frames[1])))
        p2 = mean([mean(abs2, fr) for fr in frames[nh+1:end]])
        @printf("%s T=%.3f L=%d dt0=%g tol=%g: %.1fs, retcode=%s, steps=%d(+%drej), cons-drift=%.3e, <phi^2>(late)=%.4f, S(kmin)(late)=%.4f\n",
            solvername, T, L, dtval, tolval, t_run, sol.retcode,
            sol.stats.naccept, sol.stats.nreject, consdrift, p2, Sacc[1])
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
