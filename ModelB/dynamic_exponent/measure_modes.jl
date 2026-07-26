using CUDA, JLD2, LinearAlgebra, Statistics, Printf

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

function mode_workspace(u)
    L, _, n = size(u)
    gx = similar(u, Float32, (L, 1, n))
    gy = similar(u, Float32, (1, L, n))
    px = zeros(Float32, L, n)
    py = similar(px)
    angle = Float32.(2π .* (0:L-1) ./ L)
    phase = vcat(cos.(angle)', -sin.(angle)', ones(Float32, 1, L))
    return (; gx, gy, px, py, phase,
            mx=zeros(Float32, 3, n), my=zeros(Float32, 3, n))
end

function observe!(modes, i, u, w)
    L, _, n = size(u)
    sum!(w.gx, u)
    sum!(w.gy, u)
    copyto!(w.px, reshape(w.gx, L, n))
    copyto!(w.py, reshape(w.gy, L, n))
    mul!(w.mx, w.phase, w.px)
    mul!(w.my, w.phase, w.py)
    scale = inv(Float32(L^2))
    @views modes[i, 1, :] .= complex.(w.mx[1, :], w.mx[2, :]) .* scale
    @views modes[i, 2, :] .= complex.(w.my[1, :], w.my[2, :]) .* scale
    return vec(@view(w.mx[3, :])) .* scale
end

function main(args)
    @assert length(args) == 6
    input = abspath(args[1])
    output = abspath(args[2])
    tmax = parse(Float64, args[3])
    sample_dt = parse(Float64, args[4])
    dt = parse(Float64, args[5])
    seed = parse(UInt64, args[6])
    @assert isfile(input) && !ispath(output)

    data = load(input)
    cfgs = data["cfgs"]
    L = Int(data["L"])
    kappa = Float64(data["kappa"])
    lambda = Float64(data["lambda"])
    n = size(cfgs, 3)
    @assert size(cfgs) == (L, L, n)
    @assert maximum(abs, vec(mean(cfgs; dims=(1, 2)))) < 1e-6
    ninterval = round(Int, tmax / sample_dt)
    nstep = round(Int, sample_dt / dt)
    @assert isapprox(ninterval * sample_dt, tmax; atol=1e-7, rtol=0)
    @assert isapprox(nstep * dt, sample_dt; atol=1e-7, rtol=0)

    u = CuArray{Float32}(cfgs)
    buffers = modelb_fixed_buffers(u; seed)
    w = mode_workspace(u)
    modelB_2d_fixed!(
        u, buffers; κ=kappa, λ=lambda, dt=dt, tspan=(0.0, dt),
    )
    copyto!(u, cfgs)
    seed_modelb_fixed!(buffers, seed)
    CUDA.synchronize()

    modes = Array{ComplexF32}(undef, ninterval + 1, 2, n)
    times = collect(0:ninterval) .* sample_dt
    m0 = observe!(modes, 1, u, w)
    max_delta_m = 0.0f0
    stride = max(1, ninterval ÷ 10)
    t0 = time_ns()
    for i in 1:ninterval
        modelB_2d_fixed!(
            u, buffers;
            κ=kappa, λ=lambda, dt=(times[2] - times[1]) / nstep,
            tspan=(0.0, times[2] - times[1]),
        )
        m = observe!(modes, i + 1, u, w)
        @assert all(isfinite, @view(modes[i + 1, :, :]))
        max_delta_m = max(max_delta_m, maximum(abs.(m .- m0)))
        i % stride == 0 && @printf(
            "t=%g/%g  max|ΔM|=%.3e\n", times[i + 1], times[end], max_delta_m,
        )
    end
    seconds = (time_ns() - t0) * 1e-9

    mkpath(dirname(output))
    jldsave(
        output;
        modes, times, L, Nensemble=n, kappa, lambda,
        dt=(times[2] - times[1]) / nstep,
        sample_dt=times[2] - times[1],
        tmax=times[end],
        noise_seed=seed, input, initial_magnetization=m0,
        max_abs_conservation_error=max_delta_m, seconds,
    )
    @printf(
        "saved %s  %.3f μs/step  max|ΔM|=%.3e\n",
        output, 1e6 * seconds / (ninterval * nstep), max_delta_m,
    )
end

main(ARGS)
