using CUDA, JLD2, Statistics, Printf

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

# 保留实空间历史，避免在线投影预先限制动量与采样筛选。
@assert length(ARGS) == 7
input = abspath(ARGS[1])
output = abspath(ARGS[2])
nuse = parse(Int, ARGS[3])
tmax = parse(Float64, ARGS[4])
sample_dt = parse(Float64, ARGS[5])
dt = parse(Float64, ARGS[6])
seed = parse(UInt64, ARGS[7])
@assert isfile(input) && !ispath(output)

data = load(input)
cfgs = data["cfgs"][:, :, 1:nuse]
L = Int(data["L"])
kappa = Float64(data["kappa"])
lambda = Float64(data["lambda"])
ninterval = round(Int, tmax / sample_dt)
nstep = round(Int, sample_dt / dt)
@assert isapprox(ninterval * sample_dt, tmax; atol=1e-7, rtol=0)
@assert isapprox(nstep * dt, sample_dt; atol=1e-7, rtol=0)

u = CuArray{Float32}(cfgs)
buffers = modelb_fixed_buffers(u; seed)
fields = Array{Float32}(undef, L, L, nuse, ninterval + 1)
fields[:, :, :, 1] .= cfgs
host = Array{Float32}(undef, L, L, nuse)
t0 = time_ns()
stride = max(1, ninterval ÷ 10)
for i in 1:ninterval
    modelB_2d_fixed!(u, buffers; κ=kappa, λ=lambda, dt, tspan=(0.0, sample_dt))
    copyto!(host, u)
    @views fields[:, :, :, i + 1] .= host
    i % stride == 0 && @printf("t=%g/%g\n", i * sample_dt, tmax)
end
CUDA.synchronize()
seconds = (time_ns() - t0) * 1e-9

m0 = vec(mean(Float64, cfgs; dims=(1, 2)))
m = dropdims(mean(Float64, fields; dims=(1, 2)); dims=(1, 2))
mag_drift = maximum(abs, m .- reshape(m0, :, 1))
@assert all(isfinite, @view(fields[:, :, :, end]))
jldsave(output;
    fields, L, kappa, lambda, Nensemble=nuse,
    times=collect(0:ninterval) .* sample_dt,
    dt, sample_dt, tmax, noise_seed=seed, input,
    max_abs_conservation_error=mag_drift, seconds)
@printf("saved %s  %.2f GB  %.2f μs/step  max|ΔM|=%.3e\n",
    output, sizeof(fields) / 2^30, 1e6 * seconds / (ninterval * nstep), mag_drift)
