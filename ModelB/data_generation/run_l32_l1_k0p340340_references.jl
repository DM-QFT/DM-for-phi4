using CUDA, JLD2, NPZ, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L, Nensemble = 32, 1024
κ, λ, γ, T, dt, Δt = 0.340340f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
run = normpath(joinpath(
    @__DIR__, "..", "runs",
    "modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip",
))
chains = (
    (
        "hmc",
        joinpath(
            run, "samples", "hmc_chain_ep10000_N1024_seed20260901",
            "hmc_ep10000_t0_N1024.npz",
        ),
        UInt64(20261101),
    ),
    (
        "gaussian",
        joinpath(
            run, "samples", "gaussian_chain_ep10000_N1024_seed20261001",
            "gaussian_ep10000_t0_N1024_seed20260721.npz",
        ),
        UInt64(20261201),
    ),
)

magnetization(cfgs) = vec(mean(Float64, cfgs; dims=(1, 2)))
for (initial, source, seed) in chains
    data = npzread(source)
    cfgs = permutedims(data["phi_gen"], (2, 3, 1))
    indices = Vector{Int64}(data["cond_indices"])
    @assert size(cfgs) == (L, L, Nensemble)
    M0 = magnetization(cfgs)
    u = CuArray(cfgs)
    buffers = modelb_fixed_buffers(u; seed)
    dir = joinpath(dirname(source), "physical_reference_t200")
    mkpath(dir)
    for t in Δt:Δt:200
        seconds = @elapsed begin
            modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, Δt))
            CUDA.synchronize()
        end
        @assert all(isfinite, u)
        cfgs = Array(u)
        drift = magnetization(cfgs) - M0
        output = joinpath(
            dir,
            "modelB_L32_Ne1024_$(initial)_t$(Int(t))_chain_dt0.002_seed$(seed).jld2",
        )
        @assert !ispath(output)
        jldsave(
            output;
            cfgs, cond_indices_0based=indices, L, Nensemble, initial,
            time=Float32(t), kappa=κ, lambda=λ, dt, noise_seed=seed,
            source_path=source, max_abs_conservation_drift=maximum(abs, drift),
        )
        println(
            "$initial t=$(Int(t)): $(round(seconds; digits=3)) s, " *
            "max|ΔM|=$(maximum(abs, drift))",
        )
        flush(stdout)
    end
end
