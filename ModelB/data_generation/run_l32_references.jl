using CUDA, JLD2, NPZ, Statistics
include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))
L, Nensemble = 32, 5120
κ, λ, γ, T, dt, Δt = 0.28f0, 0.022f0, 1.0f0, 1.0f0, 0.002f0, 10.0f0
root = normpath(joinpath(@__DIR__, ".."))
samples(run) = joinpath(root, "runs", run, "samples")
noskip = samples("modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip")
all_data = samples("modelBcond_L32_k0.28_l0.022_t10_ncsnpp_all_data")
chains = (
    ("hmc", joinpath(noskip, "hmc_chain_ep10000_N5120_seed20260901", "hmc_ep10000_t0_N5120.npz"),
     UInt64(20261101), 200, "physical_reference_t200"),
    ("gaussian", joinpath(noskip, "gaussian_chain_ep10000_N5120_seed20261001",
                          "gaussian_ep10000_t0_N5120_seed20260721.npz"),
     UInt64(20261201), 200, "physical_reference_t200"),
    ("gaussian", joinpath(all_data, "gaussian_chain_ep2257_N5120_seed20260721",
                          "gaussian_ep2257_t0_N5120_seed20260721.npz"),
     UInt64(20260801), 100, "physical_reference"),
)
magnetization(cfgs) = vec(mean(Float64, cfgs; dims=(1, 2)))
for (initial, source, seed, last, subdir) in chains
    data = npzread(source)
    cfgs = permutedims(data["phi_gen"], (2, 3, 1))
    indices, M0 = Vector{Int64}(data["cond_indices"]), magnetization(cfgs); @assert size(cfgs) == (L, L, Nensemble)
    u = CuArray(cfgs)
    buffers = modelb_fixed_buffers(u; seed)
    dir = joinpath(dirname(source), subdir); mkpath(dir)
    for t in Δt:Δt:last
        seconds = @elapsed begin
            modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, Δt))
            CUDA.synchronize()
        end
        @assert all(isfinite, u)
        cfgs = Array(u)
        drift = magnetization(cfgs) - M0
        jldsave(joinpath(dir, "modelB_L32_Ne5120_$(initial)_t$(Int(t))_chain_dt0.002_seed$(seed).jld2");
                cfgs, cond_indices_0based=indices, L, Nensemble, initial, time=Float32(t),
                kappa=κ, lambda=λ, dt, noise_seed=seed, source_path=source)
        println("$initial t=$(Int(t)): $(round(seconds; digits=3)) s, max|ΔM|=$(maximum(abs, drift))"); flush(stdout)
    end
end
