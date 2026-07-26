using CUDA, JLD2

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L = isempty(ARGS) ? 128 : parse(Int, only(ARGS))
Nensemble = 10240
κs, times = (0.26f0, 0.2705f0, 0.28f0), (1.0f0, 10.0f0, 100.0f0)
λ, γ, T, dt = 0.022f0, 1.0f0, 1.0f0, 0.002f0
noise_seeds = UInt64.((20260722, 20260723, 20260724))
hmc_dir = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata"))
dir = normpath(joinpath(@__DIR__, "..", "runs", "L$(L)_Ne10240_dt0p002_hmc_t1_t10_t100"))
mkpath(dir)

u = CUDA.zeros(Float32, L, L, Nensemble)
buffers = modelb_fixed_buffers(u; seed=noise_seeds[1])
for (κ, noise_seed) in zip(κs, noise_seeds)
    source = joinpath(hmc_dir, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^2.jld2")
    jldopen(source, "r") do f
        cfgs = JLD2.readmmap(JLD2.get_dataset(f, "cfgs"))
        @assert f["N"] == L && f["n_samples"] == Nensemble && isapprox(f["κ"], κ) &&
                isapprox(f["λ"], λ) && size(cfgs) == size(u)
        copyto!(u, cfgs)
    end
    seed_modelb_fixed!(buffers, noise_seed)
    t0 = 0.0f0
    for t in times
        seconds = @elapsed begin
            modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, t - t0))
            CUDA.synchronize()
        end
        @assert all(isfinite, u)
        jldsave(joinpath(dir, "modelB_L$(L)_Ne10240_hmc_kappa_$(κ)_t$(Int(t))_dt0.002.jld2");
                cfgs=Array(u), L, Nensemble, time=t, kappa=κ, lambda=λ, gamma=γ,
                temperature=T, dt, noise_seed, source_path=source)
        println("kappa=$κ t=$(Int(t)): $(round(seconds; digits=1)) s"); flush(stdout)
        t0 = t
    end
end
