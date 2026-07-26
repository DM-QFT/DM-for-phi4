using CUDA, JLD2
include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed_sciml.jl"))
L, Nensemble = 128, 512
κs, initials = (0.26f0, 0.28f0, 0.2705f0), (:gaussian, :hmc)
λ, γ, T, dt = 0.022f0, 1.0f0, 1.0f0, 0.005f0
tspan = (0.0f0, 10000.0f0)
# Project logarithmic times onto the fixed-step grid.
saveat = dt .* round.(Int, exp.(range(log(20.0), log(2.0e6); length=100)))
initial_seed, noise_seed = UInt64(20260719), UInt64(20260720)
hmc_dir = normpath(joinpath(@__DIR__, "..", "..", "2Dphi4", "trainingdata"))
dir = normpath(joinpath(@__DIR__, "..", "runs", "L128_Ne512_dt0p005_t10000_log0p1_N100_gaussian_hmc"))
mkpath(dir)
function load_hmc(κ)
    jldopen(joinpath(hmc_dir, "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^2.jld2"), "r") do f
        n = Int(f["n_samples"])
        indices = round.(Int, range(n / Nensemble, n; length=Nensemble))
        data = JLD2.readmmap(JLD2.get_dataset(f, "cfgs"))
        @assert f["N"] == L && n >= Nensemble && isapprox(f["κ"], κ) &&
                isapprox(f["λ"], λ) && size(data) == (L, L, n) && allunique(indices)
        u = Array{Float32}(undef, L, L, Nensemble)
        for (i, j) in enumerate(indices); copyto!(@view(u[:, :, i]), @view(data[:, :, j])); end
        CuArray(u)
    end
end
function cpu_snapshots(u)
    cfgs = Array{Float32}(undef, size(u[1])..., length(u))
    foreach(i -> copyto!(@view(cfgs[:, :, :, i]), Array(u[i])), eachindex(u))
    cfgs
end
CUDA.seed!(initial_seed); gaussian = CUDA.randn(Float32, L, L, Nensemble)
alg = ModelBFixedSOSRA(; buffers=modelb_fixed_buffers(gaussian; seed=noise_seed))
for initial in initials, κ in κs
    prob = modelB_2d_fixed_problem(initial === :gaussian ? gaussian : load_hmc(κ); γ, T, κ, λ, tspan)
    seconds = @elapsed sol = solve(prob, alg; dt, saveat, save_start=true, seed=noise_seed)
    @assert all(isfinite, sol.u[end])
    jldsave(joinpath(dir, "modelB_L128_Ne512_$(initial)_kappa_$(κ)_t0_10000_log0p1_N100_dt0.005.jld2");
            cfgs=cpu_snapshots(sol.u), times=Vector{Float32}(sol.t), L, Nensemble,
            initial=String(initial), kappa=κ, lambda=λ, gamma=γ, temperature=T, dt, initial_seed, noise_seed)
    println("initial=$initial kappa=$κ: $(round(seconds; digits=1)) s"); flush(stdout)
end
