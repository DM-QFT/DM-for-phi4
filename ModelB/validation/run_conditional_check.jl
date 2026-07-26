# Ground-truth conditional distribution: rerun Model B from a few fixed HMC
# configs with many independent noise realizations. Reference for validating
# the conditional diffusion model.
#
# Usage:
#   julia validation/run_conditional_check.jl                # 8 evenly spaced indices
#   julia validation/run_conditional_check.jl 63 121 136     # explicit 0-based indices

using CUDA, JLD2

modelB_dir = normpath(joinpath(@__DIR__, ".."))
include(joinpath(modelB_dir, "solver", "ModelB_fixed.jl"))
include(joinpath(modelB_dir, "data_generation", "io_utils.jl"))

L = 128
Nrep = 128
κ = 0.28f0
λ = 0.022f0
γ = 1.0f0
T = 1.0f0
dt = 0.002f0
horizon = 10.0f0
noise_seed = UInt64(20260901)  # fresh, differs from training-data seeds 2026072x

hmc_path = joinpath(
    modelB_dir, "..", "2Dphi4", "trainingdata",
    "cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(L)^2.jld2",
)

function default_indices(n)
    0 < n <= 10240 || error("default condition count must lie in 1:10240")
    return round.(Int, range(0, 10239; length=n))
end

indices_0based = isempty(ARGS) ? default_indices(8) : parse.(Int, ARGS)
Ncond = length(indices_0based)
Ncond > 0 || error("at least one condition index is required")
all(idx -> 0 <= idx < 10240, indices_0based) ||
    error("condition index out of range")
allunique(indices_0based) || error("condition indices must be unique")
println("conditions (0-based): $indices_0based, Nrep=$Nrep")

selected_cfgs = jldopen(hmc_path, "r") do file
    required = ("cfgs", "n_samples", "N", "κ", "λ")
    all(haskey(file, key) for key in required) ||
        error("HMC source is missing required metadata: $hmc_path")
    file["N"] == L || error("HMC source L mismatch")
    file["n_samples"] > maximum(indices_0based) ||
        error("condition index out of range")
    isapprox(file["κ"], κ; rtol=1.0f-6, atol=0.0f0) ||
        error("HMC source kappa mismatch")
    isapprox(file["λ"], λ; rtol=1.0f-6, atol=0.0f0) ||
        error("HMC source lambda mismatch")
    data = JLD2.get_dataset(file, "cfgs")
    selected = Array{Float32}(undef, L, L, Ncond)
    for (ci, idx0) in enumerate(indices_0based)
        copyto!(@view(selected[:, :, ci]), data[:, :, idx0 + 1])
    end
    selected
end

u = CUDA.zeros(Float32, L, L, Ncond * Nrep)
for ci in 1:Ncond
    slice = CuArray(@view(selected_cfgs[:, :, ci]))
    u[:, :, (ci-1)*Nrep+1 : ci*Nrep] .= slice
end
selected_cfgs = nothing

buffers = modelb_fixed_buffers(u; seed=noise_seed)
elapsed = @elapsed begin
    modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt, tspan=(0.0f0, horizon))
    CUDA.synchronize()
end
@assert all(isfinite, u)
println("evolved $(Ncond * Nrep) members to t=$horizon in $(elapsed)s")

out_dir = joinpath(modelB_dir, "runs", "conditional_check")
mkpath(out_dir)
out_path = joinpath(
    out_dir,
    "modelB_condcheck_L$(L)_kappa_$(κ)_t$(Int(horizon))_c$(Ncond)x$(Nrep)_seed$(noise_seed).jld2",
)
(ispath(out_path) || islink(out_path)) && error("output already exists: $out_path")
jldsave_exclusive(
    out_path;
    cfgs=Array(u),
    cfgs_layout="x,y,ensemble",
    cond_indices_0based=collect(indices_0based),
    n_per_cond=Nrep,
    time=horizon,
    L,
    kappa=κ,
    lambda=λ,
    gamma=γ,
    temperature=T,
    dt,
    noise_seed,
    source_path=hmc_path,
    algorithm="fixed SOSRA fused kernels",
)
println("saved: $out_path")
