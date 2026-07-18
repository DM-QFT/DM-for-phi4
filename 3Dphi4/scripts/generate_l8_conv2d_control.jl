include(joinpath(@__DIR__, "..", "..", "4Dphi4", "WolffFAHMC_ND.jl"))

# Reproducible small-volume reference data for the Conv2d-only diagnostic.
N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
kappa = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.1923
lambda = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.9
n_samples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5120
output = length(ARGS) >= 5 ? ARGS[5] : joinpath(
    @__DIR__, "..", "trainingdata",
    "cfgs_wolff_fahmc_k=$(kappa)_l=$(lambda)_$(N)^3.jld2",
)

mkpath(dirname(output))
isfile(output) && error("refusing to overwrite existing data: $output")

cfgs, accept_rate, epsilon_final, _, cluster_fraction, abs_mag = wolff_fahmc(;
    D=3,
    N_lat=N,
    κ=kappa,
    λ=lambda,
    n_samples=n_samples,
    n_warmup=2000,
    n_steps=40,
    n_wolff=5,
    ε0=0.4,
    m2_eff=1.0,
    δ_target=0.65,
    seed=20260714,
    store_cfgs=true,
    show_progress=true,
    metric=:fixed,
)

jldsave(
    output;
    cfgs=cfgs,
    N=N,
    kappa=kappa,
    lambda=lambda,
    n_samples=n_samples,
    acc_rate=accept_rate,
    epsilon_final=epsilon_final,
    avg_cluster_frac=cluster_fraction,
    abs_mag=abs_mag,
)

println("saved: $output")
println("shape=$(size(cfgs)), acceptance=$accept_rate, epsilon=$epsilon_final")
