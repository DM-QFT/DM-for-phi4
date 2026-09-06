using JLD2, Statistics, Printf

include(joinpath(@__DIR__, "..", "..", "tools", "julia", "WolffFAHMC_ND.jl"))

@assert length(ARGS) == 1
output = abspath(ARGS[1])
@assert !ispath(output)

L = 32
lambda = 1.0
kappas = collect(0.3325:0.0005:0.3370)
n_samples = 65536
n_warmup = 2000
seed = 202607390

tasks = map(eachindex(kappas)) do i
    Threads.@spawn begin
        _, acceptance, _, _, cluster_fraction, abs_m = wolff_fahmc(;
            D=2, N_lat=L, κ=kappas[i], λ=lambda,
            n_samples, n_warmup, seed=seed + i,
        )
        (; acceptance, cluster_fraction, abs_m)
    end
end
runs = fetch.(tasks)
abs_m = hcat(getproperty.(runs, :abs_m)...)
acceptance = getproperty.(runs, :acceptance)
cluster_fraction = getproperty.(runs, :cluster_fraction)

mkpath(dirname(output))
jldsave(
    output;
    abs_m, kappas, L, lambda, n_samples, n_warmup, seed,
    acceptance, cluster_fraction,
)
foreach(eachindex(kappas)) do i
    @printf(
        "kappa=%.5f  acceptance=%.3f  cluster=%.3f\n",
        kappas[i], acceptance[i], cluster_fraction[i],
    )
end
println("saved $output")
