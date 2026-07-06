#!/usr/bin/env julia
# Julia port of site_cumulants_kstat.py.
#
# Single-site cumulants κₙ^site of the RAW field (no Z₂ augmentation): per site
# x take the connected cumulant κₙ(x), then average over sites (DM.tex
# Eq. eq:site_cumulants_avg).
#
#   κ₂, κ₄ : unbiased k-statistics (scipy.stats.kstat equivalent, power-sum
#            formulas — no Julia k-statistic package exists).
#   κ₆, κ₈ : moment→cumulant recursion + delete-one-block jackknife bias
#            correction; point estimate cross-checked against the central-moment
#            plug-in (k6_scipy / k8_scipy).
#
# Run from anywhere:  julia site_cumulants_kstat.jl [k1 k2 ...]   (default κ set)

using Printf, Statistics
const HERE = @__DIR__
include(joinpath(HERE, "PhysicalObservables.jl"))
using .PhysicalObservables

const PHI4_2D = dirname(dirname(dirname(HERE)))
const DATA = joinpath(PHI4_2D, "trainingdata", "cfgs_wolff_fahmc_k={k}_l=0.022_128^2.jld2")

function main(args)
    kappas = isempty(args) ? ["0.26", "0.2705", "0.28"] : args
    for k in kappas
        path = replace(DATA, "{k}" => k)
        if !isfile(path)
            println("[skip] missing $path"); continue
        end
        c = load_cfgs(path)
        s = site_cumulants_kstat(c; n_blocks = 50)
        @printf("k=%6s | k2=%.4f k4=%.4f | k6 %.3f->%.3f(±%.3f) k8 %.2f->%.2f(±%.2f) | scipy chk: k6=%.3f k8=%.2f\n",
                k, s["k2"], s["k4"],
                s["k6_plug"], s["k6"], s["k6_err"],
                s["k8_plug"], s["k8"], s["k8_err"],
                s["k6_scipy"], s["k8_scipy"])
    end
end

main(ARGS)
