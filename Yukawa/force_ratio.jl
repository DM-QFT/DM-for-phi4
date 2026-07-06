# force_ratio.jl
# Measure the force ratio <||F_ferm||> / <||F_B||> on a saved JLD2 ensemble,
# using the paper's Table II convention = even-odd preconditioned C2 trace
# estimator (eo_c2_force in YukawaFermionHMC2D.jl).
#
# Usage:
#   julia> include("YukawaFermionHMC2D.jl"); include("force_ratio.jl")
#   julia> force_ratio_from_file("samples/yukawa_g0.1_L16.jld2"; m2=-4.0, lambda=6.0, g=0.1)
#   julia> force_ratio_from_file("samples/yukawa_g0.3_L16.jld2"; m2=-1.55, lambda=2.4, g=0.3)

using JLD2, Statistics, Printf, Random

function force_ratio_from_file(file::AbstractString; m2::Float64, lambda::Float64,
                               g::Float64, mf::Float64=0.0, n_cfg::Int=5000,
                               seed::Int=777, nb::Int=25)
    d = load(file)
    cfg = d["configs"]
    N = d["N"]
    eo = eo_blocks(N)                       # constant even-odd hopping blocks
    rng = MersenneTwister(seed)
    nuse = min(n_cfg, size(cfg, 3))
    fb = Vector{Float64}(undef, nuse)
    ff = Vector{Float64}(undef, nuse)
    for k in 1:nuse
        fb[k], ff[k] = force_norms_eo_c2(cfg[:, :, k], eo, rng, m2, lambda, g, mf)
    end
    bs = div(nuse, nb)
    rs = [mean(@view ff[(b-1)*bs+1:b*bs]) / mean(@view fb[(b-1)*bs+1:b*bs]) for b in 1:nb]
    ratio = mean(ff) / mean(fb)
    err = std(rs) / sqrt(nb)
    @printf("%s : force ratio = %.3f%% +/- %.3f%%  (even-odd C2 trace, n=%d)\n",
            basename(file), 100ratio, 100err, nuse)
    return ratio, err
end
