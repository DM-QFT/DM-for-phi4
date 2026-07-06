#!/usr/bin/env julia
# Julia port of dm_vs_hmc_observables_2d.py (observables + text summary only;
# the matplotlib figures are not ported).
#
# DM vs HMC observables for 2D φ⁴ at L=128, across three κ (0.26 symmetric,
# 0.2705 critical, 0.28 broken).  Per source, with binned-jackknife errors:
#
#   Global:  ⟨|m|⟩, χ=V(⟨m²⟩−⟨|m|⟩²), U₄, c₄=κ₄, c₆=κ₆ (connected), ⟨S⟩
#   Single-site cumulants κ₂,₄,₆,₈ (spatial mean/std) and U₄^site, κ₈(m)
#   Geometric +sign clusters: ⟨s_max/V⟩
#   τ_int (Sokal) on the full HMC chain (DM is iid ⇒ τ≈0.5)
#
# DM samples are Z₂-augmented (φ,−φ interleaved); HMC is subsampled to n_match
# (last cfgs) then Z₂-interleaved to match.  If a DM .npy is absent the κ block
# is printed HMC-only.  Writes summary.txt to results/dm_vs_hmc_L128/.
#
# Run from anywhere:  julia dm_vs_hmc_observables_2d.jl [--epoch 10000]
#                          [--n_match 2048] [--n_bins 50] [--no_z2]

using Printf, Statistics
const HERE = @__DIR__
include(joinpath(HERE, "PhysicalObservables.jl"))
using .PhysicalObservables

const PHI4_2D = dirname(dirname(dirname(HERE)))

function parse_args(args)
    d = Dict{String,Vector{String}}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            key = args[i][3:end]; vals = String[]; j = i + 1
            while j <= length(args) && !startswith(args[j], "--")
                push!(vals, args[j]); j += 1
            end
            d[key] = vals; i = j
        else
            i += 1
        end
    end
    return d
end

fmtg(v, p) = isfinite(v) ? Printf.format(Printf.Format("%.$(p)g"), v) : "nan"
fmt(v, e) = "$(fmtg(v,4)) ± $(fmtg(e,2))"
reldiff(dv, hv) = (dv - hv) / (abs(hv) + 1e-30)

const RUNS = [
    ("0.26",   "phi4_L128_k0.26_l0.022_ncsnpp_sigma300",   "symmetric"),
    ("0.2705", "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450", "critical"),
    ("0.28",   "phi4_L128_k0.28_l0.022_ncsnpp_sigma640",   "broken"),
]

function main(args)
    d = parse_args(args)
    epoch    = haskey(d, "epoch")   ? parse(Int, d["epoch"][1])   : 10000
    n_match  = haskey(d, "n_match") ? parse(Int, d["n_match"][1]) : 2048
    n_bins   = haskey(d, "n_bins")  ? parse(Int, d["n_bins"][1])  : 50
    nbs      = haskey(d, "n_bins_site") ? parse(Int, d["n_bins_site"][1]) : 20
    lam      = haskey(d, "lam")     ? parse(Float64, d["lam"][1]) : 0.022
    z2       = !haskey(d, "no_z2")
    out_dir  = haskey(d, "out_dir") ? d["out_dir"][1] :
               joinpath(PHI4_2D, "results", "dm_vs_hmc_L128")
    mkpath(out_dir)

    summary = String[]
    taulines = String[]

    for (k_str, run_name, phase) in RUNS
        println("\n========== κ=$k_str  ($phase) ==========")
        k = parse(Float64, k_str)
        dm_path = joinpath(PHI4_2D, "runs", run_name, "data",
                           "samples_em_steps2000_epoch=$(@sprintf("%04d", epoch)).npy")
        hmc_path = joinpath(PHI4_2D, "trainingdata",
                            "cfgs_wolff_fahmc_k=$(k_str)_l=0.022_128^2.jld2")
        if !isfile(hmc_path)
            println("  [skip] no HMC at $hmc_path"); continue
        end

        # HMC: raw full chain (for τ_int), then last n_match, Z₂-interleave
        hmc_full = load_cfgs(hmc_path)
        n_hmc = min(n_match, size(hmc_full, 3))
        hmc_sub = hmc_full[:, :, (end - n_hmc + 1):end]
        hmc = z2 ? z2_interleave(hmc_sub) : hmc_sub

        have_dm = isfile(dm_path)
        if have_dm
            dm_full = load_cfgs(dm_path)
            dm_full = z2 ? z2_interleave(dm_full) : dm_full
            n_dm = min(n_match * (z2 ? 2 : 1), size(dm_full, 3))
            dm = dm_full[:, :, 1:n_dm]
            println("  HMC cfgs: $(size(hmc)), DM cfgs: $(size(dm)) (z2=$z2)")
        else
            println("  [no DM] $(dm_path) — HMC only")
            println("  HMC cfgs: $(size(hmc)) (z2=$z2)")
        end

        # ---- global (binned jackknife) ----
        gh, rawh = global_estimators(hmc, k, lam; n_bins = n_bins)
        gd = nothing; rawd = nothing
        have_dm && ((gd, rawd) = global_estimators(dm, k, lam; n_bins = n_bins))

        # ---- single-site cumulants (point spatial mean / std) ----
        kh = per_site_cumulants(hmc; order = 8)
        kd = have_dm ? per_site_cumulants(dm; order = 8) : nothing

        # ---- header ----
        push!(summary, "\n=== κ = $k_str ($phase) ===  HMC N=$(size(hmc,3))" *
                       (have_dm ? ", DM N=$(size(dm,3))" : " (no DM)"))
        push!(summary, @sprintf("  %14s    %18s    %18s    %22s",
                                "observable", "HMC", have_dm ? "DM" : "-", "rel diff (DM/HMC-1)"))
        for name in ["abs_m", "chi", "U4", "c4", "c6", "S_mean"]
            hv, he = gh[name]
            if have_dm
                dv, de = gd[name]
                push!(summary, @sprintf("  %14s    %18s    %18s    %+22.3f%%",
                                        name, fmt(hv, he), fmt(dv, de), 100 * reldiff(dv, hv)))
            else
                push!(summary, @sprintf("  %14s    %18s    %18s    %22s",
                                        name, fmt(hv, he), "-", "-"))
            end
        end

        # κ₂, κ₄ bar (Julia 1-indexed: kh[2]=κ₂, kh[4]=κ₄, kh[6]=κ₆, kh[8]=κ₈)
        for n in (2, 4, 6, 8)
            vh = mean(kh[n])
            if have_dm
                vd = mean(kd[n])
                push!(summary, @sprintf("  %14s    %18.4g    %18.4g    %+22.3f%%",
                                        "kappa_$(n)_bar", vh, vd, 100 * reldiff(vd, vh)))
            else
                push!(summary, @sprintf("  %14s    %18.4g    %18s    %22s",
                                        "kappa_$(n)_bar", vh, "-", "-"))
            end
        end
        push!(summary, @sprintf("  spatial std of κ₂(x):  HMC=%.3g%s",
                                std(kh[2]; corrected = false),
                                have_dm ? @sprintf("  DM=%.3g", std(kd[2]; corrected = false)) : ""))

        # ---- U4^site, κ₈(m) ----
        u4h, u4h_e = per_site_U4_jk(hmc; n_bins = nbs)
        k8h, k8h_e = kappa8_m(rawh["m"]; n_bins = n_bins)
        if have_dm
            u4d, u4d_e = per_site_U4_jk(dm; n_bins = nbs)
            k8d, k8d_e = kappa8_m(rawd["m"]; n_bins = n_bins)
            push!(summary, @sprintf("  %14s    %18s    %18s    %+22.3f%%",
                                    "U4_site", fmt(u4h, u4h_e), fmt(u4d, u4d_e), 100 * reldiff(u4d, u4h)))
            push!(summary, @sprintf("  %14s    %18s    %18s    %+22.3f%%",
                                    "kappa_8(m)", fmt(k8h, k8h_e), fmt(k8d, k8d_e), 100 * reldiff(k8d, k8h)))
        else
            push!(summary, @sprintf("  %14s    %18s    %18s    %22s", "U4_site", fmt(u4h, u4h_e), "-", "-"))
            push!(summary, @sprintf("  %14s    %18s    %18s    %22s", "kappa_8(m)", fmt(k8h, k8h_e), "-", "-"))
        end

        # ---- clusters ----
        _, mf_h = cluster_size_distribution(hmc; sign = +1)
        if have_dm
            _, mf_d = cluster_size_distribution(dm; sign = +1)
            push!(summary, @sprintf("  %14s    HMC=%.4f    DM=%.4f", "<s_max/V>", mean(mf_h), mean(mf_d)))
        else
            push!(summary, @sprintf("  %14s    HMC=%.4f", "<s_max/V>", mean(mf_h)))
        end

        # ---- τ_int on the full HMC chain ----
        m_full = magnetisation(hmc_full)
        S_full = action_per_cfg(hmc_full, k, lam)
        tau_M, _ = tau_int_series(abs.(m_full))
        tau_m2, _ = tau_int_series(m_full .^ 2)
        tau_S, _ = tau_int_series(S_full)
        tau_phi, _ = tau_int_series(hmc_full[1, 1, :])
        push!(taulines, @sprintf("  κ=%6s  HMC N=%d  τ_int(|M|)=%6.2f  τ_int(m²)=%6.2f  τ_int(S)=%6.2f  τ_int(φ_site)=%6.2f",
                                 k_str, length(m_full), tau_M, tau_m2, tau_S, tau_phi))
    end

    header = ["DM vs HMC observables — 2D phi^4, L=128, λ=0.022",
              "epoch=$epoch, n_match=$n_match, jk n_bins=$n_bins", ""]
    full = vcat(header, summary, ["",
        "tau_int (Sokal, c=5) on full HMC chain (DM samples are iid ⇒ τ≈0.5):"], taulines)
    text = join(full, "\n")
    write(joinpath(out_dir, "summary.txt"), text * "\n")
    println("\n" * text)
    println("\nsaved $(joinpath(out_dir, "summary.txt"))")
end

main(ARGS)
