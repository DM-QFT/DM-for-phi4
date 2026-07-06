#!/usr/bin/env julia
# Julia port of hmc_observables_table_2d.py — jackknife errors (no bootstrap).
#
# HMC-only physical-observable table for 2D φ⁴ at L=16,32,64 across
# κ=0.26 (symmetric), 0.2705 (near-critical), 0.28 (broken).  Reproduces the
# column set of draft/DM.tex Tab. tab:multiL_HMC_cumulants (HMC reference):
#
#   Global / action:  ⟨|φ̄|⟩, χ=V(⟨M²⟩−⟨|M|⟩²), ⟨S⟩/V, U₄
#   Zero-mode cumulants κ⁽⁰⟩₂,₄,₆,₈ of M=φ̄
#   Single-site cumulants ⟨|φ|⟩, κ₂…₈^site, U₄^site (per-site-then-average)
#
# All observables are even functions of the field, so the physical Z₂ symmetry
# (⟨φ⟩=0) is imposed via the even-moment cumulant formulas — no {φ,−φ}
# augmentation is needed.  Errors are binned jackknife over configurations
# (the binning absorbs HMC autocorrelation).  Writes CSV + txt to results/.
# The LaTeX-table emission of the Python version is intentionally omitted.
#
# Run from anywhere:  julia hmc_observables_table_2d.jl [--Ls 16 32 64] \
#                          [--kappas 0.26 0.2705 0.28] [--n_bins 50] [--stride 1]
#
# --stride s keeps every s-th configuration (1:s:end) before anything else —
# used to build the matched-N HMC rows (e.g. stride 5: 10240 → 2048) so the
# per-site cumulant finite-N bias matches the DM ensembles.

using Printf, Statistics, DelimitedFiles
const HERE = @__DIR__
include(joinpath(HERE, "PhysicalObservables.jl"))
using .PhysicalObservables

const PHI4_2D = dirname(dirname(dirname(HERE)))   # 2Dphi4/

# ----- tiny --flag value... parser -----
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
fmt(v, e, p = 5) = isfinite(v) ? "$(fmtg(v,p))($(fmtg(e,2)))" : "nan"

function main(args)
    d = parse_args(args)
    Ls      = haskey(d, "Ls")     ? parse.(Int, d["Ls"])        : [16, 32, 64]
    kappas  = haskey(d, "kappas") ? d["kappas"]                 : ["0.26", "0.2705", "0.28"]
    lam     = haskey(d, "lam")    ? parse(Float64, d["lam"][1]) : 0.022
    n_bins  = haskey(d, "n_bins") ? parse(Int, d["n_bins"][1])  : 50
    stride  = haskey(d, "stride") ? parse(Int, d["stride"][1])  : 1
    hmc_max = haskey(d, "hmc_max") ? parse(Int, d["hmc_max"][1]) : nothing
    out_dir = haskey(d, "out_dir") ? d["out_dir"][1] :
              joinpath(PHI4_2D, "results", "hmc_observables_L16_64")
    mkpath(out_dir)

    rows = Vector{Dict{String,Any}}()
    for L in Ls
        V = L * L
        for k_str in kappas
            k = parse(Float64, k_str)
            path = joinpath(PHI4_2D, "trainingdata",
                            "cfgs_wolff_fahmc_k=$(k_str)_l=$(@sprintf("%g", lam))_$(L)^2.jld2")
            if !isfile(path)
                println("[skip] missing $path"); continue
            end
            hmc = load_cfgs(path)
            stride > 1 && (hmc = hmc[:, :, 1:stride:end])
            hmc_max !== nothing && (hmc = hmc[:, :, 1:min(hmc_max, size(hmc, 3))])
            N_raw = size(hmc, 3)
            println("\n========== L=$L  κ=$k_str  (N=$N_raw) ==========")

            # global / action + zero-mode cumulants (binned jackknife)
            M = magnetisation(hmc)
            g_mean, g_err = zero_mode_observables_jk(M, V; n_bins = n_bins)
            sv_m, sv_e = binned_jackknife(action_per_cfg(hmc, k, lam) ./ V; n_bins = n_bins)
            g_mean["S_V"] = sv_m; g_err["S_V"] = sv_e

            # single-site cumulants (persite_avg = paper definition)
            s_mean, s_err = site_cumulants_avg_jk(hmc; n_bins = n_bins)

            row = Dict{String,Any}("L" => L, "kappa" => k_str, "N_raw" => N_raw, "V" => V)
            for kk in ["abs_M", "chi", "S_V", "U4", "k2", "k4", "k6", "k8"]
                row[kk] = g_mean[kk]; row["d_" * kk] = g_err[kk]
            end
            for kk in ["abs_phi", "k2", "k4", "k6", "k8", "U4_site"]
                row["site_" * kk] = s_mean[kk]; row["d_site_" * kk] = s_err[kk]
            end
            push!(rows, row)

            @printf("  <|phi_bar|> = %s   chi = %s   <S>/V = %s   U4 = %s\n",
                    fmt(g_mean["abs_M"], g_err["abs_M"]), fmt(g_mean["chi"], g_err["chi"], 4),
                    fmt(g_mean["S_V"], g_err["S_V"]), fmt(g_mean["U4"], g_err["U4"]))
            @printf("  k0_2=%s  k0_4=%s  k0_6=%s  k0_8=%s\n",
                    fmt(g_mean["k2"], g_err["k2"]), fmt(g_mean["k4"], g_err["k4"]),
                    fmt(g_mean["k6"], g_err["k6"]), fmt(g_mean["k8"], g_err["k8"]))
            @printf("  <|phi|>=%s  k2s=%s  k4s=%s  k6s=%s  k8s=%s  U4s=%s\n",
                    fmt(s_mean["abs_phi"], s_err["abs_phi"]), fmt(s_mean["k2"], s_err["k2"]),
                    fmt(s_mean["k4"], s_err["k4"]), fmt(s_mean["k6"], s_err["k6"]),
                    fmt(s_mean["k8"], s_err["k8"]), fmt(s_mean["U4_site"], s_err["U4_site"]))
        end
    end

    if isempty(rows)
        println("No data found — nothing written."); return
    end

    # -------- CSV --------
    cols = ["L", "kappa", "N_raw", "V",
            "abs_M", "d_abs_M", "chi", "d_chi", "S_V", "d_S_V", "U4", "d_U4",
            "k2", "d_k2", "k4", "d_k4", "k6", "d_k6", "k8", "d_k8",
            "site_abs_phi", "d_site_abs_phi", "site_k2", "d_site_k2",
            "site_k4", "d_site_k4", "site_k6", "d_site_k6", "site_k8", "d_site_k8",
            "site_U4_site", "d_site_U4_site"]
    csv_path = joinpath(out_dir, "hmc_observables_L16_64.csv")
    open(csv_path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join([get(r, c, "") for c in cols], ","))
        end
    end
    println("\nsaved $csv_path")

    # -------- txt table --------
    txt_path = joinpath(out_dir, "summary.txt")
    write_txt_table(rows, txt_path, lam, n_bins)
    println("saved $txt_path")
end

function write_txt_table(rows, out, lam, n_bins)
    c(v, e, p = 4) = fmt(v, e, p)
    hdr = @sprintf("%4s %7s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s",
                   "L", "k", "<|phi|>", "chi", "<S>/V", "U4",
                   "k0_2", "k0_4", "k0_6", "k0_8",
                   "<|phi_x|>", "k2_s", "k4_s", "k6_s", "k8_s", "U4_s")
    lines = String[
        "HMC (Wolff+FAHMC) physical observables — 2D phi^4, lambda=$(@sprintf("%g",lam))",
        "columns follow draft/DM.tex Tab. tab:multiL_HMC_cumulants (HMC only)",
        "Z_2-symmetric even-moment cumulants; binned jackknife over cfgs (n_bins=$n_bins).",
        "chi = V(<M^2>-<|M|>^2),  U4 = 1 - <M^4>/(3<M^2>^2),",
        "k0_n = zero-mode cumulants of M=phi_bar,  kn_s = single-site (persite_avg).",
        "", hdr, "-"^length(hdr)]
    for r in rows
        push!(lines, @sprintf("%4d %7s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s %13s",
            r["L"], r["kappa"],
            c(r["abs_M"], r["d_abs_M"]), c(r["chi"], r["d_chi"]),
            c(r["S_V"], r["d_S_V"]), c(r["U4"], r["d_U4"]),
            c(r["k2"], r["d_k2"]), c(r["k4"], r["d_k4"]),
            c(r["k6"], r["d_k6"]), c(r["k8"], r["d_k8"]),
            c(r["site_abs_phi"], r["d_site_abs_phi"]), c(r["site_k2"], r["d_site_k2"]),
            c(r["site_k4"], r["d_site_k4"]), c(r["site_k6"], r["d_site_k6"]),
            c(r["site_k8"], r["d_site_k8"]), c(r["site_U4_site"], r["d_site_U4_site"])))
    end
    write(out, join(lines, "\n") * "\n")
end

main(ARGS)
