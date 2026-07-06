using Random, Statistics, LinearAlgebra
using FFTW, Printf, ProgressMeter
using JLD2, Plots, DelimitedFiles

FFTW.set_num_threads(1)
include(joinpath(@__DIR__, "..", "..", "WolffFAHMC_ND.jl"))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║            3D thin wrappers (delegate to WolffFAHMC_ND)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    scan_kappa_wolff_fahmc_3d(κ_list; N_lat, ...)

3D κ scan: delegates to scan_kappa(κ_list; D=3, ..., save_dir=...).
Default save_dir = "3Dphi4/scan_kappa_wolff_fahmc".
"""
function scan_kappa_wolff_fahmc_3d(κ_list::AbstractVector{Float64};
    N_lat::Int,
    λ::Float64 = 0.9,
    n_samples::Int = 2000,
    n_warmup::Int = 1000,
    n_windows::Int = 5,
    n_steps::Int = 100,
    n_wolff::Int = 5,
    ε0::Float64 = 0.05,
    m2_eff::Float64 = 0.01,
    δ_target::Float64 = 0.75,
    n_skip::Int = 200,
    save_dir::String = "3Dphi4/scan_kappa_wolff_fahmc",
    save_cfgs::Bool = false,
)
    return scan_kappa(κ_list;
        D=3,
        N_lat=N_lat,
        λ=λ,
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_windows=n_windows,
        n_steps=n_steps,
        n_wolff=n_wolff,
        ε0=ε0,
        m2_eff=m2_eff,
        δ_target=δ_target,
        n_skip=n_skip,
        save_dir=save_dir,
        save_cfgs=save_cfgs,
    )
end

function get_action_3d(phi::AbstractArray{<:Real,3}, κ::Float64, λ::Float64,
    lat::Lattice{3})
    N_lat = size(phi, 1)
    S = 0.0
    @inbounds for idx in CartesianIndices(phi)
        ϕ = phi[idx]
        nn = nn_sum_all(phi, lat, idx)
        S += -2κ * ϕ * nn + (1 - 2λ) * ϕ^2 + λ * ϕ^4
    end
    return S
end

function analyze_autocorrelation_3d(cfgs::AbstractArray{<:Real,4},
    κ::Float64, λ::Float64;
    n_skip::Int=0,
    max_lag::Int=500,
    save_prefix::String="")
    cfgs_use = cfgs[:, :, :, (n_skip+1):end]
    N_lat = size(cfgs_use, 1)
    n_cfg = size(cfgs_use, 4)
    V = N_lat * N_lat * N_lat

    lat = Lattice{3}(N_lat)

    phi_bar = [mean(@view cfgs_use[:, :, :, i]) for i in 1:n_cfg]
    abs_mag = abs.(phi_bar)
    actions = [get_action_3d(cfgs_use[:, :, :, i], κ, λ, lat) / V for i in 1:n_cfg]
    mag2 = phi_bar .^ 2

    ml = min(max_lag, n_cfg ÷ 2)
    observables = [
        ("|φ̄|", abs_mag, :steelblue),
        ("φ̄", phi_bar, :darkorange),
        ("S/V", actions, :forestgreen),
        ("φ̄²", mag2, :purple),
    ]

    results = Dict{String,NamedTuple{(:τ_int, :W, :N_eff),Tuple{Float64,Int,Float64}}}()
    plots_ac = []

    println("\n" * "="^60)
    println("自相关分析 (Autocorrelation Analysis)")
    println("  N=$N_lat, n_cfgs=$n_cfg (skip=$n_skip)")
    println("="^60)

    for (name, obs, color) in observables
        C = autocorrelation(obs; max_lag=ml)
        τ_int, W = integrated_autocorrelation_time(C)
        n_eff = n_cfg / (2τ_int)
        results[name] = (τ_int=τ_int, W=W, N_eff=n_eff)

        @printf("  %-6s: τ_int = %6.2f,  W = %3d,  N_eff = %6.1f / %d  (%.1f%%)\n",
            name, τ_int, W, n_eff, n_cfg, 100n_eff / n_cfg)

        lags = 0:length(C)-1
        p = plot(lags, C,
            xlabel="lag τ", ylabel="C(τ)", title="$name autocorrelation",
            label="", lw=1.5, color=color)
        hline!(p, [0], ls=:dash, color=:gray, label="")
        vline!(p, [W], ls=:dot, color=:red,
            label="W=$W, τ_int=$(round(τ_int, digits=1))")
        push!(plots_ac, p)
    end
    println("="^60)

    if !isempty(save_prefix)
        fig_ac = plot(plots_ac..., layout=(2, 2), size=(1100, 850), margin=5Plots.mm)
        fname = "$(save_prefix)_autocorrelation.png"
        savefig(fig_ac, fname)
        println("自相关分析图已保存至: $fname")
    end

    return results
end




# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║            3D 构型生成 + κ 扫描示例                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ---- 单次 3D 构型生成 (small N for demo; 3D volume = N^3) ----
N_lat_3d = 128
κ_3d    = 0.1923
λ_3d    = 0.9


cfgs_3d, acc_rate_3d, ε_final_3d, fm_3d, avg_clf_3d, abs_mag_3d = wolff_fahmc(
    D         = 3,
    N_lat     = N_lat_3d,
    κ         = κ_3d,
    λ         = λ_3d,
    n_samples = 5,
    n_warmup  = 50,
    n_windows = 5,
    n_steps   = 50,
    n_wolff   = 5,
    ε0        = 0.05,
    m2_eff    = 0.01,
    δ_target  = 0.75,
    store_cfgs = true,
)



cfgs_3d, acc_rate_3d, ε_final_3d, fm_3d, avg_clf_3d, abs_mag_3d = wolff_fahmc(
    D         = 3,
    N_lat     = N_lat_3d,
    κ         = κ_3d,
    λ         = λ_3d,
    n_samples = 5120*2+5000,
    n_warmup  = 5000,
    n_windows = 5,
    n_steps   = 500,
    n_wolff   = 5,
    ε0        = 0.05,
    m2_eff    = 0.01,
    δ_target  = 0.75,
    store_cfgs = true,
)

fname = "cfgs_wolff_fahmc_k=$(κ_3d)_l=$(λ_3d)_$(N_lat_3d)^3.jld2"
jldsave(fname; cfgs=cfgs_3d[:, :, :, 5001:4:end], κ=κ_3d, λ=λ_3d, N=N_lat_3d,
    n_samples=5120, ε_final=ε_final_3d, acc_rate=acc_rate_3d)
println("构型已保存至: $fname")


ac_results = analyze_autocorrelation_3d(cfgs_3d[:, :, :, 5001:4:end], κ_3d, λ_3d;
    n_skip=200,
    max_lag=500,
    save_prefix="fahmc_N$(N_lat_3d)^3_k$(κ_3d)_l$(λ_3d)")


V_3d = N_lat_3d^3
mag_mean_3d, mag_err_3d, suscept_3d, suscept_err_3d, binder_3d, binder_err_3d =
    observables_from_abs_mag(abs_mag_3d, V_3d; skip=200)
println("\n3D 物理观测量 (N=$N_lat_3d, κ=$κ_3d):")
println("  ⟨|φ̄|⟩ = $(round(mag_mean_3d, digits=4)) ± $(round(mag_err_3d, digits=4))")
println("  χ      = $(round(suscept_3d, digits=2)) ± $(round(suscept_err_3d, digits=2))")
println("  U₄     = $(round(binder_3d, digits=4)) ± $(round(binder_err_3d, digits=4))")



# ---- 可选: 3D κ 扫描 (临界 κ 约 0.19–0.22 量级，用户可改范围) ----
result_3d = scan_kappa_wolff_fahmc_3d(
    collect(range(0.190, 0.195; length=81));
    N_lat = 64,
    n_samples = 20000,
    n_warmup = 2000,
    n_steps = 200,
    n_windows=10,
    n_wolff=10,
    ε0=0.05,
    m2_eff=0.01,
    δ_target=0.7,
    n_skip=200,
    save_dir = "3Dphi4/scan_kappa_wolff_fahmc",
    save_cfgs = false,
)


result_3d = scan_kappa_wolff_fahmc_3d(
    collect(range(0.190, 0.195; length=41));
    N_lat=64,
    n_samples=10000,
    n_warmup=2000,
    n_steps=100,
    n_windows=10,
    n_wolff=10,
    ε0=0.05,
    m2_eff=0.01,
    δ_target=0.7,
    n_skip=200,
    save_dir="3Dphi4/scan_kappa_wolff_fahmc",
    save_cfgs=false,
)

heatmap(cfgs_3d[:, :, 32, 4], aspect_ratio=1.0, color=:RdBu)
