# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║   WolffFAHMC_4D.jl — 4D φ⁴ Wolff + Fourier-Accelerated HMC (thin wrapper) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝#
using Random, Statistics, LinearAlgebra
using FFTW, Printf, ProgressMeter
using JLD2, Plots, DelimitedFiles
FFTW.set_num_threads(1)
include(joinpath(@__DIR__, "WolffFAHMC_ND.jl"))
BLAS.set_num_threads(8) # 1, 2, 4, 8, 16, 32, 36, 72
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║            4D thin wrappers (delegate to WolffFAHMC_ND)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    scan_kappa_wolff_fahmc_4d(κ_list; N_lat, blas_threads=1, ...)

4D κ 扫描: 转发到 scan_kappa(κ_list; D=4, ...). κ 并行已在 scan_kappa 里 (每 κ 一个 task,
用 `julia -t <n_κ>` 启动). blas_threads 设置 BLAS 线程数 (默认 1).
"""
function scan_kappa_wolff_fahmc_4d(κ_list::AbstractVector{Float64};
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
    save_dir::String = "4Dphi4/scan_kappa_wolff_fahmc",
    save_cfgs::Bool = false,
    blas_threads::Int = 1,
)
    BLAS.set_num_threads(blas_threads)
    return scan_kappa(κ_list;
        D=4,
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

# 4D 专用 FA metric 波函数常数 Z (拟合 1/G = Z·p̂² + m², 对角模标定).
# Z 依赖 (D, λ) —— 4D 的值记在这里, 不污染共享的 ND 引擎.
#   λ=0.5 : Z ≈ 0.30  (N=16, κ≈0.145 标定, R²≈0.9998)
#   其他 λ: 用 scratchpad/calib_Z_4d.jl 重新标定
const Z_FIX_4D = Dict(0.5 => 0.30)

"""
    generate_cfgs_4d(; N_lat, κ, λ, metric=:fixed, Z_fix=..., ...) -> (cfgs_or_nothing, acc, ε, fm, clf, abs_mag)

单点 (固定 κ,λ) 4D 构型生成. 转发到 wolff_fahmc(D=4, ...).
- store_cfgs=true : 全部构型留在内存 (小格子用).
- save_path!=""   : 流式写盘到 JLD2 (大格子用, 配 store_cfgs=false).
- metric=:fit 时用 4D 标定的 Z_fix (默认查 Z_FIX_4D[λ], 缺省回退 1.0).
"""
function generate_cfgs_4d(;
    N_lat::Int,
    κ::Float64,
    λ::Float64 = 0.9,
    n_samples::Int = 1000,
    n_warmup::Int = 1000,
    n_windows::Int = 5,
    n_steps::Int = 100,
    n_wolff::Int = 5,
    ε0::Float64 = 0.05,
    m2_eff::Float64 = 0.01,
    δ_target::Float64 = 0.75,
    seed::Int = 42,
    store_cfgs::Bool = false,
    save_path::String = "",
    show_progress::Bool = true,
    metric::Symbol = :fixed,
    Z_fix::Float64 = get(Z_FIX_4D, λ, 1.0),
)
    return wolff_fahmc(;
        D=4, N_lat=N_lat, κ=κ, λ=λ,
        n_samples=n_samples, n_warmup=n_warmup,
        n_windows=n_windows, n_steps=n_steps, n_wolff=n_wolff,
        ε0=ε0, m2_eff=m2_eff, δ_target=δ_target,
        seed=seed, store_cfgs=store_cfgs, save_path=save_path,
        show_progress=show_progress, metric=metric, Z_fix=Z_fix,
    )
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║            4D 作用量 (per-config) 与自相关分析                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function get_action_4d(phi::AbstractArray{<:Real,4}, κ::Float64, λ::Float64,
    lat::Lattice{4})
    S = 0.0
    @inbounds for idx in CartesianIndices(phi)
        ϕ = phi[idx]
        nn = nn_sum_fwd(phi, lat, idx)          # 只数前向邻居, 避免重复计 bond
        S += -2κ * ϕ * nn + (1 - 2λ) * ϕ^2 + λ * ϕ^4
    end
    return S
end

"""
    analyze_autocorrelation_4d(cfgs, κ, λ; n_skip, max_lag, save_prefix)

对内存中的 5 维构型数组 cfgs[:,:,:,:,i] 做 |φ̄|, φ̄, S/V, φ̄² 的积分自相关时间分析.
(大格子若用流式写盘, 请改用引擎返回的 abs_mag_series 单独算磁化自相关.)
"""
function analyze_autocorrelation_4d(cfgs::AbstractArray{<:Real,5},
    κ::Float64, λ::Float64;
    n_skip::Int=0,
    max_lag::Int=500,
    save_prefix::String="")
    cfgs_use = @view cfgs[:, :, :, :, (n_skip+1):end]
    N_lat = size(cfgs_use, 1)
    n_cfg = size(cfgs_use, 5)
    V = N_lat^4

    lat = Lattice{4}(N_lat)

    phi_bar = [mean(@view cfgs_use[:, :, :, :, i]) for i in 1:n_cfg]
    abs_mag = abs.(phi_bar)
    actions = [get_action_4d(cfgs_use[:, :, :, :, i], κ, λ, lat) / V for i in 1:n_cfg]
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
    println("4D 自相关分析 (Autocorrelation Analysis)")
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
        samples = collect(1:n_cfg)
        obs_csv = "$(save_prefix)_observables.csv"
        open(obs_csv, "w") do io
            println(io, "sample,abs_phi_bar,phi_bar,action_density,phi_bar2")
            for i in 1:n_cfg
                @printf(io, "%d,%.16e,%.16e,%.16e,%.16e\n",
                    i, abs_mag[i], phi_bar[i], actions[i], mag2[i])
            end
        end
        println("物理量演化数据已保存至: $obs_csv")

        plots_obs = [
            plot(samples, obs,
                xlabel="sample", ylabel=name, title="$name evolution",
                label="", lw=1.0, color=color)
            for (name, obs, color) in observables
        ]
        fig_obs = plot(plots_obs..., layout=(2, 2), size=(1100, 850), margin=5Plots.mm)
        obs_fig = "$(save_prefix)_observables_evolution.png"
        savefig(fig_obs, obs_fig)
        println("物理量演化图已保存至: $obs_fig")

        fig_ac = plot(plots_ac..., layout=(2, 2), size=(1100, 850), margin=5Plots.mm)
        fname = "$(save_prefix)_autocorrelation.png"
        savefig(fig_ac, fname)
        println("自相关分析图已保存至: $fname")
    end

    return results
end


"""
    run_production_4d(; N_lat, κ, λ, ...)

单点 4D 生产跑: 生成构型 → 存盘 → 自相关 + 物理观测量.
小格子 (默认 N=8) 用内存存全量; 大格子改用 save_path 流式写盘.
"""
function run_production_4d(;
    N_lat::Int = 8,
    κ::Float64 = 0.14,            # ≈κ_c(λ=0.9, N=8 实测); 换 λ/N 先 scan 重定位
    λ::Float64 = 0.9,
    n_samples::Int = 5000,
    n_warmup::Int = 2000,
    n_steps::Int = 100,
    n_windows::Int = 5,
    n_wolff::Int = 5,
    ε0::Float64 = 0.05,
    m2_eff::Float64 = 0.01,
    δ_target::Float64 = 0.75,
    seed::Int = 42,
    n_skip::Int = 200,
    thin::Int = 1,
    save_cfgs_jld2::Bool = true,
    show_progress::Bool = true,
    metric::Symbol = :fixed,
    Z_fix::Float64 = get(Z_FIX_4D, λ, 1.0),
)
    V = N_lat^4
    cfgs, acc, ε, fm, clf, abs_mag = generate_cfgs_4d(
        N_lat=N_lat, κ=κ, λ=λ,
        n_samples=n_samples, n_warmup=n_warmup,
        n_windows=n_windows, n_steps=n_steps, n_wolff=n_wolff,
        ε0=ε0, m2_eff=m2_eff, δ_target=δ_target, seed=seed,
        store_cfgs=true, show_progress=show_progress,
        metric=metric, Z_fix=Z_fix,
    )

    if save_cfgs_jld2
        idx = (n_skip+1):thin:n_samples
        fname = "4Dphi4/cfgs_wolff_fahmc_k=$(κ)_l=$(λ)_$(N_lat)^4.jld2"
        jldsave(fname; cfgs=cfgs[:, :, :, :, idx], κ=κ, λ=λ, N=N_lat,
            n_samples=length(idx), ε_final=ε, acc_rate=acc)
        println("构型已保存至: $fname")
    end

    analyze_autocorrelation_4d(cfgs[:, :, :, :, (n_skip+1):thin:end], κ, λ;
        n_skip=0, max_lag=500,
        save_prefix="4Dphi4/fahmc_N$(N_lat)^4_k$(κ)_l$(λ)")

    m, me, χ, χe, U4, U4e = observables_from_abs_mag(abs_mag, V; skip=n_skip)
    println("\n4D 物理观测量 (N=$N_lat, κ=$κ, λ=$λ):")
    println("  ⟨|φ̄|⟩ = $(round(m, digits=4)) ± $(round(me, digits=4))")
    println("  χ      = $(round(χ, digits=2)) ± $(round(χe, digits=2))")
    println("  U₄     = $(round(U4, digits=4)) ± $(round(U4e, digits=4))")

    return cfgs, acc, ε, fm, clf, abs_mag
end

@time cfgsa=run_production_4d(;
    N_lat=8,
    κ=0.145,
    λ=0.9,
    n_samples=5120*10,
    n_warmup=5000,
    n_steps=50,
    n_windows=3,
    n_wolff=5,
    ε0=0.5,
    m2_eff=1.0,
    δ_target=0.6,
    seed=42,
    n_skip=0,
    thin=10,
    save_cfgs_jld2=true,
    metric=:fit,
    Z_fix=0.30
)


using Plots
cfgsa[1]
plot(heatmap(cfgsa[1][:, :, 8, 8, 50]), aspect_ratio=1)


# @time scan_kappa_wolff_fahmc_4d([0.140, 0.142, 0.144, 0.145];
#     N_lat=16,
#     λ=0.5,
#     n_samples=2000,
#     n_warmup=100,
#     n_windows=2,
#     n_steps=30,
#     n_wolff=5,
#     ε0=1.0,
#     m2_eff=0.1,
#     δ_target=0.65,
#     n_skip=200,
#     save_dir="4Dphi4/scan_kappa_wolff_fahmc",
#     save_cfgs=false,
#     blas_threads=8,
# )
