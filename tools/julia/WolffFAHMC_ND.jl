using Random, Statistics, LinearAlgebra
using FFTW, Printf, ProgressMeter
using JLD2, Plots, DelimitedFiles

FFTW.set_num_threads(1)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║            Lattice{D} — D-dimensional cubic lattice geometry             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

struct Lattice{D}
    N::Int
    V::Int
    fwd::NTuple{D, Vector{Int}}
    bwd::NTuple{D, Vector{Int}}
end

function Lattice{D}(N::Int) where D
    V = N^D
    fwd = ntuple(d -> [mod1(i + 1, N) for i in 1:N], D)
    bwd = ntuple(d -> [mod1(i - 1, N) for i in 1:N], D)
    Lattice{D}(N, V, fwd, bwd)
end

@inline function nn_sum_fwd(phi::AbstractArray{Float64,D}, lat::Lattice{D},
                            idx::CartesianIndex{D}) where D
    s = 0.0
    @inbounds for d in 1:D
        fwd_idx = CartesianIndex(ntuple(k -> k == d ? lat.fwd[d][idx[d]] : idx[k], D))
        s += phi[fwd_idx]
    end
    return s
end

@inline function nn_sum_all(phi::AbstractArray{Float64,D}, lat::Lattice{D},
                            idx::CartesianIndex{D}) where D
    s = 0.0
    @inbounds for d in 1:D
        fwd_idx = CartesianIndex(ntuple(k -> k == d ? lat.fwd[d][idx[d]] : idx[k], D))
        bwd_idx = CartesianIndex(ntuple(k -> k == d ? lat.bwd[d][idx[d]] : idx[k], D))
        s += phi[fwd_idx] + phi[bwd_idx]
    end
    return s
end

@inline function nn_neighbors(lat::Lattice{D}, idx::CartesianIndex{D}) where D
    ntuple(Val(2D)) do n
        d = (n - 1) ÷ 2 + 1
        if isodd(n)
            CartesianIndex(ntuple(k -> k == d ? lat.fwd[d][idx[d]] : idx[k], D))
        else
            CartesianIndex(ntuple(k -> k == d ? lat.bwd[d][idx[d]] : idx[k], D))
        end
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  D-dimensional φ⁴ action and force                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function get_action(phi::AbstractArray{Float64,D}, κ::Float64, λ::Float64,
                    lat::Lattice{D}) where D
    S = 0.0
    @inbounds for idx in CartesianIndices(phi)
        ϕ = phi[idx]
        nn_fwd = nn_sum_fwd(phi, lat, idx)
        S += -2κ * ϕ * nn_fwd + (1 - 2λ) * ϕ^2 + λ * ϕ^4
    end
    return S
end

@inline function force_at(phi::AbstractArray{Float64,D}, lat::Lattice{D},
                          idx::CartesianIndex{D}, two_κ::Float64, two_λ::Float64) where D
    nn = nn_sum_all(phi, lat, idx)
    ϕ = phi[idx]
    return two_κ * nn + 2ϕ * (two_λ * (1 - ϕ^2) - 1)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               Fourier Mass Matrix (D-dimensional)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

mutable struct FourierMassMatrix{D, P1, P2}
    N::Int
    Nh::Int
    M_k::Array{Float64, D}
    sqrtM_k::Array{Float64, D}
    invM_k::Array{Float64, D}
    rfft_plan::P1
    irfft_plan::P2
    cbuf::Array{ComplexF64, D}
    v_buf::Array{Float64, D}
end

function FourierMassMatrix{D}(N::Int, m2_eff::Float64) where D
    Nh = N ÷ 2 + 1
    k_shape = ntuple(d -> d == 1 ? Nh : N, D)
    M_k = zeros(Float64, k_shape)

    # Lattice dispersion p̂² = Σ_μ 4sin²(p_μ/2), the same one the calibrated
    # metric uses.
    for idx in CartesianIndices(k_shape)
        p2 = 0.0
        for d in 1:D
            p2 += 4 * sin(π * (idx[d] - 1) / N)^2
        end
        M_k[idx] = p2 + m2_eff
    end

    real_shape = ntuple(_ -> N, D)
    real_buf = zeros(Float64, real_shape)
    rp = plan_rfft(real_buf; flags=FFTW.MEASURE)
    cbuf = zeros(ComplexF64, k_shape)
    ip = plan_irfft(cbuf, N; flags=FFTW.MEASURE)

    return FourierMassMatrix{D, typeof(rp), typeof(ip)}(
        N, Nh, M_k, sqrt.(M_k), 1 ./ M_k, rp, ip, cbuf, zeros(Float64, real_shape)
    )
end

# FA metric from a fit  1/G(p) = Z·p̂² + m²  to the measured propagator, using
# only the diagonal modes (n,n,…,n) with the zero mode excluded.  Both Z and m²
# come from the straight-line fit.  p̂² = Σ_d 4sin²(p_d/2) is the lattice
# dispersion, which is what G⁻¹ is linear in.

# p̂² of every rfft mode.
function lattice_phat2(N::Int, ::Val{D}) where D
    k_shape = ntuple(d -> d == 1 ? N ÷ 2 + 1 : N, Val(D))
    P2 = zeros(Float64, k_shape)
    for idx in CartesianIndices(k_shape)
        s = 0.0
        for d in 1:D
            s += 4 * sin(π * (idx[d] - 1) / N)^2
        end
        P2[idx] = s
    end
    return P2
end

# Diagonal modes (n,n,…,n), n=1…N/2 (zero mode excluded): rfft indices + p̂².
function diagonal_modes(N::Int, ::Val{D}) where D
    ci = [CartesianIndex(ntuple(_ -> n + 1, Val(D))) for n in 1:N÷2]
    p2 = [D * 4 * sin(π * n / N)^2 for n in 1:N÷2]
    return ci, p2
end

# Refit Z, m² from the accumulated diagonal propagator and rebuild M(p).
# `Gd` holds Σ|φ_k|² over `n_cal` configurations. Returns (Z, m²).
function calibrate_metric_fit!(fm::FourierMassMatrix{D}, Gd::Vector{Float64},
                               n_cal::Int, P2::Array{Float64,D},
                               diag_p2::Vector{Float64}) where D
    y = (n_cal * fm.N^D) ./ Gd                      # 1/G on the diagonal modes
    x̄, ȳ = mean(diag_p2), mean(y)
    Z = sum((diag_p2 .- x̄) .* (y .- ȳ)) / sum(abs2, diag_p2 .- x̄)
    # M(k=0) = m² must stay positive; cap it at the softest non-zero mode so the
    # zero mode is never accelerated harder than the slowest physical one.
    m2 = max(ȳ - Z * x̄, Z * diag_p2[1] / D)
    @. fm.M_k     = Z * P2 + m2
    @. fm.sqrtM_k = sqrt(fm.M_k)
    @. fm.invM_k  = 1.0 / fm.M_k
    return Z, m2
end

function sample_momentum!(π_out::AbstractArray{Float64,D}, rng::AbstractRNG,
                          fm::FourierMassMatrix{D}) where D
    randn!(rng, π_out)
    mul!(fm.cbuf, fm.rfft_plan, π_out)
    @. fm.cbuf *= fm.sqrtM_k
    mul!(π_out, fm.irfft_plan, fm.cbuf)
    return nothing
end

function apply_invM!(v_out::AbstractArray{Float64,D}, fm::FourierMassMatrix{D},
                     π_field::AbstractArray{Float64,D}) where D
    mul!(fm.cbuf, fm.rfft_plan, π_field)
    @. fm.cbuf *= fm.invM_k
    mul!(v_out, fm.irfft_plan, fm.cbuf)
    return nothing
end

function kinetic_energy(fm::FourierMassMatrix{D}, π_field::AbstractArray{Float64,D}) where D
    apply_invM!(fm.v_buf, fm, π_field)
    K = 0.0
    @inbounds @simd for i in eachindex(π_field)
        K += π_field[i] * fm.v_buf[i]
    end
    return 0.5 * K
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  Leapfrog integrator (D-dimensional)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const LEAPFROG_DIVERGENCE_THRESHOLD = 1e10

function fourier_leapfrog!(φ::AbstractArray{Float64,D}, π_field::AbstractArray{Float64,D},
                           fm::FourierMassMatrix{D}, ε::Float64, n_steps::Int,
                           κ::Float64, λ::Float64, lat::Lattice{D}) where D
    v = fm.v_buf
    half_ε = 0.5 * ε
    two_κ = 2.0 * κ
    two_λ = 2.0 * λ

    @inbounds for idx in CartesianIndices(φ)
        π_field[idx] += half_ε * force_at(φ, lat, idx, two_κ, two_λ)
    end

    for _ in 1:n_steps-1
        apply_invM!(v, fm, π_field)
        max_abs = 0.0
        @inbounds @simd for k in eachindex(φ)
            φ[k] += ε * v[k]
            a = abs(φ[k])
            max_abs = ifelse(a > max_abs, a, max_abs)
        end
        max_abs > LEAPFROG_DIVERGENCE_THRESHOLD && return false

        @inbounds for idx in CartesianIndices(φ)
            π_field[idx] += ε * force_at(φ, lat, idx, two_κ, two_λ)
        end
    end

    apply_invM!(v, fm, π_field)
    @inbounds @simd for k in eachindex(φ)
        φ[k] += ε * v[k]
    end

    @inbounds for idx in CartesianIndices(φ)
        π_field[idx] += half_ε * force_at(φ, lat, idx, two_κ, two_λ)
    end

    return true
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  Single FA-HMC step                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function fahmc_step!(rng, φ::AbstractArray{Float64,D}, π_field, fm::FourierMassMatrix{D},
                     ε, n_steps, κ, λ, φ_old, lat::Lattice{D}) where D
    sample_momentum!(π_field, rng, fm)
    copyto!(φ_old, φ)

    S_old = get_action(φ, κ, λ, lat)
    K_old = kinetic_energy(fm, π_field)

    ok = fourier_leapfrog!(φ, π_field, fm, ε, n_steps, κ, λ, lat)

    if !ok
        copyto!(φ, φ_old)
        return false, Inf
    end

    S_new = get_action(φ, κ, λ, lat)
    K_new = kinetic_energy(fm, π_field)
    ΔH = (S_new + K_new) - (S_old + K_old)

    if isnan(ΔH) || isinf(ΔH)
        copyto!(φ, φ_old)
        return false, ΔH
    elseif log(rand(rng)) < -ΔH
        return true, ΔH
    else
        copyto!(φ, φ_old)
        return false, ΔH
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║             Dual Averaging step-size adaptation                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

mutable struct DualAveraging
    μ::Float64; log_εbar::Float64; Hbar::Float64
    γ::Float64; t0::Float64; κ_da::Float64
    δ_target::Float64; m::Int
end

function DualAveraging(ε0::Float64; δ_target=0.75, γ=0.05, t0=10.0, κ_da=0.75)
    DualAveraging(log(10ε0), 0.0, 0.0, γ, t0, κ_da, δ_target, 0)
end

function da_update!(da::DualAveraging, accept_prob::Float64)
    if isnan(accept_prob) || isinf(accept_prob)
        accept_prob = 0.0
    end
    da.m += 1
    m = da.m
    w = 1.0 / (m + da.t0)
    da.Hbar = (1 - w) * da.Hbar + w * (da.δ_target - accept_prob)
    log_ε = da.μ - sqrt(m) / da.γ * da.Hbar
    m_κ = m^(-da.κ_da)
    da.log_εbar = m_κ * log_ε + (1 - m_κ) * da.log_εbar
    return exp(log_ε)
end

da_final_ε(da::DualAveraging) = exp(da.log_εbar)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              Wolff cluster algorithm (D-dimensional)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

mutable struct WolffWorkspace{D}
    in_cluster::BitArray{D}
    stack::Vector{Int}
end

function WolffWorkspace{D}(N::Int) where D
    shape = ntuple(_ -> N, D)
    WolffWorkspace{D}(falses(shape), sizehint!(Int[], N^D))
end

function wolff_flip!(phi::AbstractArray{Float64,D}, kappa::Float64,
                     rng::AbstractRNG, ws::WolffWorkspace{D},
                     lat::Lattice{D}) where D
    N = lat.N
    fill!(ws.in_cluster, false)
    empty!(ws.stack)

    seed_idx = CartesianIndex(ntuple(_ -> rand(rng, 1:N), D))
    ws.in_cluster[seed_idx] = true
    push!(ws.stack, LinearIndices(phi)[seed_idx])
    cluster_size = 1

    four_kappa = 4.0 * kappa
    ci = CartesianIndices(phi)

    @inbounds while !isempty(ws.stack)
        lin = pop!(ws.stack)
        c_idx = ci[lin]
        phi_c = phi[lin]

        for nb_idx in nn_neighbors(lat, c_idx)
            ws.in_cluster[nb_idx] && continue
            phi_n = phi[nb_idx]
            bond = phi_c * phi_n
            bond <= 0.0 && continue
            if rand(rng) < -expm1(-four_kappa * bond)
                ws.in_cluster[nb_idx] = true
                push!(ws.stack, LinearIndices(phi)[nb_idx])
                cluster_size += 1
            end
        end
    end

    @inbounds for i in eachindex(phi)
        if ws.in_cluster[i]
            phi[i] = -phi[i]
        end
    end
    return cluster_size
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║           Autocorrelation analysis                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function autocorrelation(x::AbstractVector; max_lag::Union{Int,Nothing}=nothing)
    n = length(x)
    max_lag = isnothing(max_lag) ? n ÷ 2 : min(max_lag, n - 1)
    x_mean = mean(x)
    x_var = var(x; corrected=false)
    x_var < 1e-15 && return zeros(max_lag + 1)
    C = zeros(max_lag + 1)
    for τ in 0:max_lag
        s = 0.0
        for t in 1:(n - τ)
            s += (x[t] - x_mean) * (x[t + τ] - x_mean)
        end
        C[τ + 1] = s / ((n - τ) * x_var)
    end
    return C
end

function integrated_autocorrelation_time(C::AbstractVector; c::Float64=5.0)
    τ_int = 0.5
    for τ in 1:(length(C) - 1)
        τ_int += C[τ + 1]
        τ >= c * τ_int && return τ_int, τ
    end
    println("Warning: autocorrelation did not decay, τ_int may be underestimated")
    return τ_int, length(C) - 1
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║         Observables + Jackknife errors                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function observables_from_abs_mag(abs_mag_in::AbstractVector{Float64}, V::Int; skip::Int=0)
    abs_mag = @view abs_mag_in[(skip+1):end]
    N = length(abs_mag)
    N <= 1 && error("need N>1 samples after skip")

    mag_mean = mean(abs_mag)
    mag_err  = std(abs_mag) / sqrt(N - 1)

    am2 = mean(abs_mag .^ 2)
    suscept = V * (am2 - mag_mean^2)
    am4 = mean(abs_mag .^ 4)
    binder = 1 - am4 / (3 * am2^2)

    sum_am  = sum(abs_mag)
    sum_am2 = sum(abs_mag .^ 2)
    sum_am4 = sum(abs_mag .^ 4)
    jk_am   = [(sum_am  - abs_mag[i])     / (N - 1) for i in 1:N]
    jk_am2  = [(sum_am2 - abs_mag[i]^2)   / (N - 1) for i in 1:N]
    jk_am4  = [(sum_am4 - abs_mag[i]^4)   / (N - 1) for i in 1:N]
    jk_suscept = V .* (jk_am2 .- jk_am.^2)
    jk_binder  = 1 .- jk_am4 ./ (3 .* jk_am2.^2)
    var_suscept = (N - 1) * (mean(jk_suscept.^2) - mean(jk_suscept)^2)
    var_binder  = (N - 1) * (mean(jk_binder.^2) - mean(jk_binder)^2)
    suscept_err = sqrt(max(zero(var_suscept), var_suscept))
    binder_err  = sqrt(max(zero(var_binder), var_binder))

    return mag_mean, mag_err, suscept, suscept_err, binder, binder_err
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║       Wolff + FA-HMC main sampler (D-dimensional)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    wolff_fahmc(; D=2, N_lat, κ, λ=0.022, …, metric=:fit, …)

D-dimensional Wolff + Fourier-Accelerated HMC sampler for φ⁴ theory on
an N^D cubic lattice with periodic boundary conditions.

`metric` selects the FA mass matrix:
  * `:fit`   — refit M(p) = Z·p̂² + m² to the measured propagator on the diagonal
               modes at `n_windows` checkpoints during warmup, then freeze.
  * `:fixed` — M(p) = p̂² + m2_eff throughout (`n_windows` unused).

Returns `(cfgs_or_nothing, accept_rate, ε_final, fm, avg_cluster_frac, abs_mag_series)`.
"""
function wolff_fahmc(;
    D::Int = 2,
    N_lat::Int,
    κ::Float64,
    λ::Float64 = 0.022,
    n_samples::Int = 2000,
    n_warmup::Int = 1000,
    n_windows::Int = 3,
    n_steps::Int = 50,
    n_wolff::Int = 5,
    n_thin::Int = 1,
    ε0::Float64 = 0.1,
    m2_eff::Float64 = 0.1,
    δ_target::Float64 = 0.75,
    seed::Int = 42,
    store_cfgs::Bool = false,
    save_path::String = "",
    metric::Symbol = :fit,
    phi0 = nothing,
)
    rng = MersenneTwister(seed)
    N = N_lat

    lat = Lattice{D}(N)
    real_shape = ntuple(_ -> N, D)
    V = lat.V

    π_field = Array{Float64}(undef, real_shape)
    φ_old   = Array{Float64}(undef, real_shape)
    φ       = phi0 === nothing ? randn(rng, real_shape) : Float64.(copy(phi0))

    fm = FourierMassMatrix{D}(N, m2_eff)
    wolff_ws = WolffWorkspace{D}(N)

    da = DualAveraging(ε0; δ_target=δ_target)
    ε = ε0

    # Metric calibration (:fit): accumulate the diagonal propagator from the
    # middle third of warmup (after an initial equilibration third) and refit at
    # `n_windows` checkpoints, then freeze.  The accumulator is never reset, so
    # the frozen fit is the one with the most statistics.
    P2 = lattice_phat2(N, Val(D))
    diag_ci, diag_p2 = diagonal_modes(N, Val(D))
    cal_lo = n_warmup ÷ 3
    cal_steps = metric == :fit ?
        round.(Int, range(cal_lo, 2 * n_warmup ÷ 3, length = n_windows + 1))[2:end] : Int[]
    Gd = zeros(length(diag_ci))
    n_cal = 0

    n_prod = n_samples * n_thin
    n_total = n_warmup + n_prod
    cfgs_shape = (real_shape..., n_samples)
    cfgs = store_cfgs ? zeros(Float64, cfgs_shape) : nothing
    abs_mag_series = Vector{Float64}(undef, n_prod)

    save_to_disk = !isempty(save_path)
    jld_file = nothing
    if save_to_disk
        jld_file = jldopen(save_path, "w")
        jld_file["D"] = D
        jld_file["N"] = N
        jld_file["kappa"] = κ
        jld_file["lambda"] = λ
        jld_file["n_samples"] = n_samples
    end

    n_accept = 0
    total_cluster_size = 0
    total_wolff_flips = 0

    mem_mode = store_cfgs ? "full array" : (save_to_disk ? "stream to disk" : "observables only")
    mem_est = store_cfgs ? @sprintf("%.0f MB", V*n_samples*8/1e6) : "< 1 MB"

    metric_desc = metric == :fit ?
        "fit  (M=Z·p̂²+m², refit@$cal_steps)" : "fixed (M=p̂²+$m2_eff)"
    println("=" ^ 60)
    println("Wolff + Fourier-Accelerated HMC ($(D)D)")
    println("  N=$N ($(D)D, V=$V), κ=$κ, λ=$λ")
    println("  n_steps=$n_steps (HMC leapfrog), n_wolff=$n_wolff (cluster flips/iter)")
    println("  n_warmup=$n_warmup, metric: $metric_desc")
    println("  n_samples=$n_samples (thin=$n_thin -> $n_prod production iters), δ_target=$δ_target")
    println("  memory mode: $mem_mode ($mem_est)")
    println("=" ^ 60)

    prog = Progress(n_total; desc="Wolff+FA-HMC $(D)D ", showspeed=true)
    for step in 1:n_total

        n_steps_actual = rand(rng, round(Int, 0.8n_steps):round(Int, 1.2n_steps))
        accepted, ΔH = fahmc_step!(rng, φ, π_field, fm, ε, n_steps_actual, κ, λ, φ_old, lat)
        accept_prob = (isnan(ΔH) || isinf(ΔH)) ? 0.0 : min(1.0, exp(-ΔH))

        step_cluster_sum = 0
        for _ in 1:n_wolff
            cs = wolff_flip!(φ, κ, rng, wolff_ws, lat)
            step_cluster_sum += cs
        end
        if step > n_warmup
            total_cluster_size += step_cluster_sum
            total_wolff_flips += n_wolff
        end

        if step <= n_warmup
            ε = da_update!(da, accept_prob)

            if metric == :fit && !isempty(cal_steps) && step >= cal_lo && step <= cal_steps[end]
                mul!(fm.cbuf, fm.rfft_plan, φ)
                @inbounds for j in eachindex(diag_ci)
                    Gd[j] += abs2(fm.cbuf[diag_ci[j]])
                end
                n_cal += 1
                if step in cal_steps
                    Z, m2 = calibrate_metric_fit!(fm, Gd, n_cal, P2, diag_p2)
                    da = DualAveraging(ε; δ_target=δ_target)   # re-adapt ε to new metric (no reset-down)
                    @info @sprintf("calibrate @ step %d (%d cfgs): Z=%.4f, m²=%.4f, M cond=%.1f",
                                   step, n_cal, Z, m2, maximum(fm.M_k) / minimum(fm.M_k))
                end
            end

            if step == n_warmup
                ε = da_final_ε(da)
                @info @sprintf("Warmup done. Final ε = %.6f", ε)
            end
        end

        if step > n_warmup
            idx = step - n_warmup
            phi_bar = 0.0
            @inbounds @simd for k in eachindex(φ)
                phi_bar += φ[k]
            end
            abs_mag_series[idx] = abs(phi_bar / V)

            if idx % n_thin == 0
                j = idx ÷ n_thin
                if store_cfgs
                    selectdim(cfgs, D + 1, j) .= φ
                end
                if save_to_disk
                    jld_file[@sprintf("cfgs/%06d", j)] = copy(φ)
                end
            end
            if accepted
                n_accept += 1
            end
        end

        phase_str = step <= n_warmup ? "warmup" : "production"

        avg_cs = step_cluster_sum / max(n_wolff, 1)
        next!(prog; showvalues=[
            (:step, step),
            (:phase, phase_str),
            (:ε, round(ε, digits=5)),
            (:accept_prob, round(accept_prob, digits=3)),
            (:ΔH, isnan(ΔH) || isinf(ΔH) ? ΔH : round(ΔH, digits=3)),
            (:wolff_cluster, @sprintf("%.1f%% of V", 100avg_cs / V)),
        ])
    end
    finish!(prog)

    if save_to_disk
        jld_file["abs_mag"] = abs_mag_series
        close(jld_file)
        println("Configs streamed to: $save_path")
    end

    accept_rate = n_accept / n_prod
    avg_cluster_frac = total_wolff_flips > 0 ?
        total_cluster_size / (total_wolff_flips * V) : 0.0

    println("\n" * "=" ^ 60)
    println("  Production HMC accept rate: $(round(100accept_rate, digits=1))%")
    println("  Final step size ε = $(round(ε, digits=6))")
    println("  Avg Wolff cluster: $(round(100avg_cluster_frac, digits=1))% of volume")
    println("=" ^ 60)

    return cfgs, accept_rate, ε, fm, avg_cluster_frac, abs_mag_series
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║           Parallel κ scan                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function scan_kappa(κ_list::AbstractVector{Float64};
    D::Int = 2,
    N_lat::Int,
    λ::Float64 = 0.022,
    n_samples::Int = 2000,
    n_warmup::Int = 1000,
    n_windows::Int = 5,
    n_steps::Int = 500,
    n_wolff::Int = 5,
    ε0::Float64 = 0.05,
    m2_eff::Float64 = 0.01,
    δ_target::Float64 = 0.75,
    n_skip::Int = 200,
    save_dir::String = "scan_kappa",
    save_cfgs::Bool = false,
)
    mkpath(save_dir)
    n_κ = length(κ_list)
    V = N_lat^D

    mag_v = zeros(n_κ); mag_e = zeros(n_κ)
    sus_v = zeros(n_κ); sus_e = zeros(n_κ)
    bnd_v = zeros(n_κ); bnd_e = zeros(n_κ)
    acc_v = zeros(n_κ); clf_v = zeros(n_κ)

    println("=" ^ 60)
    println("Parallel κ scan — Wolff + FA-HMC ($(D)D)")
    println("  N=$N_lat, D=$D, λ=$λ, n_κ=$n_κ, n_wolff=$n_wolff")
    println("  κ range: $(minimum(κ_list)) → $(maximum(κ_list))")
    println("  n_samples=$n_samples, n_warmup=$n_warmup, n_steps=$n_steps")
    println("  threads: $(Threads.nthreads())")
    println("=" ^ 60)

    tasks = Vector{Task}(undef, n_κ)
    for (i, κ) in enumerate(κ_list)
        tasks[i] = Threads.@spawn begin
            seed = 1000 + i * 137
            disk_path = ""
            if save_cfgs
                κ_str = @sprintf("%.5f", κ)
                disk_path = joinpath(save_dir, "cfgs_$(D)D_L$(N_lat)_k$(κ_str).jld2")
            end
            _, acc_i, _, _, clf_i, abs_mag_i = wolff_fahmc(
                D=D, N_lat=N_lat, κ=κ, λ=λ,
                n_samples=n_samples, n_warmup=n_warmup,
                n_windows=n_windows, n_steps=n_steps, n_wolff=n_wolff,
                ε0=ε0, m2_eff=m2_eff, δ_target=δ_target,
                seed=seed, store_cfgs=false, save_path=disk_path,
            )
            obs = observables_from_abs_mag(abs_mag_i, V; skip=n_skip)
            (acc_i, clf_i, obs)
        end
    end

    for (i, κ) in enumerate(κ_list)
        acc_i, clf_i, obs = fetch(tasks[i])
        mag_v[i], mag_e[i] = obs[1], obs[2]
        sus_v[i], sus_e[i] = obs[3], obs[4]
        bnd_v[i], bnd_e[i] = obs[5], obs[6]
        acc_v[i] = acc_i
        clf_v[i] = clf_i
        @printf("  κ=%.5f: ⟨|φ̄|⟩=%.4f±%.4f  χ=%.2f±%.2f  U₄=%.4f±%.4f  acc=%.1f%%  cl=%.1f%%\n",
                κ, mag_v[i], mag_e[i], sus_v[i], sus_e[i],
                bnd_v[i], bnd_e[i], 100acc_v[i], 100clf_v[i])
    end

    csv_path = joinpath(save_dir, "scan_$(D)D_L$(N_lat)_n$(n_samples).csv")
    open(csv_path, "w") do io
        println(io, "kappa,mag,mag_err,suscept,suscept_err,binder,binder_err,acc_rate,cluster_frac")
        for i in 1:n_κ
            @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n",
                    κ_list[i], mag_v[i], mag_e[i], sus_v[i], sus_e[i],
                    bnd_v[i], bnd_e[i], acc_v[i], clf_v[i])
        end
    end
    println("CSV saved: $csv_path")

    p1 = plot(κ_list, mag_v, yerr=mag_e, seriestype=:scatter,
        xlabel="κ", ylabel="⟨|φ̄|⟩", title="Magnetization ($(D)D, L=$N_lat)",
        label="", ms=4, color=:steelblue)
    p2 = plot(κ_list, sus_v, yerr=sus_e, seriestype=:scatter,
        xlabel="κ", ylabel="χ", title="Susceptibility ($(D)D, L=$N_lat)",
        label="", ms=4, color=:crimson)
    p3 = plot(κ_list, bnd_v, yerr=bnd_e, seriestype=:scatter,
        xlabel="κ", ylabel="U₄", title="Binder Cumulant ($(D)D, L=$N_lat)",
        label="", ms=4, color=:forestgreen)
    hline!(p3, [2/3], ls=:dash, color=:gray, label="2/3")
    p4 = plot(κ_list, 100 .* clf_v, seriestype=:scatter,
        xlabel="κ", ylabel="Cluster %V", title="Avg Wolff Cluster ($(D)D)",
        label="", ms=4, color=:darkorange)

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 850), margin=5Plots.mm)
    fig_path = joinpath(save_dir, "scan_$(D)D_L$(N_lat)_n$(n_samples).png")
    savefig(fig, fig_path)
    println("Plot saved: $fig_path")

    i_peak = argmax(sus_v)
    κ_c = κ_list[i_peak]
    println("\n  χ peak at κ_c ≈ $κ_c ($(D)D, L=$N_lat)")

    return (κ=κ_list, mag=mag_v, mag_err=mag_e,
            χ=sus_v, χ_err=sus_e, U₄=bnd_v, U₄_err=bnd_e,
            acc_rate=acc_v, cluster_frac=clf_v, κ_c=κ_c)
end
