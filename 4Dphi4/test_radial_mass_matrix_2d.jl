using FFTW
using JLD2
using LinearAlgebra
using Printf
using Random
using Statistics

FFTW.set_num_threads(parse(Int, get(ENV, "FFTW_THREADS", "1")))

const N = parse(Int, get(ENV, "L", "128"))
const NH = N ÷ 2 + 1
const MAX_CFGS = parse(Int, get(ENV, "MAX_CFGS", "2048"))
const REPEATS = parse(Int, get(ENV, "REPEATS", "10000"))
const M2_EFF = parse(Float64, get(ENV, "M2_EFF", "0.1"))
const KAPPAS = parse.(Float64, split(get(ENV, "KAPPAS", "0.28,0.2705"), ","))

cfg_path(kappa) = @sprintf("2Dphi4/trainingdata/cfgs_wolff_fahmc_k=%s_l=0.022_%d^2.jld2",
                           string(kappa), N)

signed_freq(i, n) = begin
    k = i - 1
    k > n ÷ 2 ? k - n : k
end

function build_r2_map(n::Int)
    nh = n ÷ 2 + 1
    r2_map = zeros(Int, nh, n)
    max_r2 = 2 * (n ÷ 2)^2
    counts = zeros(Int, max_r2 + 1)
    for i2 in 1:n
        k2 = signed_freq(i2, n)
        for i1 in 1:nh
            k1 = signed_freq(i1, n)
            r2 = k1^2 + k2^2
            r2_map[i1, i2] = r2
            counts[r2 + 1] += 1
        end
    end
    return r2_map, counts
end

function full_invM(n::Int, m2_eff::Float64)
    nh = n ÷ 2 + 1
    two_pi_over_n = 2π / n
    invM = zeros(Float64, nh, n)
    for i2 in 1:n
        k2 = signed_freq(i2, n)
        for i1 in 1:nh
            k1 = signed_freq(i1, n)
            p2 = (two_pi_over_n^2) * (k1^2 + k2^2)
            invM[i1, i2] = 1.0 / (p2 + m2_eff)
        end
    end
    return invM
end

function radial_invM_by_r2(n::Int, m2_eff::Float64)
    max_r2 = 2 * (n ÷ 2)^2
    two_pi_over_n = 2π / n
    return [1.0 / ((two_pi_over_n^2) * r2 + m2_eff) for r2 in 0:max_r2]
end

function kx_only_M(n::Int, m2_eff::Float64)
    nh = n ÷ 2 + 1
    two_pi_over_n = 2π / n
    return [(two_pi_over_n * signed_freq(i1, n))^2 + m2_eff for i1 in 1:nh]
end

function quantile_or_nan(x, q)
    isempty(x) && return NaN
    return quantile(x, q)
end

function analyze_isotropy(kappa::Float64, r2_map::Matrix{Int}, counts::Vector{Int})
    path = cfg_path(kappa)
    G = zeros(Float64, size(r2_map))
    ns_used = 0
    jldopen(path, "r") do f
        cfgs = f["cfgs"]
        ncfg = size(cfgs, 3)
        stride = max(1, ncfg ÷ MAX_CFGS)
        ids = collect(1:stride:ncfg)
        if length(ids) > MAX_CFGS
            resize!(ids, MAX_CFGS)
        end
        real_buf = zeros(Float64, N, N)
        rplan = plan_rfft(real_buf; flags=FFTW.MEASURE)
        cbuf = zeros(ComplexF64, size(G))
        invV = 1.0 / (N * N)
        for t in ids
            phi = @view cfgs[:, :, t]
            mul!(cbuf, rplan, phi)
            @inbounds for i in eachindex(cbuf)
                z = cbuf[i]
                G[i] += (real(z)^2 + imag(z)^2) * invV
            end
        end
        ns_used = length(ids)
    end
    G ./= ns_used

    shell_sum = zeros(Float64, length(counts))
    @inbounds for i in eachindex(G)
        shell_sum[r2_map[i] + 1] += G[i]
    end
    shell_mean = zeros(Float64, length(counts))
    @inbounds for i in eachindex(shell_mean)
        counts[i] > 0 && (shell_mean[i] = shell_sum[i] / counts[i])
    end

    rel_all = Float64[]
    rel_low = Float64[]
    rel_mid = Float64[]
    @inbounds for i in eachindex(G)
        r2 = r2_map[i]
        c = counts[r2 + 1]
        if c >= 4 && r2 > 0
            mu = shell_mean[r2 + 1]
            rel = abs(G[i] - mu) / max(abs(mu), eps(Float64))
            push!(rel_all, rel)
            r2 <= 100 && push!(rel_low, rel)
            100 < r2 <= 1024 && push!(rel_mid, rel)
        end
    end

    return (
        ns_used=ns_used,
        rel_all=rel_all,
        rel_low=rel_low,
        rel_mid=rel_mid,
        G=G,
        shell_mean=shell_mean,
    )
end

function summarize_rel(label, rel)
    @printf("  %-15s count=%5d  median=%8.3g  p90=%8.3g  p99=%8.3g  rms=%8.3g\n",
            label, length(rel), quantile_or_nan(rel, 0.50),
            quantile_or_nan(rel, 0.90), quantile_or_nan(rel, 0.99),
            isempty(rel) ? NaN : sqrt(mean(abs2, rel)))
end

function benchmark_apply_invM(r2_map::Matrix{Int})
    rng = MersenneTwister(1234)
    field = randn(rng, N, N)
    out = similar(field)
    real_buf = zeros(Float64, N, N)
    rplan = plan_rfft(real_buf; flags=FFTW.MEASURE)
    cbuf = zeros(ComplexF64, size(r2_map))
    inv_full = full_invM(N, M2_EFF)
    inv_radial = radial_invM_by_r2(N, M2_EFF)

    function apply_full!()
        mul!(cbuf, rplan, field)
        @. cbuf *= inv_full
        mul!(out, plan_irfft(cbuf, N; flags=FFTW.MEASURE), cbuf)
        return out
    end

    iplan = plan_irfft(cbuf, N; flags=FFTW.MEASURE)
    function apply_full_cached!()
        mul!(cbuf, rplan, field)
        @. cbuf *= inv_full
        mul!(out, iplan, cbuf)
        return out
    end

    function apply_radial_lookup!()
        mul!(cbuf, rplan, field)
        @inbounds @simd for i in eachindex(cbuf)
            cbuf[i] *= inv_radial[r2_map[i] + 1]
        end
        mul!(out, iplan, cbuf)
        return out
    end

    apply_full_cached!()
    ref = copy(out)
    apply_radial_lookup!()
    max_err = maximum(abs.(out .- ref))

    GC.gc()
    t_full = @elapsed begin
        for _ in 1:REPEATS
            apply_full_cached!()
        end
    end
    GC.gc()
    t_radial = @elapsed begin
        for _ in 1:REPEATS
            apply_radial_lookup!()
        end
    end

    c0 = randn(rng, ComplexF64, size(r2_map))
    c = similar(c0)
    function mult_full!()
        copyto!(c, c0)
        @. c *= inv_full
        return c
    end
    function mult_radial!()
        copyto!(c, c0)
        @inbounds @simd for i in eachindex(c)
            c[i] *= inv_radial[r2_map[i] + 1]
        end
        return c
    end
    mult_full!()
    cref = copy(c)
    mult_radial!()
    max_mult_err = maximum(abs.(c .- cref))

    GC.gc()
    t_mult_full = @elapsed begin
        for _ in 1:REPEATS
            mult_full!()
        end
    end
    GC.gc()
    t_mult_radial = @elapsed begin
        for _ in 1:REPEATS
            mult_radial!()
        end
    end

    return (
        apply_full=t_full,
        apply_radial=t_radial,
        mult_full=t_mult_full,
        mult_radial=t_mult_radial,
        max_err=max_err,
        max_mult_err=max_mult_err,
    )
end

function main()
    r2_map, counts = build_r2_map(N)
    n_full = length(r2_map)
    n_shells = count(>(0), counts)
    max_r2 = length(counts) - 1

    @printf("2D radial mass-matrix test: L=%d, Nh=%d, MAX_CFGS=%d, REPEATS=%d\n",
            N, NH, MAX_CFGS, REPEATS)
    @printf("Stored rfft modes: %d = (%d, %d)\n", n_full, NH, N)
    @printf("Radial shells by r2=kx^2+ky^2: unique=%d, dense r2 length=%d\n",
            n_shells, max_r2 + 1)
    @printf("Float64 storage: full M_k=%.2f KiB, dense radial=%.2f KiB, kx-only=%.2f KiB\n",
            8n_full / 1024, 8(max_r2 + 1) / 1024, 8NH / 1024)

    inv_full = full_invM(N, M2_EFF)
    M_full = 1.0 ./ inv_full
    M_kx = kx_only_M(N, M2_EFF)
    rel_kx = Float64[]
    @inbounds for i2 in 1:N, i1 in 1:NH
        push!(rel_kx, abs(M_kx[i1] - M_full[i1, i2]) / M_full[i1, i2])
    end
    @printf("\nIf M is stored as kx-only length Nh, analytic M(kx,ky) relative error:\n")
    summarize_rel("all modes", rel_kx)

    @printf("\nBenchmark exact radial lookup vs full M_k multiply:\n")
    b = benchmark_apply_invM(r2_map)
    @printf("  multiply only: full=%8.4fs  radial_lookup=%8.4fs  ratio=%6.3f  maxerr=%.3e\n",
            b.mult_full, b.mult_radial, b.mult_radial / b.mult_full, b.max_mult_err)
    @printf("  rfft*mult*irfft: full=%8.4fs  radial_lookup=%8.4fs  ratio=%6.3f  maxerr=%.3e\n",
            b.apply_full, b.apply_radial, b.apply_radial / b.apply_full, b.max_err)

    @printf("\nMeasured G(k) isotropy from saved configurations:\n")
    for kappa in KAPPAS
        @printf("\nkappa=%.6g\n", kappa)
        a = analyze_isotropy(kappa, r2_map, counts)
        @printf("  configs used: %d\n", a.ns_used)
        summarize_rel("r2<=100", a.rel_low)
        summarize_rel("100<r2<=1024", a.rel_mid)
        summarize_rel("all r2", a.rel_all)
    end
end

main()
