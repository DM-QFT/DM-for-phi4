using FFTW
using JLD2
using LinearAlgebra
using Printf
using Statistics

FFTW.set_num_threads(parse(Int, get(ENV, "FFTW_THREADS", "1")))

const N = parse(Int, get(ENV, "L", "128"))
const NH = N ÷ 2 + 1
const MAX_CFGS = parse(Int, get(ENV, "MAX_CFGS", "2048"))
const KAPPAS = parse.(Float64, split(get(ENV, "KAPPAS", "0.28,0.2705"), ","))

cfg_path(kappa) = @sprintf("2Dphi4/trainingdata/cfgs_wolff_fahmc_k=%s_l=0.022_%d^2.jld2",
                           string(kappa), N)

signed_freq(i, n) = begin
    k = i - 1
    k > n ÷ 2 ? k - n : k
end

function build_maps(n::Int)
    nh = n ÷ 2 + 1
    max_r2 = 2 * (n ÷ 2)^2
    r2_map = zeros(Int, nh, n)
    counts = zeros(Int, max_r2 + 1)
    diag_i = fill(0, max_r2 + 1)
    diag_r2 = [2k^2 for k in 0:(n ÷ 2)]
    for i2 in 1:n
        k2 = signed_freq(i2, n)
        for i1 in 1:nh
            k1 = signed_freq(i1, n)
            r2 = k1^2 + k2^2
            r2_map[i1, i2] = r2
            counts[r2 + 1] += 1
        end
    end
    for i in 1:nh
        k = i - 1
        diag_i[2k^2 + 1] = i
    end
    return r2_map, counts, diag_r2
end

function measure_G(kappa::Float64)
    G = zeros(Float64, NH, N)
    ns_used = 0
    jldopen(cfg_path(kappa), "r") do f
        cfgs = f["cfgs"]
        ncfg = size(cfgs, 3)
        stride = max(1, ncfg ÷ MAX_CFGS)
        ids = collect(1:stride:ncfg)
        length(ids) > MAX_CFGS && resize!(ids, MAX_CFGS)
        real_buf = zeros(Float64, N, N)
        rplan = plan_rfft(real_buf; flags=FFTW.MEASURE)
        cbuf = zeros(ComplexF64, NH, N)
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
    return G ./ ns_used, ns_used
end

function shell_mean(G, r2_map, counts)
    sums = zeros(Float64, length(counts))
    @inbounds for i in eachindex(G)
        sums[r2_map[i] + 1] += G[i]
    end
    means = zeros(Float64, length(counts))
    @inbounds for i in eachindex(means)
        counts[i] > 0 && (means[i] = sums[i] / counts[i])
    end
    return means
end

function interp_diag_values(diag_values::Vector{Float64}, max_r2::Int)
    out = zeros(Float64, max_r2 + 1)
    # Diagonal radii are r2 = 2k^2. Interpolate linearly in r2.
    k_hi = 1
    @inbounds for r2 in 0:max_r2
        while k_hi < length(diag_values) && 2(k_hi - 1)^2 < r2
            k_hi += 1
        end
        if k_hi == 1
            out[r2 + 1] = diag_values[1]
        elseif 2(k_hi - 1)^2 == r2
            out[r2 + 1] = diag_values[k_hi]
        else
            k0 = k_hi - 2
            k1 = k_hi - 1
            x0 = 2k0^2
            x1 = 2k1^2
            y0 = diag_values[k0 + 1]
            y1 = diag_values[k1 + 1]
            w = (r2 - x0) / (x1 - x0)
            out[r2 + 1] = (1 - w) * y0 + w * y1
        end
    end
    return out
end

function nearest_diag_values(diag_values::Vector{Float64}, max_r2::Int)
    out = zeros(Float64, max_r2 + 1)
    @inbounds for r2 in 0:max_r2
        k = clamp(round(Int, sqrt(r2 / 2)), 0, length(diag_values) - 1)
        out[r2 + 1] = diag_values[k + 1]
    end
    return out
end

function rel_errors(reference, approx, counts; max_r2=typemax(Int))
    errs = Float64[]
    @inbounds for r2 in 1:min(length(reference) - 1, max_r2)
        counts[r2 + 1] == 0 && continue
        ref = reference[r2 + 1]
        ref <= 0 && continue
        push!(errs, abs(approx[r2 + 1] - ref) / ref)
    end
    return errs
end

function summarize(label, errs)
    @printf("  %-18s count=%5d  median=%8.3g  p90=%8.3g  p99=%8.3g  rms=%8.3g\n",
            label, length(errs), quantile(errs, 0.5), quantile(errs, 0.9),
            quantile(errs, 0.99), sqrt(mean(abs2, errs)))
end

function main()
    r2_map, counts, diag_r2 = build_maps(N)
    max_r2 = length(counts) - 1
    @printf("Diagonal-only radial test: L=%d, Nh=%d, MAX_CFGS=%d\n", N, NH, MAX_CFGS)
    @printf("Full half-spectrum modes=%d, populated radial shells=%d, diagonal samples=%d\n",
            length(r2_map), count(>(0), counts), NH)

    for kappa in KAPPAS
        @printf("\nkappa=%.6g\n", kappa)
        G, ns = measure_G(kappa)
        G_shell = shell_mean(G, r2_map, counts)
        G_diag = [G[k + 1, k + 1] for k in 0:(N ÷ 2)]
        G_near = nearest_diag_values(G_diag, max_r2)
        G_lin = interp_diag_values(G_diag, max_r2)

        @printf("  configs used: %d\n", ns)
        summarize("nearest all", rel_errors(G_shell, G_near, counts))
        summarize("linear all", rel_errors(G_shell, G_lin, counts))
        summarize("nearest r2<=100", rel_errors(G_shell, G_near, counts; max_r2=100))
        summarize("linear r2<=100", rel_errors(G_shell, G_lin, counts; max_r2=100))
        summarize("nearest <=1024", rel_errors(G_shell, G_near, counts; max_r2=1024))
        summarize("linear <=1024", rel_errors(G_shell, G_lin, counts; max_r2=1024))
    end
end

main()
