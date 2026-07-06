"""
PhysicalObservables — Julia port of the 2D φ⁴ physical-observable estimators in

    2Dphi4/analysis_clean/physical observable/{dm_vs_hmc_observables_2d.py,
                                              hmc_observables_table_2d.py,
                                              site_cumulants_kstat.py,
                                              acf_utils.py}

and the dimension-agnostic estimators they import from

    3Dphi4/analysis_clean/physical observable/compare_L64_multiL_HMC_cumulants.py

The neural-network pieces (`beta_conf_*`) and the matplotlib figures are NOT
ported — only the observables are computed here.

Data-layout convention
=======================
Configurations are stored sample-axis-LAST: a 2D ensemble is `(L, L, N)`.
This is the natural Julia (column-major) layout of both
  * the HMC `.jld2` training files (`jldopen(f)["cfgs"]` → `(L, L, N)`), and
  * the DM `.npy` samples (`npzread` → `(L, L, N)`),
so HMC and DM are handled by exactly the same code.  Note that the spatial axes
are *transposed* relative to the Python arrays (h5py reports `(N, L, L)`); every
quantity reported here is a spatial average / sum / connected cumulant, all of
which are invariant under that transpose, so the scalars match Python exactly.

Connected cumulants are computed with **`StatsBase.cumulant`** (the Smith-1995
recursion on plug-in central moments): directly for the 1-D zero-mode
magnetisation κ₁…κ₈, and per lattice site for the single-site κₙ(x).  The
jackknife error machinery instead uses a vectorised batch form of the same
recursion (`cumulants_from_raw_moments`), because calling `StatsBase.cumulant`
once per site × per jackknife sample would be 10³–10⁶× slower; the two give
bitwise-identical numbers (verified).

All errors are **binned jackknife** (no bootstrap): the binning absorbs HMC
autocorrelation, and the leave-one-bin-out resampling propagates errors through
the non-linear observables (χ, U₄, κₙ).
"""
module PhysicalObservables

using JLD2, NPZ, Statistics, FFTW
using StatsBase: cumulant

export load_cfgs, z2_concat, z2_interleave,
       magnetisation, action_per_cfg,
       binned_jackknife, global_estimators,
       cumulants_from_raw_moments, magnetisation_cumulants,
       per_site_cumulants, per_site_U4_jk, kappa8_m,
       cluster_size_distribution,
       acf_per_chain_fft, sokal_window, tau_int_series,
       zero_mode_observables_jk, site_cumulants_avg_jk,
       kstat_site, site_cumulants_kstat


# ===========================================================================
#  I/O
# ===========================================================================

"""
    load_cfgs(path) -> Array{Float64,3}

Load an ensemble as `(L, L, N)` Float64.  Accepts a `.jld2` HMC training file
(dataset `"cfgs"`) or a `.npy` DM-sample file.  Mirrors `load_hmc` / `load_dm`
in `dm_vs_hmc_observables_2d.py` and `load_configs` in `site_cumulants_kstat.py`.
"""
function load_cfgs(path::AbstractString)
    if endswith(path, ".npy")
        return Float64.(npzread(path))
    else
        return jldopen(path, "r") do f
            Array{Float64}(f["cfgs"])
        end
    end
end


# ===========================================================================
#  Z₂ symmetrisation (physical φ → −φ)
# ===========================================================================

"Block Z₂: `cat(cfgs, -cfgs)` along the sample axis — the `np.concatenate`
convention of `hmc_observables_table_2d.py` (paired bootstrap)."
z2_concat(cfgs::AbstractArray) = cat(cfgs, -cfgs; dims = ndims(cfgs))

"""
    z2_interleave(cfgs)

Interleave φ and −φ so each Z₂ pair is adjacent (`_z2_interleave` in
`dm_vs_hmc_observables_2d.py`): keeps the binned jackknife Z₂-balanced.
Result order along the sample axis is cfg₁,−cfg₁,cfg₂,−cfg₂,…
"""
function z2_interleave(cfgs::AbstractArray{T,3}) where {T}
    L1, L2, N = size(cfgs)
    out = Array{T}(undef, L1, L2, 2N)
    @views out[:, :, 1:2:end] .= cfgs
    @views out[:, :, 2:2:end] .= -cfgs
    return out
end


# ===========================================================================
#  Global observables on (L, L, N) cfgs
# ===========================================================================

"Per-cfg magnetisation `m = ⟨φ⟩_V` (signed), length-N vector."
magnetisation(cfgs::AbstractArray) =
    vec(mean(cfgs; dims = ntuple(i -> i, ndims(cfgs) - 1)))

"""
    action_per_cfg(cfgs, k, l) -> Vector

Per-cfg φ⁴ action S(φ), forward-only neighbour sum, matching `phi4_action`
(`phi_min=-1, phi_max=+1`, i.e. the identity denormalisation):

  S = Σ_x [ -2k φ_x Σ_d φ_{x+ê_d} + (1-2l) φ_x² + l φ_x⁴ ].
"""
function action_per_cfg(cfgs::AbstractArray, k::Real, l::Real)
    nd = ndims(cfgs)
    sdims = nd - 1
    nb = zeros(eltype(cfgs), size(cfgs))
    for d in 1:sdims
        nb = nb .+ circshift(cfgs, ntuple(i -> i == d ? 1 : 0, nd))
    end
    dens = @. -2 * k * cfgs * nb + (1 - 2 * l) * cfgs^2 + l * cfgs^4
    return vec(sum(dens; dims = ntuple(i -> i, sdims)))
end


# ===========================================================================
#  Binned jackknife (handles HMC autocorrelation)
# ===========================================================================

"Block means of each column of `cols` (N, ncols) → (n_bins, ncols).
Consecutive blocks of `bs = N ÷ n_bins` samples, exactly as
`x[:n_used].reshape(n_bins, bs, …).mean(axis=1)` in Python."
function _bin_means(cols::AbstractMatrix, n_bins::Int)
    N, ncols = size(cols)
    n_bins = min(n_bins, N)
    bs = div(N, n_bins)
    nused = bs * n_bins
    B = Array{Float64}(undef, n_bins, ncols)
    for j in 1:ncols
        # column-major reshape: first reshaped dim (bs) varies fastest ⇒
        # column b holds the consecutive block [(b-1)*bs+1 : b*bs].
        blk = reshape(view(cols, 1:nused, j), bs, n_bins)
        B[:, j] = vec(mean(blk; dims = 1))
    end
    return B
end

"""
    binned_jackknife(cols::AbstractMatrix, func; n_bins=50) -> (val, err)

`cols` is `(N, ncols)`, each column a per-cfg quantity.  `func` maps the length-
`ncols` vector of (jackknifed) column means to a scalar.  Mirrors
`binned_jackknife` in `dm_vs_hmc_observables_2d.py`.
"""
function binned_jackknife(cols::AbstractMatrix, func; n_bins::Int = 50)
    B = _bin_means(cols, n_bins)
    nb = size(B, 1)
    full = float(func(vec(mean(B; dims = 1))))
    jk = Vector{Float64}(undef, nb)
    keep = trues(nb)
    for i in 1:nb
        keep[i] = false
        jk[i] = func(vec(mean(view(B, keep, :); dims = 1)))
        keep[i] = true
    end
    err = sqrt((nb - 1) * mean((jk .- full) .^ 2))
    return full, err
end

"Scalar series → binned-jackknife (mean, err)."
binned_jackknife(x::AbstractVector; n_bins::Int = 50) =
    binned_jackknife(reshape(collect(float.(x)), :, 1), v -> v[1]; n_bins = n_bins)

"""
    global_estimators(cfgs, k, l; n_bins=50) -> (out, raw)

`out` maps "abs_m","m2","chi","U4","c4","c6","S_mean" → `(mean, err)` (binned
jackknife); `raw` holds the per-cfg `m, abs_m, m2, m4, S`.  Mirrors
`global_estimators` in `dm_vs_hmc_observables_2d.py`.
"""
function global_estimators(cfgs::AbstractArray, k::Real, l::Real; n_bins::Int = 50)
    L = size(cfgs, 1)
    V = L^2
    m = magnetisation(cfgs)
    am = abs.(m); m2 = m .^ 2; m4 = m .^ 4; m6 = m .^ 6
    S = action_per_cfg(cfgs, k, l)

    out = Dict{String,Tuple{Float64,Float64}}()
    out["abs_m"]  = binned_jackknife(am; n_bins = n_bins)
    out["m2"]     = binned_jackknife(m2; n_bins = n_bins)
    out["chi"]    = binned_jackknife(hcat(am, m2), v -> V * (v[2] - v[1]^2); n_bins = n_bins)
    out["U4"]     = binned_jackknife(hcat(m2, m4), v -> 1 - v[2] / (3 * v[1]^2); n_bins = n_bins)
    out["c4"]     = binned_jackknife(hcat(m2, m4), v -> v[2] - 3 * v[1]^2; n_bins = n_bins)
    out["c6"]     = binned_jackknife(hcat(m2, m4, m6),
                                     v -> v[3] - 15 * v[1] * v[2] + 30 * v[1]^3; n_bins = n_bins)
    out["S_mean"] = binned_jackknife(S; n_bins = n_bins)

    raw = Dict("m" => m, "abs_m" => am, "m2" => m2, "m4" => m4, "S" => S)
    return out, raw
end


# ===========================================================================
#  Cumulants
# ===========================================================================

"""
    cumulants_from_raw_moments(m) -> Vector

Vectorised batch form of `StatsBase.cumulant`: connected cumulants κ₁…κ_order
from RAW moments `m[r] = ⟨xʳ⟩` (`r = 1…order`), via the standard recursion
(i runs from 1, so a non-zero mean is handled):

  κ_n = m_n − Σ_{i=1}^{n-1} C(n-1, i-1) κ_i m_{n-i}.

Each `m[r]` may be a scalar or an array, so the recursion runs over all sites /
bootstrap resamples at once (where per-element `StatsBase.cumulant` calls would
be prohibitive).  For κ≥2 of a single series it equals `StatsBase.cumulant`.
Matches `cumulants_from_moments` / `_cumulants_from_raw` (2D) and
`_cumulants_from_raw_moments_vec` (3D).
"""
function cumulants_from_raw_moments(m::AbstractVector)
    order = length(m)
    k = Vector{Any}(undef, order)
    k[1] = m[1]
    for n in 2:order
        v = m[n] isa Number ? m[n] : copy(m[n])
        for i in 1:(n - 1)
            v = v .- binomial(n - 1, i - 1) .* (k[i] .* m[n - i])
        end
        k[n] = v
    end
    return k
end

"""
    magnetisation_cumulants(M; order=8) -> Vector{Float64}

Connected cumulants κ₁…κ_order of the scalar magnetisation series `M`, via
**`StatsBase.cumulant(M, 1:order)`**.  These are the plug-in cumulants,
identical to the central-moment recursion `nth_cumulants` used by the Python
reference; on a Z₂-symmetrised ensemble the odd cumulants vanish.
"""
magnetisation_cumulants(M::AbstractVector; order::Int = 8) =
    cumulant(collect(float.(M)), 1:order)


# ===========================================================================
#  Single-site cumulants κₙ(x)  (vectorised over sites)
# ===========================================================================

"""
    per_site_cumulants(cfgs; order=6) -> Vector of (L,L)

Per-site connected cumulants κₙ(x), treating each lattice site as a 1-D sample
over the ensemble and calling `StatsBase.cumulant(φ(x,·), 1:order)` per site.
`out[n]` is the `(L,L)` map of κₙ(x).  Matches `per_site_cumulants` in
`dm_vs_hmc_observables_2d.py`.
"""
function per_site_cumulants(cfgs::AbstractArray{<:Real,3}; order::Int = 6)
    L1, L2, N = size(cfgs)
    flat = reshape(Array{Float64}(cfgs), L1 * L2, N)
    K = Array{Float64}(undef, order, L1 * L2)
    @inbounds for x in 1:(L1 * L2)
        K[:, x] = cumulant(view(flat, x, :), 1:order)
    end
    return [reshape(K[n, :], L1, L2) for n in 1:order]
end

"""
    per_site_U4_jk(cfgs; n_bins=20) -> (mean, err)

Single-site Binder cumulant, per-site-then-spatial-average (binned jackknife):
`U₄^site = (1/V) Σ_x [1 − m₄(x)/(3 m₂(x)²)]`.  Matches `per_site_U4_jk`.
"""
function per_site_U4_jk(cfgs::AbstractArray{<:Real,3}; n_bins::Int = 20)
    L1, L2, N = size(cfgs)
    n_bins = min(n_bins, N)
    bs = div(N, n_bins)
    bm2 = Array{Float64}(undef, L1, L2, n_bins)
    bm4 = Array{Float64}(undef, L1, L2, n_bins)
    for b in 1:n_bins
        xb = @view cfgs[:, :, ((b - 1) * bs + 1):(b * bs)]
        x2 = xb .^ 2
        bm2[:, :, b] = dropdims(mean(x2; dims = 3); dims = 3)
        bm4[:, :, b] = dropdims(mean(x2 .^ 2; dims = 3); dims = 3)
    end
    u4avg(m2, m4) = mean(@. 1 - m4 / (3 * m2^2))
    full = u4avg(dropdims(mean(bm2; dims = 3); dims = 3),
                 dropdims(mean(bm4; dims = 3); dims = 3))
    t2 = dropdims(sum(bm2; dims = 3); dims = 3)
    t4 = dropdims(sum(bm4; dims = 3); dims = 3)
    jk = Vector{Float64}(undef, n_bins)
    for i in 1:n_bins
        jk[i] = u4avg((t2 .- bm2[:, :, i]) ./ (n_bins - 1),
                      (t4 .- bm4[:, :, i]) ./ (n_bins - 1))
    end
    err = sqrt((n_bins - 1) * mean((jk .- full) .^ 2))
    return full, err
end

"""
    kappa8_m(m; n_bins=50) -> (val, err)

Eighth connected magnetisation cumulant κ₈(m), Z₂-symmetric form (⟨m⟩=0),
binned jackknife.  Matches `kappa8_m` in `dm_vs_hmc_observables_2d.py`.
"""
function kappa8_m(m::AbstractVector; n_bins::Int = 50)
    cols = hcat(m .^ 2, m .^ 4, m .^ 6, m .^ 8)
    f(v) = v[4] - 28 * v[3] * v[1] - 35 * v[2]^2 + 420 * v[2] * v[1]^2 - 630 * v[1]^4
    return binned_jackknife(cols, f; n_bins = n_bins)
end


# ===========================================================================
#  Geometric +sign clusters (4-connectivity, open BC)
# ===========================================================================

"Connected-component sizes of a boolean mask (4-connectivity, open BC) by
iterative flood fill — the Julia stand-in for `scipy.ndimage.label`."
function _label_sizes(mask::AbstractMatrix{Bool})
    L1, L2 = size(mask)
    lab = falses(L1, L2)
    sizes = Int[]
    stack = Tuple{Int,Int}[]
    @inbounds for j in 1:L2, i in 1:L1
        (mask[i, j] && !lab[i, j]) || continue
        sz = 0
        empty!(stack); push!(stack, (i, j)); lab[i, j] = true
        while !isempty(stack)
            a, b = pop!(stack); sz += 1
            for (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                na, nb = a + da, b + db
                if 1 <= na <= L1 && 1 <= nb <= L2 && mask[na, nb] && !lab[na, nb]
                    lab[na, nb] = true; push!(stack, (na, nb))
                end
            end
        end
        push!(sizes, sz)
    end
    return sizes
end

"""
    cluster_size_distribution(cfgs; sign=+1) -> (sizes, max_frac)

For each cfg, list connected-component sizes of the `sign(φ)=sign` mask
(4-connectivity, open BC) and the largest-cluster fraction `s_max/V`.
Matches `cluster_size_distribution` in `dm_vs_hmc_observables_2d.py`.
"""
function cluster_size_distribution(cfgs::AbstractArray{<:Real,3}; sign::Int = +1)
    L1, L2, N = size(cfgs)
    V = L1 * L2
    allsizes = Int[]
    maxfrac = Float64[]
    for n in 1:N
        x = @view cfgs[:, :, n]
        mask = sign > 0 ? (x .> 0) : (x .< 0)
        sz = _label_sizes(mask)
        if isempty(sz)
            push!(maxfrac, 0.0)
        else
            append!(allsizes, sz)
            push!(maxfrac, maximum(sz) / V)
        end
    end
    return allsizes, maxfrac
end


# ===========================================================================
#  Autocorrelation + Sokal windowing (acf_utils.py)
# ===========================================================================

"""
    acf_per_chain_fft(series) -> rho

Normalised ACF for each column (chain) via FFT.  `series` is `(T, M)`;
`rho` is `(T, M)` with `rho[1, :] = 1`.  Matches `acf_per_chain_fft`.
"""
function acf_per_chain_fft(series::AbstractMatrix)
    T, M = size(series)
    x = series .- mean(series; dims = 1)
    n2 = nextpow(2, 2T)                       # 1 << ceil(log2(2T))
    xp = vcat(x, zeros(eltype(x), n2 - T, M))
    f = rfft(xp, 1)
    acf = irfft(abs2.(f), n2, 1)[1:T, :]      # f·conj(f) → autocovariance
    norm = collect(T .- (0:T-1))              # unbiased divisor [T, T-1, …, 1]
    acf = acf ./ norm
    return acf ./ acf[1:1, :]
end

"""
    sokal_window(rho_mean; c=5.0) -> (tau, W, tau_hat)

Sokal automatic windowing.  `tau_hat(W) = 1/2 + Σ_{t=1}^{W} rho(t)`; pick the
smallest `W ≥ 1` with `W ≥ c·tau_hat(W)`, else the last `W`.  Matches
`sokal_window`.
"""
function sokal_window(rho_mean::AbstractVector; c::Real = 5.0)
    T = length(rho_mean)
    tau_hat = 0.5 .+ cumsum(@view rho_mean[2:end])   # index W = 1 … T-1
    W = findfirst(W -> W >= c * tau_hat[W], 1:(T - 1))
    W === nothing && (W = T - 1)
    return tau_hat[W], W, tau_hat
end

"τ_int (Sokal) of a 1-D time series.  Matches `tau_int_series`."
function tau_int_series(series::AbstractVector; sokal_c::Real = 5.0)
    rho = vec(acf_per_chain_fft(reshape(collect(float.(series)), :, 1)))
    tau, W, _ = sokal_window(rho; c = sokal_c)
    return tau, W
end


# ===========================================================================
#  Jackknife estimators for the HMC table  (Z₂-symmetric, binned)
# ===========================================================================
#
# Every table observable is an even function of the field, so the physical Z₂
# symmetry (⟨φ⟩=0) is imposed directly through the even-moment cumulant formulas
# (odd moments set to 0) rather than by augmenting the ensemble with {φ,−φ};
# the two are equivalent for even observables.  Errors are binned jackknife,
# which (unlike a per-cfg bootstrap) also absorbs residual HMC autocorrelation
# through the binning.

"""
    zero_mode_observables_jk(M, V; n_bins=50) -> (mean, err)

Binned-jackknife of the zero-mode (per-cfg magnetisation M=φ̄) observables:
abs_M, chi=V(⟨M²⟩−⟨|M|⟩²), U4, and the Z₂ connected cumulants κ₂,₄,₆,₈ (odd
moments imposed zero).
"""
function zero_mode_observables_jk(M::AbstractVector, V::Integer; n_bins::Int = 50)
    cols = hcat(abs.(M), M .^ 2, M .^ 4, M .^ 6, M .^ 8)   # [|M|, m2, m4, m6, m8]
    zmcum(v, n) = cumulants_from_raw_moments([0.0, v[2], 0.0, v[3], 0.0, v[4], 0.0, v[5]])[n]
    specs = [("abs_M", v -> v[1]),
             ("chi",   v -> V * (v[2] - v[1]^2)),
             ("U4",    v -> 1 - v[3] / (3 * v[2]^2)),
             ("k2",    v -> zmcum(v, 2)), ("k4", v -> zmcum(v, 4)),
             ("k6",    v -> zmcum(v, 6)), ("k8", v -> zmcum(v, 8))]
    mean_d = Dict{String,Float64}(); err_d = Dict{String,Float64}()
    for (name, f) in specs
        mean_d[name], err_d[name] = binned_jackknife(cols, f; n_bins = n_bins)
    end
    return mean_d, err_d
end

"""
    site_cumulants_avg_jk(cfgs; n_bins=50) -> (mean, err)

Binned-jackknife of the single-site cumulants averaged over sites (persite_avg,
DM.tex Eq. eq:site_cumulants_avg): ⟨|φ|⟩, κ₂,₄,₆,₈^site, U4_site.  Per-site
κₙ(x) are built from the even moments m_r(x)=⟨φ(x)ʳ⟩ (Z₂: odd moments 0) and
then averaged over sites.
"""
function site_cumulants_avg_jk(cfgs::AbstractArray{<:Real,3}; n_bins::Int = 50)
    L1, L2, N = size(cfgs)
    V = L1 * L2
    n_bins = min(n_bins, N)
    bs = div(N, n_bins)
    flat = reshape(Array{Float64}(cfgs[:, :, 1:(bs * n_bins)]), V, bs * n_bins)
    bm = Dict(r => Array{Float64}(undef, V, n_bins) for r in (2, 4, 6, 8))
    babs = Array{Float64}(undef, V, n_bins)
    for b in 1:n_bins
        blk = @view flat[:, ((b - 1) * bs + 1):(b * bs)]
        babs[:, b] = vec(mean(abs.(blk); dims = 2))
        p2 = blk .^ 2
        bm[2][:, b] = vec(mean(p2; dims = 2));      bm[4][:, b] = vec(mean(p2 .^ 2; dims = 2))
        bm[6][:, b] = vec(mean(p2 .^ 3; dims = 2)); bm[8][:, b] = vec(mean(p2 .^ 4; dims = 2))
    end

    keys = ["abs_phi", "k2", "k4", "k6", "k8", "U4_site"]
    function stats(m2, m4, m6, m8, absm)            # each (V,) → spatial-avg scalars
        z = zeros(V)
        k = cumulants_from_raw_moments([z, m2, z, m4, z, m6, z, m8])
        return Dict("abs_phi" => mean(absm), "k2" => mean(k[2]), "k4" => mean(k[4]),
                    "k6" => mean(k[6]), "k8" => mean(k[8]),
                    "U4_site" => mean(@. 1 - m4 / (3 * m2^2)))
    end
    allr(r) = vec(mean(bm[r]; dims = 2))
    full = stats(allr(2), allr(4), allr(6), allr(8), vec(mean(babs; dims = 2)))
    tot = Dict(r => vec(sum(bm[r]; dims = 2)) for r in (2, 4, 6, 8))
    totabs = vec(sum(babs; dims = 2))
    jk = Dict(k => zeros(n_bins) for k in keys)
    for i in 1:n_bins
        loo(r) = (tot[r] .- bm[r][:, i]) ./ (n_bins - 1)
        s = stats(loo(2), loo(4), loo(6), loo(8), (totabs .- babs[:, i]) ./ (n_bins - 1))
        for k in keys
            jk[k][i] = s[k]
        end
    end
    mean_d = Dict(k => full[k] for k in keys)
    err_d = Dict(k => sqrt((n_bins - 1) * mean((jk[k] .- full[k]) .^ 2)) for k in keys)
    return mean_d, err_d
end

# ===========================================================================
#  Unbiased k-statistics + jackknife-debiased site cumulants
#  (port of site_cumulants_kstat.py)
# ===========================================================================

"""
    kstat_site(cfgs, n) -> Float64

Site-averaged unbiased k-statistic of order `n ∈ {2,4}`: the per-site (per-
column) k-statistic `scipy.stats.kstat(·, n)` averaged over sites.  Implemented
from the standard power-sum formulas (no Julia k-statistic package exists).
"""
function kstat_site(cfgs::AbstractArray{<:Real,3}, n::Int)
    N = size(cfgs, 3)
    S1 = dropdims(sum(cfgs; dims = 3); dims = 3)
    S2 = dropdims(sum(cfgs .^ 2; dims = 3); dims = 3)
    if n == 2
        ks = @. (N * S2 - S1^2) / (N * (N - 1))
    elseif n == 4
        S3 = dropdims(sum(cfgs .^ 3; dims = 3); dims = 3)
        S4 = dropdims(sum(cfgs .^ 4; dims = 3); dims = 3)
        ks = @. (-6 * S1^4 + 12 * N * S1^2 * S2 - 3 * N * (N - 1) * S2^2 -
                 4 * N * (N + 1) * S1 * S3 + N^2 * (N + 1) * S4) /
                (N * (N - 1) * (N - 2) * (N - 3))
    else
        error("kstat_site: only n=2,4 supported")
    end
    return mean(ks)
end

"""
    site_cumulants_kstat(cfgs; n_blocks=50) -> Dict

Single-site cumulants κₙ^site of the RAW field (no Z₂ augmentation):
κ₂,κ₄ as unbiased k-statistics; κ₆,κ₈ as delete-one-block jackknife-debiased
moment→cumulant estimates, cross-checked against the central-moment plug-in.
Port of `site_cumulants` in `site_cumulants_kstat.py`.
"""
function site_cumulants_kstat(cfgs::AbstractArray{<:Real,3}; n_blocks::Int = 50)
    L1, L2, N = size(cfgs)
    V = L1 * L2
    k2 = kstat_site(cfgs, 2)
    k4 = kstat_site(cfgs, 4)

    bs = div(N, n_blocks)
    nused = bs * n_blocks
    # per-block raw power sums S[r][b] : (V,) for r=1..8
    flat = reshape(Array{Float64}(cfgs[:, :, 1:nused]), V, nused)
    S = [Array{Float64}(undef, V, n_blocks) for _ in 1:8]
    for b in 1:n_blocks
        blk = @view flat[:, ((b - 1) * bs + 1):(b * bs)]
        for r in 1:8
            S[r][:, b] = vec(sum(blk .^ r; dims = 2))
        end
    end
    tot = [vec(sum(S[r]; dims = 2)) for r in 1:8]   # (V,)

    # κ₆,κ₈ from raw power sums divided by count, spatially averaged.
    function k6k8(rawsum, nn)
        ks = cumulants_from_raw_moments([rawsum[r] ./ nn for r in 1:8])
        return (mean(ks[6]), mean(ks[8]))
    end
    full6, full8 = k6k8(tot, nused)
    jk = Array{Float64}(undef, n_blocks, 2)
    for b in 1:n_blocks
        rs = [tot[r] .- S[r][:, b] for r in 1:8]
        jk[b, 1], jk[b, 2] = k6k8(rs, nused - bs)
    end
    jkmean = vec(mean(jk; dims = 1))
    bc = n_blocks .* [full6, full8] .- (n_blocks - 1) .* jkmean
    err = [sqrt((n_blocks - 1) / n_blocks * sum((jk[:, c] .- jkmean[c]) .^ 2)) for c in 1:2]

    # cross-check the plug-in against the central-moment formula (no scipy needed)
    mu = vec(mean(flat; dims = 2))
    cm = [vec(mean((flat .- mu) .^ r; dims = 2)) for r in 2:8]   # cm[r-1] = central moment of order r
    m2c, m3c, m4c, m5c, m6c, m7c, m8c = cm
    chk6 = mean(@. m6c - 15 * m4c * m2c - 10 * m3c^2 + 30 * m2c^3)
    chk8 = mean(@. m8c - 28 * m6c * m2c - 56 * m5c * m3c - 35 * m4c^2 +
                   420 * m4c * m2c^2 + 560 * m3c^2 * m2c - 630 * m2c^4)

    return Dict("k2" => k2, "k4" => k4,
                "k6" => bc[1], "k6_err" => err[1], "k6_plug" => full6, "k6_scipy" => chk6,
                "k8" => bc[2], "k8_err" => err[2], "k8_plug" => full8, "k8_scipy" => chk8)
end

end # module
