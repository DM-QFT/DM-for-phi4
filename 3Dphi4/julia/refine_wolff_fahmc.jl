# Refine an ensemble of 3D phi^4 configurations with short FA-HMC + Wolff
# chains, one independent chain per configuration.
#
# Reuses the kernels of 4Dphi4/WolffFAHMC_ND.jl (fahmc_step!, wolff_flip!,
# FourierMassMatrix). The HMC step size is dual-averaging-adapted on a
# discarded copy of config 1, then frozen for all chains (fixed kernel,
# exact MH). Metric is :fixed with m2_eff.
#
# Usage:
#   julia -t 32 julia/refine_wolff_fahmc.jl IN.bin OUT.bin Ncfg L kappa lambda n_cycles
#
# IN/OUT are raw little-endian Float64 streams laid out as Julia
# Array{Float64}(N, N, N, Ncfg) column-major (config index slowest).

include("/work/gw97/w97000/project/DM/4Dphi4/WolffFAHMC_ND.jl")

using Random, Printf
using Base.Threads: @threads, nthreads, threadid

const IN   = ARGS[1]
const OUT  = ARGS[2]
const NCFG = parse(Int, ARGS[3])
const N    = parse(Int, ARGS[4])
const KAPPA  = parse(Float64, ARGS[5])
const LAMBDA = parse(Float64, ARGS[6])
const NCYC = parse(Int, ARGS[7])

const M2_EFF  = 0.01   # ~ finite-volume gap at L=64; keeps FA active in the IR
const N_LEAP  = 50
const N_WOLFF = 5
const DELTA_TARGET = 0.5    # dual-averaging target acceptance
# metric mode: "m2" = fixed M(k)=p_hat^2+M2_EFF; "dm" = nonparametric
# M(k) = 1/G(k) with G measured from the input (DM) ensemble itself.
const METRIC = length(ARGS) >= 8 ? ARGS[8] : "m2"
# optional fixed step size (skips dual-averaging adaptation when > 0)
const EPS_FIXED = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 0.0

println("refine_wolff_fahmc: N=$N D=3 Ncfg=$NCFG kappa=$KAPPA lambda=$LAMBDA " *
        "cycles=$NCYC leapfrog=$N_LEAP n_wolff=$N_WOLFF threads=$(nthreads())")

phi_all = Array{Float64}(undef, N, N, N, NCFG)
open(IN, "r") do io
    read!(io, phi_all)
end
lat = Lattice{3}(N)

# ---- optional: nonparametric FA metric from the input ensemble ----
# M(k) = 1/G(k) equalizes the relaxation rates of all momentum modes.
# G is the mode variance of the input (DM) ensemble, whose propagator is
# accurate across the full momentum range.
Mk_dm = nothing
if METRIC == "dm"
    local Gd = nothing
    for n in 1:NCFG
        spec = abs2.(rfft(phi_all[:, :, :, n]))
        Gd === nothing ? (Gd = spec) : (Gd .+= spec)
    end
    Gd ./= (NCFG * lat.V)
    global Mk_dm = max.(1.0 ./ Gd, 1e-8)
    @printf("DM-propagator metric: M(k) range [%.3e, %.3e]  cond=%.1f\n",
            minimum(Mk_dm), maximum(Mk_dm), maximum(Mk_dm) / minimum(Mk_dm))
end

set_metric!(fm) = if Mk_dm !== nothing
    fm.M_k .= Mk_dm
    fm.sqrtM_k .= sqrt.(Mk_dm)
    fm.invM_k .= 1.0 ./ Mk_dm
end

# ---- step-size adaptation on a discarded copy of config 1 ----
if EPS_FIXED > 0
    global EPS = EPS_FIXED
    @printf("fixed eps = %.6f (adaptation skipped)\n", EPS)
else
begin
    rng = MersenneTwister(1)
    fm  = FourierMassMatrix{3}(N, M2_EFF)
    set_metric!(fm)
    ws  = WolffWorkspace{3}(N)
    phi = phi_all[:, :, :, 1]
    pif = similar(phi); phio = similar(phi)
    fbuf = similar(phi); abuf = similar(phi)
    # start near the known stability region so dual averaging converges to
    # the target instead of recovering from early divergences
    da = DualAveraging(0.02; δ_target = DELTA_TARGET)
    global EPS = 0.02
    for s in 1:150
        nact = rand(rng, round(Int, 0.8 * N_LEAP):round(Int, 1.2 * N_LEAP))
        _, dH = fahmc_step!(rng, phi, pif, fm, EPS, nact, KAPPA, LAMBDA,
                            phio, fbuf, abuf)
        ap = (isnan(dH) || isinf(dH)) ? 0.0 : min(1.0, exp(-dH))
        global EPS = da_update!(da, ap)
        for _ in 1:N_WOLFF
            wolff_flip!(phi, KAPPA, rng, ws, lat)
        end
    end
    global EPS = da_final_ε(da)
    @printf("adapted eps = %.6f\n", EPS)
end
end

# ---- per-thread workspaces (created outside the loop; :static schedule) ----
# maxthreadid: threadid() can exceed nthreads() when an interactive pool exists
nt = Threads.maxthreadid()
fms  = [FourierMassMatrix{3}(N, M2_EFF) for _ in 1:nt]
foreach(set_metric!, fms)
wss  = [WolffWorkspace{3}(N) for _ in 1:nt]
bufs = [(Array{Float64}(undef, N, N, N), Array{Float64}(undef, N, N, N),
         Array{Float64}(undef, N, N, N), Array{Float64}(undef, N, N, N))
        for _ in 1:nt]

acc_cnt = zeros(Int, NCFG)
@threads :static for n in 1:NCFG
    tid = threadid()
    fm = fms[tid]; ws = wss[tid]
    pif, phio, fbuf, abuf = bufs[tid]
    rng = MersenneTwister(10_000 + n)
    phi = phi_all[:, :, :, n]
    for s in 1:NCYC
        nact = rand(rng, round(Int, 0.8 * N_LEAP):round(Int, 1.2 * N_LEAP))
        acc, _ = fahmc_step!(rng, phi, pif, fm, EPS, nact, KAPPA, LAMBDA,
                             phio, fbuf, abuf)
        acc && (acc_cnt[n] += 1)
        for _ in 1:N_WOLFF
            wolff_flip!(phi, KAPPA, rng, ws, lat)
        end
    end
    phi_all[:, :, :, n] .= phi
end

@printf("FA-HMC acceptance over refinement: %.4f\n",
        sum(acc_cnt) / (NCFG * NCYC))
open(OUT, "w") do io
    write(io, phi_all)
end
println("wrote $OUT")
