# YukawaFermionHMC2D.jl
# Minimal two-flavor pseudofermion HMC for the 2D staggered Yukawa model
# in Albergo et al., arXiv:2106.05934.
#
# Target action:
#   S(phi, varphi) = S_B(phi) + varphi' * inv(M(phi)) * varphi
#   M(phi) = D(phi) * D(phi)'
#   varphi = D(phi) * chi,   chi ~ exp(-chi' chi)
#
# This version is intentionally simple. It uses explicit sparse matrices and
# sparse Cholesky solves. This is fine for the 16x16 benchmark in the paper.
# For larger lattices, replace solve_M by a matrix-free CG solve and add
# even-odd preconditioning.

using Random
using Statistics
using LinearAlgebra
using SparseArrays
using Printf

@inline pbc(i, N) = i == N + 1 ? 1 : (i == 0 ? N : i)

@inline function apbc(i, N)
    if i == N + 1
        return 1, -1.0
    elseif i == 0
        return N, -1.0
    else
        return i, 1.0
    end
end

@inline lin(x, t, N) = x + (t - 1) * N
@inline eta2(x) = isodd(x - 1) ? -1.0 : 1.0

function complex_gaussian(rng::AbstractRNG, V::Int)
    return (randn(rng, V) .+ im .* randn(rng, V)) ./ sqrt(2.0)
end

# D_xy(phi) for 2D staggered fermions.
# x direction is periodic. t direction is antiperiodic for fermions.
function build_D(phi::AbstractMatrix{Float64}, g::Float64, mf::Float64)
    N = size(phi, 1)
    V = N * N
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, 5 * V)
    sizehint!(cols, 5 * V)
    sizehint!(vals, 5 * V)

    @inbounds for t in 1:N
        tp, stp = apbc(t + 1, N)
        tm, stm = apbc(t - 1, N)
        for x in 1:N
            xp = pbc(x + 1, N)
            xm = pbc(x - 1, N)
            i = lin(x, t, N)

            # spatial derivative, eta_1(x)=1
            push!(rows, i); push!(cols, lin(xp, t, N)); push!(vals,  0.5)
            push!(rows, i); push!(cols, lin(xm, t, N)); push!(vals, -0.5)

            # temporal derivative, eta_2(x)=(-1)^(x-1), antiperiodic in time
            e2 = eta2(x)
            push!(rows, i); push!(cols, lin(x, tp, N)); push!(vals,  0.5 * e2 * stp)
            push!(rows, i); push!(cols, lin(x, tm, N)); push!(vals, -0.5 * e2 * stm)

            # local mass plus Yukawa term
            push!(rows, i); push!(cols, i); push!(vals, mf + g * phi[x, t])
        end
    end

    return sparse(rows, cols, vals, V, V)
end

# S_B = sum_x [-2 sum_mu phi_x phi_{x+mu} + (m2+4) phi_x^2 + lambda phi_x^4]
function boson_action(phi::AbstractMatrix{Float64}, m2::Float64, lambda::Float64)
    N = size(phi, 1)
    s = 0.0
    @inbounds for t in 1:N
        tp = pbc(t + 1, N)
        for x in 1:N
            xp = pbc(x + 1, N)
            f = phi[x, t]
            s += -2.0 * f * (phi[xp, t] + phi[x, tp]) + (m2 + 4.0) * f^2 + lambda * f^4
        end
    end
    return s
end

# Force is -dS_B/dphi.
function boson_force(phi::AbstractMatrix{Float64}, m2::Float64, lambda::Float64)
    N = size(phi, 1)
    F = similar(phi)
    @inbounds for t in 1:N
        tp = pbc(t + 1, N)
        tm = pbc(t - 1, N)
        for x in 1:N
            xp = pbc(x + 1, N)
            xm = pbc(x - 1, N)
            f = phi[x, t]
            nn = phi[xp, t] + phi[xm, t] + phi[x, tp] + phi[x, tm]
            F[x, t] = 2.0 * nn - 2.0 * (m2 + 4.0) * f - 4.0 * lambda * f^3
        end
    end
    return F
end

# ---------------------------------------------------------------------------
# Matrix-free application of D, D^T and M = D D^T using the staggered stencil.
# No explicit matrix is built. Vectors are length V, indexed by lin(x,t,N).
# D = Hop + diag(mf + g phi), with Hop antisymmetric, so D^T = -Hop + diag.
# ---------------------------------------------------------------------------
function apply_D!(out::AbstractVector, v::AbstractVector,
                  phi::AbstractMatrix{Float64}, g::Float64, mf::Float64)
    N = size(phi, 1)
    @inbounds for t in 1:N
        tp, stp = apbc(t + 1, N)
        tm, stm = apbc(t - 1, N)
        for x in 1:N
            xp = pbc(x + 1, N); xm = pbc(x - 1, N)
            i = lin(x, t, N)
            e2 = eta2(x)
            out[i] = 0.5 * (v[lin(xp, t, N)] - v[lin(xm, t, N)]) +
                     0.5 * e2 * (stp * v[lin(x, tp, N)] - stm * v[lin(x, tm, N)]) +
                     (mf + g * phi[x, t]) * v[i]
        end
    end
    return out
end

function apply_Dt!(out::AbstractVector, v::AbstractVector,
                   phi::AbstractMatrix{Float64}, g::Float64, mf::Float64)
    N = size(phi, 1)
    @inbounds for t in 1:N
        tp, stp = apbc(t + 1, N)
        tm, stm = apbc(t - 1, N)
        for x in 1:N
            xp = pbc(x + 1, N); xm = pbc(x - 1, N)
            i = lin(x, t, N)
            e2 = eta2(x)
            out[i] = -0.5 * (v[lin(xp, t, N)] - v[lin(xm, t, N)]) -
                      0.5 * e2 * (stp * v[lin(x, tp, N)] - stm * v[lin(x, tm, N)]) +
                      (mf + g * phi[x, t]) * v[i]
        end
    end
    return out
end

# M v = D (D^T v).  tmp is scratch of the same eltype/length as v.
function apply_M!(out::AbstractVector, v::AbstractVector, tmp::AbstractVector,
                  phi::AbstractMatrix{Float64}, g::Float64, mf::Float64)
    apply_Dt!(tmp, v, phi, g, mf)
    apply_D!(out, tmp, phi, g, mf)
    return out
end

# Conjugate gradient for the real SPD system A x = b, A applied via applyA!(out, p).
function cg_solve(applyA!, b::Vector{Float64}; tol::Float64=1e-10, maxiter::Int=5000)
    x = zeros(length(b))
    r = copy(b)
    p = copy(r)
    Ap = similar(b)
    rsold = dot(r, r)
    bnorm = sqrt(rsold)
    bnorm == 0.0 && return x
    for _ in 1:maxiter
        applyA!(Ap, p)
        alpha = rsold / dot(p, Ap)
        @. x += alpha * p
        @. r -= alpha * Ap
        rsnew = dot(r, r)
        sqrt(rsnew) <= tol * bnorm && break
        @. p = r + (rsnew / rsold) * p
        rsold = rsnew
    end
    return x
end

# Solve M(phi) u = rhs with M = D D^T, matrix-free CG.
# M is real, so the real and imaginary parts of the complex rhs are solved separately.
function solve_M(phi::AbstractMatrix{Float64}, rhs::Vector{ComplexF64},
                 g::Float64, mf::Float64; tol::Float64=1e-10)
    V = length(rhs)
    tmp = Vector{Float64}(undef, V)
    applyA! = (out, p) -> apply_M!(out, p, tmp, phi, g, mf)
    ur = cg_solve(applyA!, real(rhs); tol=tol)
    ui = cg_solve(applyA!, imag(rhs); tol=tol)
    return ComplexF64.(ur, ui)
end

# Reference dense/sparse Cholesky solve, kept for cross-checking the CG path.
function solve_M_chol(phi::AbstractMatrix{Float64}, rhs::Vector{ComplexF64},
                      g::Float64, mf::Float64)
    D = build_D(phi, g, mf)
    F = cholesky(Symmetric(D * transpose(D)); check=false)
    return ComplexF64.(F \ real(rhs), F \ imag(rhs))
end

function pseudofermion_action(phi::AbstractMatrix{Float64}, varphi::Vector{ComplexF64},
                              g::Float64, mf::Float64)
    u = solve_M(phi, varphi, g, mf)
    return real(dot(varphi, u))
end

# For M = D D^T and D = D0 + diag(mf + g phi),
# F_pf(x) = -dS_pf/dphi_x = 2g Re[conj(u_x) * (D^T u)_x], u = M^{-1} varphi.
function total_force(phi::AbstractMatrix{Float64}, varphi::Vector{ComplexF64},
                     m2::Float64, lambda::Float64, g::Float64, mf::Float64)
    N = size(phi, 1)
    F = boson_force(phi, m2, lambda)
    u = solve_M(phi, varphi, g, mf)
    r = Vector{ComplexF64}(undef, length(u))
    apply_Dt!(r, u, phi, g, mf)
    pf = 2.0 * g .* real.(conj.(u) .* r)
    @inbounds for t in 1:N, x in 1:N
        F[x, t] += pf[lin(x, t, N)]
    end
    return F
end

function hamiltonian(phi::AbstractMatrix{Float64}, mom::AbstractMatrix{Float64},
                     varphi::Vector{ComplexF64}, m2::Float64, lambda::Float64,
                     g::Float64, mf::Float64)
    return 0.5 * sum(abs2, mom) + boson_action(phi, m2, lambda) +
           pseudofermion_action(phi, varphi, g, mf)
end

function leapfrog!(phi::AbstractMatrix{Float64}, mom::AbstractMatrix{Float64},
                   varphi::Vector{ComplexF64}, eps::Float64, n_steps::Int,
                   m2::Float64, lambda::Float64, g::Float64, mf::Float64)
    F = total_force(phi, varphi, m2, lambda, g, mf)
    @. mom += 0.5 * eps * F

    for step in 1:n_steps
        @. phi += eps * mom
        if step != n_steps
            F = total_force(phi, varphi, m2, lambda, g, mf)
            @. mom += eps * F
        end
    end

    F = total_force(phi, varphi, m2, lambda, g, mf)
    @. mom += 0.5 * eps * F
    return nothing
end

function hmc_step!(rng::AbstractRNG, phi::AbstractMatrix{Float64}, eps::Float64, n_steps::Int,
                   m2::Float64, lambda::Float64, g::Float64, mf::Float64)
    N = size(phi, 1)
    V = N * N

    # Pseudofermion heatbath at the beginning of the trajectory: varphi = D chi.
    chi = complex_gaussian(rng, V)
    varphi = Vector{ComplexF64}(undef, V)
    apply_D!(varphi, chi, phi, g, mf)

    mom = randn(rng, N, N)
    phi_old = copy(phi)
    H_old = hamiltonian(phi, mom, varphi, m2, lambda, g, mf)

    leapfrog!(phi, mom, varphi, eps, n_steps, m2, lambda, g, mf)

    H_new = hamiltonian(phi, mom, varphi, m2, lambda, g, mf)
    dH = H_new - H_old

    if log(rand(rng)) < -dH
        return true, dH
    else
        copyto!(phi, phi_old)
        return false, dH
    end
end

# Exact but expensive. Use mainly for small lattices or occasional checks.
function chiral_condensate(phi::AbstractMatrix{Float64}, g::Float64, mf::Float64)
    D = Matrix(build_D(phi, g, mf))
    V = size(D, 1)
    Id = Matrix{Float64}(I, V, V)
    return real(tr(D \ Id)) / V
end

# ============================================================================
# Even-odd preconditioned C2 trace-estimator fermion force.
# This is the estimator behind the paper's Table II force ratio (App. B + C):
#   even-odd reduced operator Q = C (A C^{-1} E - B), A=diag(mf+g phi_odd),
#   E=diag(mf+g phi_even), B=D_oe, C=D_eo (constant); and the C2 gradient
#   estimator  d log det(Q Q^T) = (M^{-1} chi)^dagger (dM) chi,  chi unit Gaussian.
# Constant hopping blocks depend only on N; precompute once per run with eo_blocks.
# ============================================================================
function eo_blocks(N::Int)
    odd = Int[]; even = Int[]
    for t in 1:N, x in 1:N
        ((x + t) % 2 == 0) ? push!(even, lin(x, t, N)) : push!(odd, lin(x, t, N))
    end
    D0 = Matrix(build_D(zeros(N, N), 0.0, 0.0))   # pure hopping (g=0, mf=0)
    return (odd=odd, even=even, B=D0[odd, even], C=D0[even, odd], Cinv=inv(D0[even, odd]))
end

# Accumulate one bilinear Re[y' (dQ/dphi_x) z] for Q = C (A Cinv E - B): yy=C'y, zz=z.
function _eo_c2_bilinear!(F::Vector{Float64}, y::Vector{ComplexF64}, z::Vector{ComplexF64},
                          ao::Vector{Float64}, ae::Vector{Float64}, eo, g::Float64)
    yy = transpose(eo.C) * y
    @inbounds F[eo.odd]  .+= g .* real.(conj.(yy) .* (eo.Cinv * (ae .* z)))
    @inbounds F[eo.even] .+= g .* real.(conj.(transpose(eo.Cinv) * (ao .* yy)) .* z)
    return F
end

function eo_c2_force(phi::AbstractMatrix{Float64}, eo, rng::AbstractRNG, g::Float64, mf::Float64)
    V = length(phi); n = length(eo.odd)
    ao = mf .+ g .* vec(phi)[eo.odd]
    ae = mf .+ g .* vec(phi)[eo.even]
    Q = eo.C * ((Diagonal(ao) * eo.Cinv) * Diagonal(ae) .- eo.B)   # = C (A Cinv E - B)
    chi = complex_gaussian(rng, n)
    w = (Q * transpose(Q)) \ chi                                   # w = (Q Q^T)^{-1} chi
    F = zeros(Float64, V)
    _eo_c2_bilinear!(F, w,   transpose(Q) * chi, ao, ae, eo, g)    # bilinear(w, Q^T chi)
    _eo_c2_bilinear!(F, chi, transpose(Q) * w,   ao, ae, eo, g)    # bilinear(chi, Q^T w)
    return F
end

# <||F_B||>, <||F_eo_c2||> using the paper's even-odd C2 trace-estimator force.
function force_norms_eo_c2(phi::AbstractMatrix{Float64}, eo, rng::AbstractRNG,
                           m2::Float64, lambda::Float64, g::Float64, mf::Float64)
    return norm(vec(boson_force(phi, m2, lambda))), norm(eo_c2_force(phi, eo, rng, g, mf))
end

stderr_of(x) = std(x) / sqrt(length(x))

function run_yukawa_hmc(; N::Int=16,
    m2::Float64=-4.00,
    lambda::Float64=6.0,
    g::Float64=0.1,
    mf::Float64=0.0,
    n_therm::Int=2000,
    n_samples::Int=8000,
    eps::Float64=0.11,
    n_steps::Int=18,
    seed::Int=1234,
    measure_chiral::Bool=false,
    measure_force::Bool=false,
    print_every::Int=100)

    rng = MersenneTwister(seed)
    phi = 0.1 .* randn(rng, N, N)
    eo = measure_force ? eo_blocks(N) : nothing
    absM = Vector{Float64}(undef, n_samples)
    absChiral = measure_chiral ? Vector{Float64}(undef, n_samples) : Float64[]
    fbN  = measure_force ? Vector{Float64}(undef, n_samples) : Float64[]
    fpfN = measure_force ? Vector{Float64}(undef, n_samples) : Float64[]

    n_accept_prod = 0
    n_total = n_therm + n_samples

    println("="^72)
    println("2D staggered Yukawa pseudofermion HMC")
    @printf("N=%d, V=%d, m2=%.4f, lambda=%.4f, g=%.4f, mf=%.4f\n", N, N*N, m2, lambda, g, mf)
    @printf("n_therm=%d, n_samples=%d, eps=%.6f, n_steps=%d\n", n_therm, n_samples, eps, n_steps)
    println("="^72)

    for step in 1:n_total
        accepted, dH = hmc_step!(rng, phi, eps, n_steps, m2, lambda, g, mf)

        if step > n_therm
            k = step - n_therm
            absM[k] = abs(mean(phi))
            if measure_chiral
                absChiral[k] = abs(chiral_condensate(phi, g, mf))
            end
            if measure_force
                fbN[k], fpfN[k] = force_norms_eo_c2(phi, eo, rng, m2, lambda, g, mf)
            end
            n_accept_prod += accepted ? 1 : 0
        end

        if print_every > 0 && step % print_every == 0
            phase = step <= n_therm ? "therm" : "prod"
            acc_so_far = step <= n_therm ? NaN : n_accept_prod / (step - n_therm)
            @printf("step %6d / %6d  %-5s  dH=% .3e  accepted=%s  prod_acc=%.3f\n",
                    step, n_total, phase, dH, string(accepted), acc_so_far)
        end
    end

    acc_rate = n_accept_prod / n_samples
    force_ratio = measure_force ? mean(fpfN) / mean(fbN) : NaN

    println("\nProduction summary")
    @printf("acceptance = %.4f\n", acc_rate)
    @printf("<|M|>      = %.8f +/- %.8f\n", mean(absM), stderr_of(absM))
    if measure_chiral
        @printf("<|psibar psi|> = %.8f +/- %.8f\n", mean(absChiral), stderr_of(absChiral))
    end
    if measure_force
        @printf("<||F_B||>  = %.6f   <||F_ferm||> = %.6f   force ratio = %.4f (%.1f%%)  [even-odd C2 trace, paper Table II]\n",
                mean(fbN), mean(fpfN), force_ratio, 100 * force_ratio)
    end

    return (phi=phi, absM=absM, absChiral=absChiral, accept_rate=acc_rate,
            force_ratio=force_ratio, fbN=fbN, fpfN=fpfN,
            N=N, m2=m2, lambda=lambda, g=g, mf=mf, eps=eps, n_steps=n_steps)
end

# Tuned integrator: trajectory length tau = eps*n_steps ~ 2.0, acceptance ~ 0.72.
function run_paper_g01(; kwargs...)
    return run_yukawa_hmc(; N=16, m2=-4.00, lambda=6.0, g=0.1, mf=0.0,
                          eps=0.11, n_steps=18, kwargs...)
end

# Tuned integrator: trajectory length tau = eps*n_steps ~ 1.95, acceptance ~ 0.73.
function run_paper_g03(; kwargs...)
    return run_yukawa_hmc(; N=16, m2=-1.55, lambda=2.4, g=0.3, mf=0.0,
                          eps=0.13, n_steps=15, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # A short smoke run. Increase n_therm and n_samples for physics measurements.
    result = run_paper_g01(n_therm=50, n_samples=50, measure_chiral=true,
                           measure_force=true, print_every=25)
end
