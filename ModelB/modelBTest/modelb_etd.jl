using CUDA, CUDA.CUFFT, LinearAlgebra, Random

include(joinpath(@__DIR__, "modelb_fast.jl"))

# Exponential-integrator (ETD1) Model B, state carried as the unnormalised
# half-spectrum phi_k = sum_x phi_x exp(-ikx).
#
# Splitting: dphi_k = -lambda_k phi_k dt + N_k dt + noise, with
#   khat2       = 4(sin^2(kx/2) + sin^2(ky/2))          lattice -Laplacian
#   lambda_k    = khat2 (2 kappa khat2 + m02) / gamma    linear (stiff) part
#   N_k         = -khat2 * c3 * (phi^3)_k / gamma        nonlinear part
# The linear drift and the conserved noise are integrated exactly over dt:
#   phi_k <- e^{-lambda dt} phi_k + dt g1(lambda dt) N_k + sqrt(V_k) xi
#   V_k   = (2T/gamma) khat2 N dt g2(lambda dt)
# so every Gaussian mode keeps its exact stationary variance T/(m02+2 kappa khat2)
# at any dt.  Only the phi^3 coupling carries a step-size error.
#
# lambda_k < 0 for the long-wavelength spinodal modes; the same expressions hold.

# The propagator is applied as u + expm1(-x)*u, never as exp(-x)*u.  The slowest
# modes have |x| ~ 1e-6, so exp(-x) sits ~20 ulp from 1.0 in Float32 and storing
# it as a factor quantises the decay rate by up to 2.5%.  Those modes are
# linearly unstable near kappa_c and need ~1e5 time units to relax, so the rate
# error accumulates unchecked and moves S(k) by several percent at t=2000.
@inline g1(x) = abs(x) < 1.0f-4 ? 1.0f0 - 0.5f0 * x : -expm1(-x) / x
@inline g2(x) = abs(x) < 1.0f-4 ? 1.0f0 - x : -expm1(-2.0f0 * x) / (2.0f0 * x)

@inline function etd_mode(i, j, L, two_kappa, m02, inv_gamma)
    khat2 = 4.0f0 * (sinpi(Float32(i) / L)^2 + sinpi(Float32(j) / L)^2)
    return khat2, inv_gamma * khat2 * (two_kappa * khat2 + m02)
end

# Columns kx=0 and kx=L/2 store both k and -k, so those two entries are not
# independent: they must stay exact conjugates.  The antisymmetric part of such a
# column is invisible to the nonlinear term (the FFT round trip symmetrises it)
# and to the noise, so it only ever gets multiplied by exp(-lambda dt).  For
# kappa=0.28 the long-wavelength modes have lambda<0, so fp32 round-off in that
# unphysical component grows like exp(0.046 t) and destroys the run around t~600.
# Every step therefore updates only the canonical entry (2*ky <= L) and writes the
# mirror as its conjugate, which projects the spurious component out.
@inline function etd_canonical(i, j, H, L)
    edge = (i == Int32(0)) | (i == H - Int32(1))
    selfconj = edge & ((j == Int32(0)) | (Int32(2) * j == L))
    skip = edge & (Int32(2) * j > L)          # handled by its canonical partner
    mirror = edge & !selfconj & !skip
    return selfconj, skip, mirror
end

@inline function etd_noise(i, j, e, H, L, step, k0, k1, selfconj)
    idx = reinterpret(Int32, UInt32(i + H * (j + L * e)))
    a, b, _, _ = philox4x32(reinterpret(UInt32, idx), step, 0x1, 0x0, k0, k1, Val(10))
    re, im = box_muller(a, b)
    # a self-conjugate mode is real and carries the full variance of a pair
    return selfconj ? (re * 1.4142136f0, 0.0f0) : (re, im)
end

function etd_update_kernel!(
    uk, cubek, dt, two_kappa, m02, c3, inv_gamma, noise_power,
    step, k0, k1, H, L, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        z = linear - Int32(1)
        i = z % H
        j = (z ÷ H) % L
        e = z ÷ (H * L)
        selfconj, skip, mirror = etd_canonical(i, j, H, L)
        skip && continue
        khat2, λ = etd_mode(i, j, L, two_kappa, m02, inv_gamma)
        x = λ * dt
        # 0.5 splits the variance between Re and Im of a non-self-conjugate mode.
        σ = sqrt(0.5f0 * noise_power * khat2 * Float32(L * L) * dt * g2(x))
        re, im = etd_noise(i, j, e, H, L, step, k0, k1, selfconj)
        nk = (-inv_gamma * c3 * khat2 * dt * g1(x)) * cubek[linear]
        old = uk[linear]
        v = old + expm1(-x) * old + nk + Complex(σ * re, σ * im)
        selfconj && (v = Complex(real(v), 0.0f0))
        uk[linear] = v
        mirror && (uk[linear + H * (L - Int32(2) * j)] = conj(v))
    end
    return nothing
end

# brfft leaves the L^2 factor in, so fold it in while cubing.
function etd_cube_kernel!(r, inv_n, n)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        v = r[linear] * inv_n
        r[linear] = v * v * v
    end
    return nothing
end

struct ETDPlan{P, Q, R, C}
    forward::P
    backward::Q
    real::R
    cubek::C
    L::Int
    H::Int
end

function etd_plan(u)
    L = size(u, 1)
    r = similar(u)
    forward = plan_rfft(r, (1, 2))
    cubek = forward * r
    backward = plan_brfft(cubek, L, (1, 2))
    return ETDPlan(forward, backward, r, cubek, L, L ÷ 2 + 1)
end

etd_spectrum(plan, u) = plan.forward * u

function etd_state(plan, uk)
    mul!(plan.real, plan.backward, uk)
    return plan.real ./ Float32(plan.L^2)
end

function etd_step!(uk, plan, step, key, dt, two_kappa, m02, c3, inv_gamma, noise_power)
    L, H = plan.L, plan.H
    nr = Int32(length(plan.real))
    nk = Int32(length(uk))
    mul!(plan.real, plan.backward, uk)
    @cuda threads=1024 blocks=min(cld(length(plan.real), 1024), 264) etd_cube_kernel!(
        plan.real, 1.0f0 / Float32(L * L), nr)
    mul!(plan.cubek, plan.forward, plan.real)
    @cuda threads=256 blocks=min(cld(length(uk), 256), 1056) etd_update_kernel!(
        uk, plan.cubek, dt, two_kappa, m02, c3, inv_gamma, noise_power,
        UInt32(step), key[1], key[2], Int32(H), Int32(L), nk)
    return nothing
end

# ------------------------------------------------------------------- ETDRK2
# Cox--Matthews predictor/corrector: second order in the nonlinear part, at the
# cost of a second phi^3 evaluation (4 FFTs per step).  The noise enters only the
# predictor, so the linear part and the noise stay exact.
# (exp(-x)-1+x)/x^2 loses ~eps/x^2 to cancellation, which is 24% at x=1e-3 in
# Float32.  expm1 plus a wider series branch keeps it below 1e-5 everywhere.
@inline g2c(x) = abs(x) < 0.06f0 ? 0.5f0 - x / 6.0f0 + x * x / 24.0f0 :
                 (expm1(-x) + x) / (x * x)

function etdrk2_predict_kernel!(
    ak, uk, cube1, dt, two_kappa, m02, c3, inv_gamma, noise_power,
    step, k0, k1, H, L, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        z = linear - Int32(1)
        i = z % H; j = (z ÷ H) % L; e = z ÷ (H * L)
        selfconj, skip, mirror = etd_canonical(i, j, H, L)
        skip && continue
        khat2, λ = etd_mode(i, j, L, two_kappa, m02, inv_gamma)
        x = λ * dt
        σ = sqrt(0.5f0 * noise_power * khat2 * Float32(L * L) * dt * g2(x))
        re, im = etd_noise(i, j, e, H, L, step, k0, k1, selfconj)
        old = uk[linear]
        v = old + expm1(-x) * old +
            (-inv_gamma * c3 * khat2 * dt * g1(x)) * cube1[linear] +
            Complex(σ * re, σ * im)
        selfconj && (v = Complex(real(v), 0.0f0))
        ak[linear] = v
        mirror && (ak[linear + H * (L - Int32(2) * j)] = conj(v))
    end
    return nothing
end

function etdrk2_correct_kernel!(
    uk, ak, cube1, cube2, dt, two_kappa, m02, c3, inv_gamma, H, L, n,
)
    id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:n
        z = linear - Int32(1)
        i = z % H; j = (z ÷ H) % L
        selfconj, skip, mirror = etd_canonical(i, j, H, L)
        skip && continue
        khat2, λ = etd_mode(i, j, L, two_kappa, m02, inv_gamma)
        v = ak[linear] + (-inv_gamma * c3 * khat2 * dt * g2c(λ * dt)) *
                         (cube2[linear] - cube1[linear])
        selfconj && (v = Complex(real(v), 0.0f0))
        uk[linear] = v
        mirror && (uk[linear + H * (L - Int32(2) * j)] = conj(v))
    end
    return nothing
end

function etdrk2_step!(uk, plan, ak, cube1, step, key, dt,
                      two_kappa, m02, c3, inv_gamma, noise_power)
    L, H = plan.L, plan.H
    nr = Int32(length(plan.real)); nk = Int32(length(uk))
    rb = min(cld(length(plan.real), 1024), 264)
    kb = min(cld(length(uk), 256), 1056)
    invn = 1.0f0 / Float32(L * L)

    mul!(plan.real, plan.backward, uk)
    @cuda threads=1024 blocks=rb etd_cube_kernel!(plan.real, invn, nr)
    mul!(cube1, plan.forward, plan.real)
    @cuda threads=256 blocks=kb etdrk2_predict_kernel!(
        ak, uk, cube1, dt, two_kappa, m02, c3, inv_gamma, noise_power,
        UInt32(step), key[1], key[2], Int32(H), Int32(L), nk)
    mul!(plan.real, plan.backward, ak)
    @cuda threads=1024 blocks=rb etd_cube_kernel!(plan.real, invn, nr)
    mul!(plan.cubek, plan.forward, plan.real)
    @cuda threads=256 blocks=kb etdrk2_correct_kernel!(
        uk, ak, cube1, plan.cubek, dt, two_kappa, m02, c3, inv_gamma,
        Int32(H), Int32(L), nk)
    return nothing
end

function etdrk2_evolve!(uk, plan, steps, key, dt; γ=1.0f0, T=1.0f0, κ=0.28f0,
                        λ=0.022f0, step0=0)
    ak = similar(uk); cube1 = similar(uk)
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    inv_gamma = 1.0f0 / γ
    noise_power = 2.0f0 * T / γ
    for s in 1:steps
        etdrk2_step!(uk, plan, ak, cube1, step0 + s, key, dt,
                     two_kappa, m02, c3, inv_gamma, noise_power)
    end
    return uk
end

function etd_evolve!(uk, plan, steps, key, dt; γ=1.0f0, T=1.0f0, κ=0.28f0, λ=0.022f0,
                     step0=0)
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    inv_gamma = 1.0f0 / γ
    noise_power = 2.0f0 * T / γ
    for s in 1:steps
        etd_step!(uk, plan, step0 + s, key, dt,
                  two_kappa, m02, c3, inv_gamma, noise_power)
    end
    return uk
end
