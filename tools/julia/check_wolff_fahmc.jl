# 验证 WolffFAHMC_ND.jl 的傅立叶加速 HMC 是精确算法。
# 依次检查力与作用量一致、动量采样与动能归一化一致、蛙跳可逆且 ΔH ∝ ε²、
# 自由场 (λ=0) 的精确传播子、以及 φ⁴ L=8 与单点 Metropolis 的可观测量对比。
using Random, Statistics, LinearAlgebra, FFTW, Printf
include(joinpath(@__DIR__, "WolffFAHMC_ND.jl"))

const D = 2
rng = MersenneTwister(7)

# 分块 jackknife，f 把一段样本序列映射成一个标量
function jackknife(f, x, nb)
    blocks = reshape(x[1:(length(x) ÷ nb) * nb], :, nb)
    full = f(vec(blocks))
    jk = [f(vec(blocks[:, setdiff(1:nb, b)])) for b in 1:nb]
    return full, sqrt((nb - 1) * mean(abs2, jk .- mean(jk)))
end

absM(M) = mean(abs, M)
chiV(M) = mean(abs2, M) - mean(abs, M)^2
U4(M) = 1 - mean(x -> x^4, M) / (3 * mean(abs2, M)^2)

# ---- 1. force_at 与 get_action 的中心差分 ----
let N = 6, κ = 0.2705, λ = 0.022
    lat = Lattice{D}(N)
    φ = randn(rng, N, N)
    h = 1e-5
    err = 0.0
    for idx in CartesianIndices(φ)
        φp = copy(φ)
        φm = copy(φ)
        φp[idx] += h
        φm[idx] -= h
        fd = -(get_action(φp, κ, λ, lat) - get_action(φm, κ, λ, lat)) / (2h)
        err = max(err, abs(fd - force_at(φ, lat, idx, 2κ, 2λ)))
    end
    @printf("1. force vs finite difference   max|Δ| = %.1e\n", err)
    flush(stdout)
end

# ---- 2. 动量协方差 ⟨|π̃_k|²⟩ = V M_k，⟨K⟩ = V/2 ----
let N = 16, V = N^D, n = 20000
    fm = FourierMassMatrix{D}(N, 0.1)
    pf = zeros(N, N)
    P = zeros(size(fm.M_k))
    Ksum = 0.0
    for i in 1:n
        sample_momentum!(pf, rng, fm)
        Ksum += kinetic_energy(fm, pf)
        mul!(fm.cbuf, fm.rfft_plan, pf)
        P .+= abs2.(fm.cbuf)
    end
    ratio = P ./ (n * V .* fm.M_k)
    @printf("2. ⟨K⟩/(V/2) = %.4f    ⟨|π̃_k|²⟩/(V M_k) over %d modes: mean %.4f  min %.4f  max %.4f   (expect 1, per-mode noise ~%.3f)\n",
            Ksum / n / (V / 2), length(ratio), mean(ratio), minimum(ratio), maximum(ratio), sqrt(2 / n))
    flush(stdout)
end

# ---- 3. 蛙跳可逆性与 ΔH ∝ ε² ----
let N = 16, κ = 0.2705, λ = 0.022
    lat = Lattice{D}(N)
    fm = FourierMassMatrix{D}(N, 0.1)
    φ0 = randn(rng, N, N)
    p0 = zeros(N, N)
    sample_momentum!(p0, rng, fm)
    φ = copy(φ0)
    pf = copy(p0)
    fourier_leapfrog!(φ, pf, fm, 0.05, 40, κ, λ, lat)
    pf .*= -1
    fourier_leapfrog!(φ, pf, fm, 0.05, 40, κ, λ, lat)
    @printf("3a. reversibility   max|φ-φ0| = %.1e   max|π+π0| = %.1e\n",
            maximum(abs.(φ .- φ0)), maximum(abs.(pf .+ p0)))
    function H(φ, pf)
        return get_action(φ, κ, λ, lat) + kinetic_energy(fm, pf)
    end
    ΔH = Float64[]
    for (ε, n) in ((0.08, 25), (0.04, 50), (0.02, 100))
        φ = copy(φ0)
        pf = copy(p0)
        H0 = H(φ, pf)
        fourier_leapfrog!(φ, pf, fm, ε, n, κ, λ, lat)
        push!(ΔH, H(φ, pf) - H0)
    end
    @printf("3b. ΔH at ε = 0.08 / 0.04 / 0.02, same trajectory length: %.3e / %.3e / %.3e   ratios %.2f, %.2f (expect 4)\n",
            ΔH[1], ΔH[2], ΔH[3], ΔH[1] / ΔH[2], ΔH[2] / ΔH[3])
    flush(stdout)
end

# ---- 4. 自由场 λ=0，S = ½ φᵀAφ，A(p) = 2 − 4κ Σ_μ cos p_μ，G(p) = 1/A(p) ----
let N = 16, V = N^D, κ = 0.2, n = 20000
    cfgs, acc, ε, fm, clf, _ = wolff_fahmc(;
        D, N_lat=N, κ, λ=0.0, n_samples=n, n_warmup=1000,
        store_cfgs=true, seed=11,
    )
    k = 2π .* (0:N-1) ./ N
    A = [2 - 4κ * (cos(k[i]) + cos(k[j])) for i in 1:N, j in 1:N]
    G = 1 ./ A
    M = vec(mean(cfgs; dims=(1, 2)))
    φ2 = vec(mean(abs2, cfgs; dims=(1, 2)))
    P10 = zeros(n)
    P22 = zeros(n)
    Ppp = zeros(n)
    h = N ÷ 2 + 1
    for s in 1:n
        f = fft(cfgs[:, :, s])
        P10[s] = abs2(f[2, 1]) / V
        P22[s] = abs2(f[3, 3]) / V
        Ppp[s] = abs2(f[h, h]) / V
    end
    @printf("4. free field N=%d κ=%.1f, acc=%.2f, ε=%.3f, M(p)=Zp̂²+m² fitted (exact: Z=%.3f m²=%.3f)\n",
            N, κ, acc, ε, 2κ, 2 - 4D * κ)
    checks = (
        ("⟨φ²⟩", φ2, mean, mean(G)),
        ("⟨M²⟩", M, x -> mean(abs2, x), G[1, 1] / V),
        ("G(1,0)", P10, mean, G[2, 1]),
        ("G(2,2)", P22, mean, G[3, 3]),
        ("G(π,π)", Ppp, mean, G[h, h]),
        ("U4", M, U4, 0.0),
    )
    for (name, x, f, exact) in checks
        v, e = jackknife(f, x, 50)
        @printf("   %-7s %.5f(%.5f)   exact %.5f   diff %+.1fσ\n", name, v, e, exact, (v - exact) / e)
    end
    flush(stdout)
end

# ---- 5. φ⁴ L=8 κ=0.2705 λ=0.022，FA-HMC 对比单点 Metropolis ----
function metropolis_sweep!(φ, κ, λ, lat, rng, δ)
    @inbounds for idx in CartesianIndices(φ)
        ϕ = φ[idx]
        nn = nn_sum_all(φ, lat, idx)
        ϕn = ϕ + δ * (2rand(rng) - 1)
        ΔS = -2κ * nn * (ϕn - ϕ) + (1 - 2λ) * (ϕn^2 - ϕ^2) + λ * (ϕn^4 - ϕ^4)
        if ΔS < 0 || rand(rng) < exp(-ΔS)
            φ[idx] = ϕn
        end
    end
    return nothing
end

let N = 8, V = N^D, κ = 0.2705, λ = 0.022, nsw = 2_000_000, nh = 100_000
    lat = Lattice{D}(N)
    φ = randn(rng, N, N)
    for _ in 1:20_000
        metropolis_sweep!(φ, κ, λ, lat, rng, 1.5)
    end
    Mm = zeros(nsw)
    φ2m = zeros(nsw)
    for s in 1:nsw
        metropolis_sweep!(φ, κ, λ, lat, rng, 1.5)
        Mm[s] = mean(φ)
        φ2m[s] = mean(abs2, φ)
    end
    cfgs, acc, ε, fm, clf, _ = wolff_fahmc(;
        D, N_lat=N, κ, λ, n_samples=nh, n_warmup=2000,
        store_cfgs=true, seed=5,
    )
    Mh = vec(mean(cfgs; dims=(1, 2)))
    φ2h = vec(mean(abs2, cfgs; dims=(1, 2)))
    @printf("5. φ⁴ N=%d κ=%.4f λ=%.3f: Metropolis %d sweeps vs FA-HMC %d iterations (acc=%.2f, ε=%.3f)\n",
            N, κ, λ, nsw, nh, acc, ε)
    checks = (
        ("⟨|M|⟩", Mm, Mh, absM),
        ("χ/V", Mm, Mh, chiV),
        ("U4", Mm, Mh, U4),
        ("⟨φ²⟩", φ2m, φ2h, mean),
    )
    for (name, xm, xh, f) in checks
        vm, em = jackknife(f, xm, 100)
        vh, eh = jackknife(f, xh, 100)
        @printf("   %-6s Metropolis %.5f(%.5f)   FA-HMC %.5f(%.5f)   diff %+.1fσ\n",
                name, vm, em, vh, eh, (vh - vm) / hypot(em, eh))
    end
    flush(stdout)
end
