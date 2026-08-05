# Tightened accuracy comparison at κ=0.2705: batched over N_acc HMC configs so the bias
# statistical error drops from ±0.77% (N=24) to ~±0.17% (N=256). Batches all three methods:
# hand-written EM and spectral ETD (both batched kernels), SOSRA via StochasticDiffEq(3D u0).
# Bias = ensemble ⟨φ²⟩ drift from the HMC equilibrium seed. Action = 2Dphi4, dx=1.
using CUDA, CUDA.CUFFT, FFTW, StochasticDiffEq, DiffEqNoiseProcess, JLD2, Statistics, Printf, Random
const myT = Float32

limitbound(a, n) = a == n + 1 ? 1 : (a == 0 ? n : a)
limitbound2(a, n) = a == n + 1 ? 1 : (a == 0 ? n : (a == n + 2 ? 2 : (a == -1 ? n - 1 : a)))
@inline Ufn(φ, m02, c3) = m02 * φ + c3 * φ * φ * φ

# batched drift (L,L,B), periodic in dims 1,2, independent in dim 3
function drift_kernel!(dσ, σ, m02, c3, twoκ, invγ)
    N = size(σ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y, b = Tuple(cind[i])
        xp1, xm1 = limitbound2(x + 1, N), limitbound2(x - 1, N)
        yp1, ym1 = limitbound2(y + 1, N), limitbound2(y - 1, N)
        xp2, xm2 = limitbound2(x + 2, N), limitbound2(x - 2, N)
        yp2, ym2 = limitbound2(y + 2, N), limitbound2(y - 2, N)
        @inbounds bih = -20 * σ[x, y, b] + 8 * σ[x, ym1, b] - σ[x, ym2, b] + 8 * σ[x, yp1, b] -
                        σ[x, yp2, b] + 8 * σ[xm1, y, b] - 2 * σ[xm1, ym1, b] - 2 * σ[xm1, yp1, b] -
                        σ[xm2, y, b] - 2 * (-4 * σ[xp1, y, b] + σ[xp1, ym1, b] + σ[xp1, yp1, b]) - σ[xp2, y, b]
        @inbounds lapU = 4 * Ufn(σ[x, y, b], m02, c3) - Ufn(σ[xm1, y, b], m02, c3) -
                         Ufn(σ[xp1, y, b], m02, c3) - Ufn(σ[x, yp1, b], m02, c3) - Ufn(σ[x, ym1, b], m02, c3)
        @inbounds dσ[x, y, b] = invγ * (bih * twoκ - lapU)
    end
    return nothing
end
function cnoise_kernel!(dσ, ξx, ξy)
    N = size(dσ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(dσ)
    for i = id:blockDim().x*gridDim().x:prod(size(dσ))
        x, y, b = Tuple(cind[i])
        xm1 = limitbound(x - 1, N); ym1 = limitbound(y - 1, N)
        @inbounds dσ[x, y, b] = (ξx[x, y, b] - ξx[xm1, y, b]) + (ξy[x, y, b] - ξy[x, ym1, b])
    end
    return nothing
end
kdrift(dσ, σ, m02, c3, twoκ, invγ) = @cuda blocks=64 threads=256 drift_kernel!(dσ, σ, m02, c3, twoκ, invγ)
kcnoise(dσ, ξx, ξy) = @cuda blocks=64 threads=256 cnoise_kernel!(dσ, ξx, ξy)

# hand-written batched EM: φ += drift*dt + sqrt(2T/γ)*sqrt(dt)*div(ξ)
function em_batch(φ0, dt, tend, m02, c3, twoκ, invγ, T, rng)
    φ = copy(φ0); dσ = similar(φ); dn = similar(φ); ξx = similar(φ); ξy = similar(φ)
    gsq = sqrt(2 * T * invγ); sqdt = sqrt(dt)
    for _ in 1:round(Int, tend / dt)
        kdrift(dσ, φ, m02, c3, twoκ, invγ)
        randn!(rng, ξx); randn!(rng, ξy)
        kcnoise(dn, ξx, ξy)
        @. φ = φ + dσ * dt + gsq * sqdt * dn
    end
    return φ
end

# batched spectral ETD
function etd_batch(φ0, dt, tend, κ, λ, T, γ, rng; dealias=true)
    L = size(φ0, 1)
    k = myT.(2π .* (0:L-1) ./ L); kh1 = @. 2 * (1 - cos(k))
    kh2 = CuArray([kh1[i] + kh1[j] for i in 1:L, j in 1:L])
    m02 = myT(2 * (1 - 2λ) - 8κ); twoκ = myT(2κ)
    Lop = @. -(kh2 / γ) * (twoκ * kh2 + m02); Lsafe = @. ifelse(abs(Lop) < 1f-12, 1f0, Lop)
    E = reshape(exp.(Lop .* dt), L, L, 1)
    Φ1 = reshape((@. ifelse(abs(Lop * dt) < 1f-7, dt, expm1(Lop * dt) / Lsafe)), L, L, 1)
    Var = @. ifelse(abs(2 * Lop * dt) < 1f-7, (2T / γ) * kh2 * dt, (2T / γ) * kh2 * expm1(2 * Lop * dt) / (2 * Lsafe))
    nc = reshape(sqrt.(max.(Var, 0f0)), L, L, 1)
    kc = reshape(kh2 .* myT(4λ / γ), L, L, 1)
    freq(n) = n <= L ÷ 2 ? n - 1 : n - 1 - L; cut = L ÷ 3
    mask = reshape(CuArray(myT[abs(freq(i)) <= cut && abs(freq(j)) <= cut for i in 1:L, j in 1:L]), L, L, 1)
    dealias || (mask = reshape(CUDA.ones(myT, L, L), L, L, 1))
    φ = copy(φ0); η = similar(φ0); cube = similar(φ0)   # keep φ real, project each step
    pf = plan_fft(φ .+ 0im, (1, 2)); pif = plan_ifft(φ .+ 0im, (1, 2))
    for _ in 1:round(Int, tend / dt)
        φh = pf * (φ .+ 0im)
        @. cube = φ * φ * φ; cubeh = pf * (cube .+ 0im)
        Nh = @. -kc * mask * cubeh
        randn!(rng, η); ηh = pf * (η .+ 0im)
        φh = @. E * φh + Φ1 * Nh + nc * ηh
        φ .= real.(pif * φh)
    end
    return φ
end

# SOSRA via StochasticDiffEq with 3D u0
function NOISE_DIST_B(rand_vec, W, dt, u, p, t, rng)
    sq = @fastmath sqrt(abs(dt)); p1, ξx, ξy = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξx)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξy)
    CUDA.@sync kcnoise(p1, ξx, ξy)
    CUDA.@sync @. rand_vec = sq * p1
end
function NOISE_BRIDGE_B(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    p1, ξx, ξy = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξx)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξy)
    CUDA.@sync kcnoise(p1, ξx, ξy)
    CUDA.@sync @. rand_vec = sqrt((1 - q) * q * abs(h)) * p1 + q * Wh
end
function sosra_batch(φ0, dt, tend, m02, c3, twoκ, invγ, T)
    drift(dσ, σ, p, t) = (CUDA.@sync kdrift(dσ, σ, m02, c3, twoκ, invγ); nothing)
    g(du, u, p, t) = (du .= sqrt(2 * T * invγ))
    myp = (similar(φ0), similar(φ0), similar(φ0))
    W = NoiseProcess{true}(0.0f0, zero(φ0), zero(φ0), NOISE_DIST_B, NOISE_BRIDGE_B; save_everystep=false)
    prob = SDEProblem(drift, g, φ0, (0.0f0, myT(tend)), myp; noise=W)
    sol = solve(prob, SOSRA(); adaptive=false, dt=myT(dt), save_everystep=false, save_start=false, saveat=[myT(tend)])
    return sol.u[end]
end

function main()
    L = 128; κ = 0.2705f0; λ = 0.022f0; T = 1.0f0; γ = 1.0f0; invγ = 1.0f0
    m02 = myT(2 * (1 - 2λ) - 8κ); c3 = myT(4λ); twoκ = myT(2κ)
    Nacc = 256; t_acc = 40.0f0
    hmc = "/work/gw97/w97000/project/DM/2Dphi4/validation/cfgs_new_k=0.2705_l=0.022_128^2_n2276.jld2"
    cfgs = load(hmc, "cfgs"); p2ref = mean(abs2, cfgs)
    perr = std([mean(abs2, @view cfgs[:, :, i]) for i in 1:Nacc]) / sqrt(Nacc)
    @printf("HMC ref <phi^2>=%.5f  bias stat err ≈ ±%.3f%%  (Nacc=%d)\n", p2ref, 100 * perr / p2ref, Nacc); flush(stdout)
    φ0 = CuArray(myT.(cfgs[:, :, 1:Nacc]))
    rng = CUDA.default_rng()
    biasof(φ) = 100 * (mean(abs2, φ) / p2ref - 1)

    println("# method dt bias%"); flush(stdout)
    res = []
    for dt in (0.005f0, 0.01f0, 0.02f0)
        Random.seed!(11); CUDA.seed!(11)
        b = biasof(em_batch(φ0, dt, t_acc, m02, c3, twoκ, invγ, T, rng))
        @printf("EM %.4f %+.3f\n", dt, b); flush(stdout); push!(res, ("EM", dt, b))
    end
    for dt in (0.02f0, 0.05f0, 0.1f0)
        Random.seed!(11); CUDA.seed!(11)
        b = biasof(etd_batch(φ0, dt, t_acc, κ, λ, T, γ, rng))
        @printf("ETD %.4f %+.3f\n", dt, b); flush(stdout); push!(res, ("ETD", dt, b))
    end
    for dt in (0.01f0, 0.02f0, 0.05f0)
        Random.seed!(11); CUDA.seed!(11)
        b = biasof(sosra_batch(φ0, dt, t_acc, m02, c3, twoκ, invγ, T))
        @printf("SOSRA %.4f %+.3f\n", dt, b); flush(stdout); push!(res, ("SOSRA", dt, b))
    end
    open(joinpath(@__DIR__, "spectral", "bench_accuracy_k0.2705.dat"), "w") do io
        @printf(io, "# HMC ref=%.5f  Nacc=%d  bias_err=±%.3f%%  t_acc=%g  κ=%g λ=%g\n",
            p2ref, Nacc, 100 * perr / p2ref, t_acc, κ, λ)
        for (n, dt, b) in res
            @printf(io, "%s %.4f %+.3f\n", n, dt, b)
        end
    end
    println("written bench_accuracy_k0.2705.dat")
end
main()
