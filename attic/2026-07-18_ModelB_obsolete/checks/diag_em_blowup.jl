# Diagnose EM dt=0.005 blow-up: rare noise-triggered runaway of explicit EM on the
# superlinear (φ³) drift? Run several seeds; track per-config final ⟨φ²⟩ and max|φ|.
using CUDA, JLD2, Statistics, Printf, Random
const myT = Float32
limitbound(a, n) = a == n + 1 ? 1 : (a == 0 ? n : a)
limitbound2(a, n) = a == n + 1 ? 1 : (a == 0 ? n : (a == n + 2 ? 2 : (a == -1 ? n - 1 : a)))
@inline Ufn(φ, m02, c3) = m02 * φ + c3 * φ * φ * φ

function drift_kernel!(dσ, σ, m02, c3, twoκ, invγ)
    N = size(σ, 1); id = (blockIdx().x - 1) * blockDim().x + threadIdx().x; cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y, b = Tuple(cind[i])
        xp1, xm1 = limitbound2(x + 1, N), limitbound2(x - 1, N); yp1, ym1 = limitbound2(y + 1, N), limitbound2(y - 1, N)
        xp2, xm2 = limitbound2(x + 2, N), limitbound2(x - 2, N); yp2, ym2 = limitbound2(y + 2, N), limitbound2(y - 2, N)
        @inbounds bih = -20 * σ[x, y, b] + 8 * σ[x, ym1, b] - σ[x, ym2, b] + 8 * σ[x, yp1, b] - σ[x, yp2, b] +
                        8 * σ[xm1, y, b] - 2 * σ[xm1, ym1, b] - 2 * σ[xm1, yp1, b] - σ[xm2, y, b] -
                        2 * (-4 * σ[xp1, y, b] + σ[xp1, ym1, b] + σ[xp1, yp1, b]) - σ[xp2, y, b]
        @inbounds lapU = 4 * Ufn(σ[x, y, b], m02, c3) - Ufn(σ[xm1, y, b], m02, c3) - Ufn(σ[xp1, y, b], m02, c3) -
                         Ufn(σ[x, yp1, b], m02, c3) - Ufn(σ[x, ym1, b], m02, c3)
        @inbounds dσ[x, y, b] = invγ * (bih * twoκ - lapU)
    end
    return nothing
end
function cnoise_kernel!(dσ, ξx, ξy)
    N = size(dσ, 1); id = (blockIdx().x - 1) * blockDim().x + threadIdx().x; cind = CartesianIndices(dσ)
    for i = id:blockDim().x*gridDim().x:prod(size(dσ))
        x, y, b = Tuple(cind[i]); xm1 = limitbound(x - 1, N); ym1 = limitbound(y - 1, N)
        @inbounds dσ[x, y, b] = (ξx[x, y, b] - ξx[xm1, y, b]) + (ξy[x, y, b] - ξy[x, ym1, b])
    end
    return nothing
end
kdrift(a, b, m, c, t, g) = @cuda blocks=64 threads=256 drift_kernel!(a, b, m, c, t, g)
kcn(a, b, c) = @cuda blocks=64 threads=256 cnoise_kernel!(a, b, c)

function main()
    L = 128; κ = 0.2705f0; λ = 0.022f0; T = 1.0f0; invγ = 1.0f0
    m02 = myT(2 * (1 - 2λ) - 8κ); c3 = myT(4λ); twoκ = myT(2κ)
    cfgs = load("/work/gw97/w97000/project/DM/2Dphi4/validation/cfgs_new_k=0.2705_l=0.022_128^2_n2276.jld2", "cfgs")
    Nacc = 256; tend = 40.0f0
    φ0 = CuArray(myT.(cfgs[:, :, 1:Nacc])); p2ref = mean(abs2, cfgs)
    gsq = sqrt(2 * T * invγ)
    for dt in (0.005f0, 0.01f0, 0.02f0)
        Random.seed!(11); CUDA.seed!(11); rng = CUDA.default_rng()
        sqdt = sqrt(dt); nsteps = round(Int, tend / dt)
        φ = copy(φ0); dσ = similar(φ); dn = similar(φ); ξx = similar(φ); ξy = similar(φ)
        for _ in 1:nsteps
            kdrift(dσ, φ, m02, c3, twoκ, invγ); randn!(rng, ξx); randn!(rng, ξy); kcn(dn, ξx, ξy)
            @. φ = φ + dσ * dt + gsq * sqdt * dn
        end
        pc = [mean(abs2, @view φ[:, :, b]) for b in 1:Nacc]
        nbad = count(x -> x > 10 || isnan(x), pc)
        @printf("EM dt=%.4f: bias=%+.3f%%  n_blown(>10)=%d  max_cfg<phi2>=%.4g\n",
            dt, 100 * (mean(pc) / p2ref - 1), nbad, maximum(pc)); flush(stdout)
    end
end
main()
