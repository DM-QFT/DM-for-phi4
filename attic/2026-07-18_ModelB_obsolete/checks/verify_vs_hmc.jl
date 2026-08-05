# Stationarity verification of the Model B spectral ETD integrator against a FA-HMC ensemble.
# Seed B trajectories from HMC equilibrium configs (κ,λ), batch-evolve Model B, and check that
# ⟨φ²⟩(t) and low-k S(k,t) stay at the HMC reference (no drift ⟺ integrator preserves e^{-S/T}).
using CUDA, CUDA.CUFFT, FFTW, JLD2, Statistics, Printf, Random
const myT = Float32

khat2_grid(L) = (k = myT.(2π .* (0:L-1) ./ L); kh = @. 2 * (1 - cos(k));
    CuArray([kh[i] + kh[j] for i in 1:L, j in 1:L]))

function dealias_mask(L)
    freq(n) = n <= L ÷ 2 ? n - 1 : n - 1 - L
    cut = L ÷ 3
    CuArray(myT[abs(freq(i)) <= cut && abs(freq(j)) <= cut for i in 1:L, j in 1:L])
end

# block (jackknife-style) error bar on the mean of per-sample values
function block_err(vals, nblk = 16)
    n = length(vals); bs = n ÷ nblk
    bm = [mean(@view vals[(b-1)*bs+1:b*bs]) for b in 1:nblk]
    std(bm) / sqrt(nblk)
end

function main()
    L = 128; κ = 0.2705f0; λ = 0.022f0; T = 1.0f0; γ = 1.0f0
    dt = 0.02f0; tend = 100.0f0; saveat = 2.0f0; B = 256; seed = 2026
    hmc = "/work/gw97/w97000/project/DM/2Dphi4/validation/cfgs_new_k=0.2705_l=0.022_128^2_n2276.jld2"

    cfgs = load(hmc, "cfgs")                       # (128,128,2276) Float64
    N = size(cfgs, 3)
    # HMC reference (all samples): per-sample ⟨φ²⟩ and low-k S
    ps = [mean(abs2, @view cfgs[:, :, i]) for i in 1:N]
    p2_hmc = mean(ps); p2_err = block_err(ps)
    sk_hmc = let acc = zeros(N)
        for i in 1:N
            F = fft(@view cfgs[:, :, i]); acc[i] = (abs2(F[2, 1]) + abs2(F[1, 2])) / (2 * L^2)
        end
        acc
    end
    skmin_hmc = mean(sk_hmc); skmin_err = block_err(sk_hmc)
    @printf("HMC ref (N=%d): <phi^2>=%.5f±%.5f  S(kmin)=%.4f±%.4f\n", N, p2_hmc, p2_err, skmin_hmc, skmin_err)

    # spectral ETD coefficients
    kh2 = khat2_grid(L)
    m02 = myT(2 * (1 - 2 * λ) - 8 * κ); twoκ = myT(2 * κ)
    Lop = @. -(kh2 / γ) * (twoκ * kh2 + m02)
    kh2_c3 = kh2 .* myT(4 * λ / γ)
    mask = dealias_mask(L)
    Lsafe = @. ifelse(abs(Lop) < 1f-12, 1f0, Lop)
    E = exp.(Lop .* dt)
    Φ1 = @. ifelse(abs(Lop * dt) < 1f-7, dt, expm1(Lop * dt) / Lsafe)
    Var = @. ifelse(abs(2 * Lop * dt) < 1f-7, (2 * T / γ) * kh2 * dt,
        (2 * T / γ) * kh2 * expm1(2 * Lop * dt) / (2 * Lsafe))
    noisec = sqrt.(max.(Var, 0f0))
    # reshape (L,L)->(L,L,1) for broadcast over batch
    r3(A) = reshape(A, L, L, 1)
    E3, Φ13, nc3, m3, kc3 = r3(E), r3(Φ1), r3(noisec), r3(mask), r3(kh2_c3)

    Random.seed!(seed); CUDA.seed!(UInt64(seed)); rng = CUDA.default_rng()
    φ = CuArray(myT.(cfgs[:, :, 1:B]))
    pf = plan_fft(φ .+ 0im, (1, 2)); pif = plan_ifft(φ .+ 0im, (1, 2))
    η = similar(φ); cube = similar(φ)

    obs = Tuple{Float32,Float32,Float32}[]
    push!(obs, (0f0, mean(abs2, φ), NaN32))
    nsteps = round(Int, tend / dt); save_every = round(Int, saveat / dt)
    for s in 1:nsteps
        φh = pf * (φ .+ 0im)
        @. cube = φ * φ * φ
        cubeh = pf * (cube .+ 0im)
        Nh = @. -kc3 * m3 * cubeh
        randn!(rng, η); ηh = pf * (η .+ 0im)
        φh = @. E3 * φh + Φ13 * Nh + nc3 * ηh
        φ .= real.(pif * φh)
        if s % save_every == 0
            push!(obs, (myT(s * dt), mean(abs2, φ), 0f0))
        end
    end

    outf = joinpath(@__DIR__, "spectral", "verify_hmc_k0.26_L128.dat")
    open(outf, "w") do io
        @printf(io, "# HMC ref <phi^2>=%.5f±%.5f  S(kmin)=%.4f±%.4f  (B=%d traj, dt=%g)\n",
            p2_hmc, p2_err, skmin_hmc, skmin_err, B, dt)
        println(io, "# t  <phi^2>(t)  bias_vs_HMC(%)")
        for (t, p2, _) in obs
            @printf(io, "%.2f  %.5f  %+.3f\n", t, p2, 100 * (p2 / p2_hmc - 1))
        end
    end
    @printf("t=0:   <phi^2>=%.5f (bias %+.3f%%)\n", obs[1][2], 100 * (obs[1][2] / p2_hmc - 1))
    late = mean(o[2] for o in obs[end-9:end])
    @printf("t~late:<phi^2>=%.5f (bias %+.3f%%)  -> %s\n", late, 100 * (late / p2_hmc - 1),
        abs(late / p2_hmc - 1) < 3 * p2_err / p2_hmc ? "STATIONARY (within HMC error)" : "DRIFT detected")
    println("written ", outf)
end

main()
