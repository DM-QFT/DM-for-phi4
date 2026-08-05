# Spectral stochastic exponential-Euler (ETD1) solver for 2D conserved Model B.
# Lattice action matches DM/2Dphi4 (WolffFAHMC/Langevin convention), P ∝ exp(-S/T), T=1:
#   S = Σ_x [ -2κ Σ_μ φφ' + (1-2λ)φ² + λφ⁴ ]
# In Fourier space each mode is an OU process integrated exactly (noise + stiff linear part);
# the φ³ nonlinearity is explicit. dx=1.
#
#   m₀²    = 2(1-2λ) - 8κ
#   L(k)   = -(k̂²/γ)(2κ k̂² + m₀²),       k̂² = 2(1-cos kx)+2(1-cos ky) ∈ [0,8]
#   N_k    = -(4λ k̂²/γ) * FFT(φ³)
#   Var_k  = (2T k̂²/γ) (e^{2L dt}-1)/(2L)   (→ (2T k̂²/γ)dt as L→0; =0 for k=0 ⟹ M conserved)
#   φ_k   ← e^{L dt} φ_k + φ1 N_k + sqrt(Var_k) * FFT(randn),   φ1 = (e^{L dt}-1)/L
using CUDA, CUDA.CUFFT, FFTW, Statistics, Printf, DelimitedFiles, Random
const myT = Float32

function khat2_grid(L)
    k = myT.(2π .* (0:L-1) ./ L)
    kh = @. 2 * (1 - cos(k))                       # 1D  2(1-cos k)
    return CuArray([kh[i] + kh[j] for i in 1:L, j in 1:L])
end

# 2/3-rule dealiasing mask for the cubic term (zero the top 1/3 of |frequency|)
function dealias_mask(L)
    freq(n) = n <= L ÷ 2 ? n - 1 : n - 1 - L
    cut = L ÷ 3
    return CuArray(myT[abs(freq(i)) <= cut && abs(freq(j)) <= cut for i in 1:L, j in 1:L])
end

# precompute the diagonal ETD coefficients for given linear operator Lop
function etd_coeffs(Lop, kh2, T, γ, dt)
    Ldt = Lop .* dt
    Lsafe = @. ifelse(abs(Lop) < 1f-12, 1f0, Lop)
    E = exp.(Ldt)
    Φ1 = @. ifelse(abs(Ldt) < 1f-7, dt, expm1(Ldt) / Lsafe)
    Var = @. ifelse(abs(2 * Ldt) < 1f-7,
        (2 * T / γ) * kh2 * dt,
        (2 * T / γ) * kh2 * expm1(2 * Ldt) / (2 * Lsafe))
    noisec = sqrt.(max.(Var, 0f0))
    return E, Φ1, noisec
end

# one integration run; returns (times, frames) with frames as host Float32 matrices
function run_spectral(L, T, tend, dt; γ = 1.0f0, κ = 0.27f0, λ = 0.022f0,
        dealias = true, seed = 1234, saveat = 1.0f0)
    Random.seed!(seed); CUDA.seed!(UInt64(seed))
    rng = CUDA.default_rng()
    kh2 = khat2_grid(L)
    m02 = myT(2 * (1 - 2 * λ) - 8 * κ)
    twoκ = myT(2 * κ)
    Lop = @. -(kh2 / γ) * (twoκ * kh2 + m02)
    kh2_c3 = kh2 .* myT(4 * λ / γ)
    mask = dealias ? dealias_mask(L) : CUDA.ones(myT, L, L)
    E, Φ1, noisec = etd_coeffs(Lop, kh2, T, γ, dt)

    φ = CUDA.randn(myT, L, L)
    φ .-= mean(φ)                                  # M = 0
    pf = plan_fft(φ .+ 0im); pif = plan_ifft(φ .+ 0im)

    nsteps = round(Int, tend / dt)
    save_every = max(1, round(Int, saveat / dt))
    times = myT[]; frames = Matrix{myT}[]
    push!(times, 0f0); push!(frames, Array(φ))

    cube = similar(φ); η = similar(φ)
    for s in 1:nsteps
        φh = pf * (φ .+ 0im)
        @. cube = φ * φ * φ
        cubeh = pf * (cube .+ 0im)
        Nh = @. -kh2_c3 * mask * cubeh
        randn!(rng, η)
        ηh = pf * (η .+ 0im)
        φh = @. E * φh + Φ1 * Nh + noisec * ηh
        φ .= real.(pif * φh)
        if s % save_every == 0
            push!(times, myT(s * dt)); push!(frames, Array(φ))
        end
    end
    return times, frames
end

# axis-averaged, time-averaged (2nd half) structure factor + late-time observables
function analyze(frames, L)
    nh = length(frames) ÷ 2
    Sacc = zeros(L ÷ 2)
    for fr in frames[nh+1:end]
        F = fft(Float64.(fr))
        for n in 1:L÷2
            Sacc[n] += (abs2(F[n+1, 1]) + abs2(F[1, n+1])) / (2 * L^2)
        end
    end
    Sacc ./= (length(frames) - nh)
    p2 = mean(mean(abs2, fr) for fr in frames[nh+1:end])
    drift = maximum(abs(mean(fr) - mean(frames[1])) for fr in frames)
    return Sacc, p2, drift
end

function main(args = ARGS)
    outprefix = args[1]
    L = parse(Int, args[2])
    tend = parse(myT, args[3])
    Ts = parse.(myT, split(args[4], ","))
    dt = length(args) >= 5 ? parse(myT, args[5]) : 0.05f0
    seed = length(args) >= 6 ? parse(Int, args[6]) : 1234
    κ = length(args) >= 7 ? parse(myT, args[7]) : 0.27f0
    λ = length(args) >= 8 ? parse(myT, args[8]) : 0.022f0
    dealias = length(args) >= 9 ? args[9] == "1" : true
    for T in Ts
        t_run = @elapsed (times, frames) = run_spectral(L, T, tend, dt;
            κ = κ, λ = λ, dealias = dealias, seed = seed)
        Sacc, p2, drift = analyze(frames, L)
        tag = @sprintf("%sT%.3f_L%d", outprefix, T, L)
        open(tag * "_obs.dat", "w") do io
            println(io, "# t  mean(u)  <phi^2>  S(kmin)")
            for (t, fr) in zip(times, frames)
                F = fft(Float64.(fr))
                Skmin = (abs2(F[2, 1]) + abs2(F[1, 2])) / (2 * L^2)
                @printf(io, "%.2f  %.6e  %.6f  %.6f\n", t, mean(fr), mean(abs2, fr), Skmin)
            end
        end
        writedlm(tag * "_final.dat", frames[end])
        open(tag * "_Sk.dat", "w") do io
            println(io, "# k  S(k)  (axis+time averaged, 2nd half)")
            for n in 1:L÷2
                @printf(io, "%.6f  %.6f\n", 2π * n / L, Sacc[n])
            end
        end
        @printf("spectral T=%.3f L=%d κ=%g λ=%g dt=%g dealias=%d: %.1fs, M-drift=%.2e, <phi^2>(late)=%.4f, S(kmin)=%.4f\n",
            T, L, κ, λ, dt, dealias, t_run, drift, p2, Sacc[1])
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
