# Three-way accuracy-vs-speed benchmark: EM, spectral ETD, SOSRA, on the 2Dphi4 action.
# Accuracy: seed from HMC equilibrium (k=0.2705), evolve fixed physical time, measure the
#           ensemble ⟨φ²⟩ bias vs the HMC reference. Speed: single-field L=128, fixed-dt,
#           timed AFTER a JIT warm-up. All methods use dx=1, S=Σ[-2κΣφφ'+(1-2λ)φ²+λφ⁴].
using CUDA, CUDA.CUFFT, FFTW, StochasticDiffEq, DiffEqNoiseProcess, JLD2, Statistics, Printf, Random
const myT = Float32

# ---------------- real-space kernels (EM / SOSRA), copied from ModelB.jl ----------------
limitbound(a, n) = a == n + 1 ? 1 : (a == 0 ? n : a)
limitbound2(a, n) = a == n + 1 ? 1 : (a == 0 ? n : (a == n + 2 ? 2 : (a == -1 ? n - 1 : a)))

function update_2d_modelB!(dσ, σ, fun, dx, twoκ)
    N = size(σ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y = Tuple(cind[i])
        xp1, xm1 = limitbound2(x + 1, N), limitbound2(x - 1, N)
        yp1, ym1 = limitbound2(y + 1, N), limitbound2(y - 1, N)
        xp2, xm2 = limitbound2(x + 2, N), limitbound2(x - 2, N)
        yp2, ym2 = limitbound2(y + 2, N), limitbound2(y - 2, N)
        @inbounds dσ[x, y] =
            (-20 * σ[x, y] + 8 * σ[x, ym1] - σ[x, ym2] + 8 * σ[x, yp1] - σ[x, yp2] +
             8 * σ[xm1, y] - 2 * σ[xm1, ym1] - 2 * σ[xm1, yp1] - σ[xm2, y] -
             2 * (-4 * σ[xp1, y] + σ[xp1, ym1] + σ[xp1, yp1]) - σ[xp2, y]) * twoκ / dx^4 -
            (4 * fun(σ[x, y]) - fun(σ[xm1, y]) - fun(σ[xp1, y]) - fun(σ[x, yp1]) - fun(σ[x, ym1])) / dx^2
    end
    return nothing
end
function update_conserve_noise_kernel!(dσ, ξx, ξy, dx)
    N = size(dσ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(dσ)
    for i = id:blockDim().x*gridDim().x:prod(size(dσ))
        x, y = Tuple(cind[i])
        xm1 = limitbound(x - 1, N); ym1 = limitbound(y - 1, N)
        @inbounds dσ[x, y] = ((ξx[x, y] - ξx[xm1, y]) + (ξy[x, y] - ξy[x, ym1])) / dx
    end
    return nothing
end
modelB_loop(dσ, σ, fun, dx, twoκ) = @cuda blocks=32 threads=512 maxregs=4 update_2d_modelB!(dσ, σ, fun, dx, twoκ)
noise_loop(dσ, ξx, ξy, dx) = @cuda blocks=32 threads=512 maxregs=4 update_conserve_noise_kernel!(dσ, ξx, ξy, dx)

function NOISE_DIST(rand_vec, W, dt, u, p, t, rng)
    sq = @fastmath sqrt(abs(dt)); p1, ξx, ξy = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξx)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξy)
    CUDA.@sync noise_loop(p1, ξx, ξy, 1.0f0)
    CUDA.@sync @. rand_vec = sq * p1
end
function NOISE_BRIDGE(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    p1, ξx, ξy = p
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξx)
    DiffEqNoiseProcess.wiener_randn!(CUDA.default_rng(), ξy)
    CUDA.@sync noise_loop(p1, ξx, ξy, 1.0f0)
    CUDA.@sync @. rand_vec = sqrt((1 - q) * q * abs(h)) * p1 + q * Wh
end
RealProc!(t0, W0, Z0=nothing; kw...) = NoiseProcess{true}(t0, W0, Z0, NOISE_DIST, NOISE_BRIDGE; kw...)

# real-space solve from a given initial field u0 (host Float32); returns final ⟨φ²⟩ (or times a run)
function realspace_run(u0, solver, dt, tend; κ, λ, T=1.0f0, γ=1.0f0, with_Z=false, save=false)
    m02 = myT(2*(1-2λ) - 8κ); c3 = myT(4λ); twoκ = myT(2κ)
    Ufun(x) = m02 * x + c3 * x^3
    u0g = CuArray(u0)
    drift(dσ,σ,p,t) = (CUDA.@sync modelB_loop(dσ,σ,Ufun,1.0f0,twoκ); CUDA.@sync @. dσ = (1/γ)*dσ)
    g(du,u,p,t) = (du .= sqrt(2*T/γ))
    myp = (similar(u0g), similar(u0g), similar(u0g))
    W = RealProc!(0.0f0, zero(u0g), with_Z ? zero(u0g) : nothing, save_everystep=false)
    prob = SDEProblem(drift, g, u0g, (0.0f0, myT(tend)), myp; noise=W)
    sol = solve(prob, solver; adaptive=false, dt=myT(dt), save_everystep=false,
                save_start=false, saveat=save ? (0f0:5f0:tend) : [myT(tend)])
    return mean(abs2, Array(sol.u[end]))
end

# ---------------- spectral ETD, copied from ModelB_spectral.jl ----------------
khat2_grid(L) = (k=myT.(2π.*(0:L-1)./L); kh=@. 2*(1-cos(k)); CuArray([kh[i]+kh[j] for i in 1:L, j in 1:L]))
function dealias_mask(L)
    freq(n) = n <= L÷2 ? n-1 : n-1-L; cut = L÷3
    CuArray(myT[abs(freq(i))<=cut && abs(freq(j))<=cut for i in 1:L, j in 1:L])
end
function spectral_coeffs(L, κ, λ, T, γ, dt)
    kh2 = khat2_grid(L); m02 = myT(2*(1-2λ)-8κ); twoκ = myT(2κ)
    Lop = @. -(kh2/γ)*(twoκ*kh2 + m02); Lsafe = @. ifelse(abs(Lop)<1f-12, 1f0, Lop)
    E = exp.(Lop.*dt)
    Φ1 = @. ifelse(abs(Lop*dt)<1f-7, dt, expm1(Lop*dt)/Lsafe)
    Var = @. ifelse(abs(2*Lop*dt)<1f-7, (2T/γ)*kh2*dt, (2T/γ)*kh2*expm1(2*Lop*dt)/(2*Lsafe))
    return E, Φ1, sqrt.(max.(Var,0f0)), kh2.*myT(4λ/γ), dealias_mask(L)
end
# evolve a single field φ0 (CuArray) for nsteps; returns final ⟨φ²⟩
function spectral_run!(φ, coeffs, rng, nsteps)
    E, Φ1, noisec, kc, mask = coeffs
    pf = plan_fft(φ .+ 0im); pif = plan_ifft(φ .+ 0im)
    η = similar(φ); cube = similar(φ)
    for _ in 1:nsteps
        φh = pf*(φ .+ 0im); @. cube = φ*φ*φ; cubeh = pf*(cube .+ 0im)
        Nh = @. -kc*mask*cubeh
        randn!(rng, η); ηh = pf*(η .+ 0im)
        φh = @. E*φh + Φ1*Nh + noisec*ηh
        φ .= real.(pif*φh)
    end
    return mean(abs2, φ)
end

# ---------------- driver ----------------
function main()
    L=128; κ=0.2705f0; λ=0.022f0; T=1.0f0
    hmc = "/work/gw97/w97000/project/DM/2Dphi4/validation/cfgs_new_k=0.2705_l=0.022_128^2_n2276.jld2"
    cfgs = load(hmc, "cfgs"); p2ref = mean(abs2, cfgs)
    @printf("HMC ref <phi^2>=%.5f  (N=%d)\n", p2ref, size(cfgs,3)); flush(stdout)

    N_acc = 24; t_acc = 40.0f0            # accuracy: ensemble drift from HMC
    t_speed = 15.0f0; t_warm = 1.0f0      # speed: timed after warm-up
    rng = CUDA.default_rng()

    methods = [
        ("EM",    :em,   [0.005f0, 0.01f0, 0.02f0]),
        ("ETD",   :etd,  [0.02f0, 0.05f0, 0.1f0]),
        ("SOSRA", :sosra,[0.01f0, 0.02f0, 0.05f0]),
    ]
    println("\n# method  dt     bias%    s/t(speed)  steps/t"); flush(stdout)
    results = []
    for (name, kind, dts) in methods
        for dt in dts
            # --- accuracy: N_acc trajectories from HMC configs ---
            Random.seed!(4040); CUDA.seed!(4040)
            p2s = Float64[]
            for i in 1:N_acc
                u0 = myT.(cfgs[:, :, i])
                if kind == :etd
                    coeffs = spectral_coeffs(L, κ, λ, T, T, dt)
                    push!(p2s, spectral_run!(CuArray(u0), coeffs, rng, round(Int, t_acc/dt)))
                else
                    solver = kind == :em ? EM() : SOSRA()
                    push!(p2s, realspace_run(u0, solver, dt, t_acc; κ=κ, λ=λ, T=T, with_Z=(kind==:sosra)))
                end
            end
            biaspct = 100*(mean(p2s)/p2ref - 1)
            # --- speed: single field, warm-up then time ---
            u0 = myT.(cfgs[:, :, 1])
            if kind == :etd
                coeffs = spectral_coeffs(L, κ, λ, T, T, dt)
                spectral_run!(CuArray(u0), coeffs, rng, round(Int, t_warm/dt))   # warm-up
                CUDA.synchronize()
                el = @elapsed (spectral_run!(CuArray(u0), coeffs, rng, round(Int, t_speed/dt)); CUDA.synchronize())
            else
                solver = kind == :em ? EM() : SOSRA()
                realspace_run(u0, solver, dt, t_warm; κ=κ, λ=λ, T=T, with_Z=(kind==:sosra))  # warm-up
                CUDA.synchronize()
                el = @elapsed (realspace_run(u0, solver, dt, t_speed; κ=κ, λ=λ, T=T, with_Z=(kind==:sosra)); CUDA.synchronize())
            end
            spt = el/t_speed
            @printf("%-6s %.4f  %+6.2f   %8.4f   %.0f\n", name, dt, biaspct, spt, 1/dt); flush(stdout)
            push!(results, (name, dt, biaspct, spt))
        end
    end
    open(joinpath(@__DIR__, "spectral", "bench_three.dat"), "w") do io
        @printf(io, "# HMC ref <phi^2>=%.5f  L=%d κ=%g λ=%g T=%g  N_acc=%d t_acc=%g t_speed=%g\n",
                p2ref, L, κ, λ, T, N_acc, t_acc, t_speed)
        println(io, "# method  dt  bias%  s_per_t")
        for (n,dt,b,s) in results
            @printf(io, "%s  %.4f  %+.3f  %.5f\n", n, dt, b, s)
        end
    end
    println("written bench_three.dat")
end
main()
