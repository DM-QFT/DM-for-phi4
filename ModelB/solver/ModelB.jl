using CUDA, DiffEqNoiseProcess, Random, StochasticDiffEq

@inline limitbound(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end

@inline function limitbound2(a, n)
    if a == n + 1
        1
    elseif a == 0
        n
    elseif a == n + 2
        2
    elseif a == -1
        n - 1
    else
        a
    end
end

@inline cube(x) = x * x * x

function modelb_drift_kernel!(du, u, two_kappa, m02, c3, inv_gamma)
    n = size(u, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    indices = CartesianIndices(u)

    @inbounds for linear in id:stride:length(u)
        x, y, ensemble = Tuple(indices[linear])
        xp1, xm1 = limitbound2(x + 1, n), limitbound2(x - 1, n)
        yp1, ym1 = limitbound2(y + 1, n), limitbound2(y - 1, n)
        xp2, xm2 = limitbound2(x + 2, n), limitbound2(x - 2, n)
        yp2, ym2 = limitbound2(y + 2, n), limitbound2(y - 2, n)

        center = u[x, y, ensemble]
        left, right = u[xm1, y, ensemble], u[xp1, y, ensemble]
        down, up = u[x, ym1, ensemble], u[x, yp1, ensemble]

        nearest_sum = left + right + down + up
        diagonal_sum =
            u[xm1, ym1, ensemble] + u[xm1, yp1, ensemble] +
            u[xp1, ym1, ensemble] + u[xp1, yp1, ensemble]
        second_neighbor_sum =
            u[xm2, y, ensemble] + u[xp2, y, ensemble] +
            u[x, ym2, ensemble] + u[x, yp2, ensemble]

        # Action gradient: δS/δφ = -2κΔφ + m₀²φ + 4λφ³.
        # Its outer Laplacian is fused here to avoid an intermediate GPU array:
        # drift = Δ(δS/δφ)/γ = [2κ(-Δ²φ) + m₀²Δφ + 4λΔ(φ³)]/γ.
        minus_biharmonic =
            -20 * center + 8 * nearest_sum - 2 * diagonal_sum - second_neighbor_sum
        laplacian_phi = nearest_sum - 4 * center
        laplacian_phi3 =
            cube(left) + cube(right) + cube(down) + cube(up) - 4 * cube(center)

        du[x, y, ensemble] = inv_gamma * (
            two_kappa * minus_biharmonic +
            m02 * laplacian_phi +
            c3 * laplacian_phi3
        )
    end
    return nothing
end

function conserved_noise_kernel!(out, xi_x, xi_y)
    n = size(out, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    indices = CartesianIndices(out)

    @inbounds for linear in id:stride:length(out)
        x, y, ensemble = Tuple(indices[linear])
        xm1 = limitbound(x - 1, n)
        ym1 = limitbound(y - 1, n)
        out[x, y, ensemble] =
            xi_x[x, y, ensemble] - xi_x[xm1, y, ensemble] +
            xi_y[x, y, ensemble] - xi_y[x, ym1, ensemble]
    end
    return nothing
end

function launch_modelb_drift!(du, u, two_kappa, m02, c3, inv_gamma)
    # Tuned for the 132-SM GH200 at L=128 with 1024 ensembles.
    threads = 896
    blocks = min(cld(length(u), threads), 132)
    @cuda threads=threads blocks=blocks modelb_drift_kernel!(
        du, u, two_kappa, m02, c3, inv_gamma
    )
    return nothing
end

function launch_conserved_noise!(out, xi_x, xi_y)
    threads = 640
    blocks = min(cld(length(out), threads), 264)
    @cuda threads=threads blocks=blocks conserved_noise_kernel!(out, xi_x, xi_y)
    return nothing
end

function modelb_noise_dist!(rand_vec, W, dt, u, p, t, rng)
    noise_work, xi_x, xi_y, gpu_rng = p
    # DiffEqNoiseProcess's generic GPU fallback deliberately ignores `rng`.
    # The problem therefore owns an explicitly seeded Philox generator in p.
    Random.randn!(gpu_rng, xi_x)
    Random.randn!(gpu_rng, xi_y)
    CUDA.@sync launch_conserved_noise!(noise_work, xi_x, xi_y)
    CUDA.@sync @. rand_vec = sqrt(abs(dt)) * noise_work
    return nothing
end

function modelb_noise_bridge!(rand_vec, W, W0, Wh, q, h, u, p, t, rng)
    noise_work, xi_x, xi_y, gpu_rng = p
    Random.randn!(gpu_rng, xi_x)
    Random.randn!(gpu_rng, xi_y)
    CUDA.@sync launch_conserved_noise!(noise_work, xi_x, xi_y)
    CUDA.@sync @. rand_vec = sqrt((1 - q) * q * abs(h)) * noise_work + q * Wh
    return nothing
end

function modelb_noise_process(t0, template; seed=rand(UInt64))
    # SOSRA requires both the Wiener increment W and its auxiliary integral Z.
    # Keep a copyable CPU RNG for NoiseProcess's own bookkeeping.  Actual
    # CuArray draws use the problem-owned Philox generator in p.
    rng = Random.Xoshiro(UInt64(seed))
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        modelb_noise_dist!,
        modelb_noise_bridge!;
        save_everystep=false,
        rng,
    )
end

function modelB_2d_SDE_prob(
    u0;
    γ=1.0f0,
    tspan=(0.0f0, 15.0f0),
    T=1.0f0,
    κ=0.2705f0,
    λ=0.022f0,
    saveat=1.0f0,
    dt=0.005f0,
    seed=rand(UInt64),
)
    γ, T, κ, λ, dt = Float32.((γ, T, κ, λ, dt))
    tspan = Float32.(tspan)
    saveat = Float32(saveat)
    seed = UInt64(seed)
    u = CuArray(Float32.(u0))
    ndims(u) == 3 && size(u, 1) == size(u, 2) && size(u, 1) >= 4 &&
        size(u, 3) > 0 || throw(ArgumentError(
            "u0 must have shape (L, L, Nensemble), with L >= 4 and Nensemble > 0",
        ))
    all(isfinite, u) || throw(ArgumentError("u0 must be finite"))
    all(isfinite, (γ, T, κ, λ, dt, tspan..., saveat)) ||
        throw(ArgumentError("all physical/time parameters must be finite"))
    γ > 0.0f0 || throw(ArgumentError("γ must be positive"))
    T >= 0.0f0 || throw(ArgumentError("T must be nonnegative"))
    tspan[2] >= tspan[1] || throw(ArgumentError("tspan must run forward"))
    saveat > 0.0f0 || throw(ArgumentError("saveat must be positive"))

    inv_gamma = 1.0f0 / γ
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    gpu_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    Random.seed!(gpu_rng, seed)
    workspace = (similar(u), similar(u), similar(u), gpu_rng)

    function drift!(du, u, p, t)
        CUDA.@sync launch_modelb_drift!(du, u, two_kappa, m02, c3, inv_gamma)
    end

    noise_amplitude = sqrt(2.0f0 * T * inv_gamma)
    function diffusion!(du, u, p, t)
        fill!(du, noise_amplitude)
    end

    dt > 0.0f0 || throw(ArgumentError("dt must be positive"))
    noise = modelb_noise_process(tspan[1], u; seed)
    problem = SDEProblem(drift!, diffusion!, u, tspan, workspace; noise=noise)
    alg = SOSRA()
    solve(
        problem, alg;
        adaptive=false,
        dt,
        saveat,
        save_everystep=false,
        seed,
    )
end
