using CUDA, LinearAlgebra, Statistics, Test

include(joinpath(@__DIR__, "fused_philox.jl"))

function standardized_moments(x)
    μ = mean(x)
    σ = std(x)
    z = (x .- μ) ./ σ
    return μ, σ^2, mean(z .^ 3), mean(z .^ 4) - 3
end

function main()
    seed = UInt64(20260726)
    step = UInt64(17)
    nsamples = 1_048_576
    samples = CUDA.zeros(Float32, 4, nsamples)
    threads = 256
    blocks = min(cld(nsamples, threads), 264)
    @cuda threads=threads blocks=blocks fill_link_normals_kernel!(
        samples, seed, step, Int32(nsamples),
    )
    host_samples = Array(samples)

    for component in axes(host_samples, 1)
        μ, variance, skewness, excess_kurtosis =
            standardized_moments(@view host_samples[component, :])
        @test abs(μ) < 5.0e-3
        @test abs(variance - 1.0) < 1.0e-2
        @test abs(skewness) < 1.5e-2
        @test abs(excess_kurtosis) < 3.0e-2
    end
    @test maximum(
        abs,
        cor(permutedims(host_samples)) - I,
    ) < 5.0e-3

    repeat_samples = similar(samples)
    @cuda threads=threads blocks=blocks fill_link_normals_kernel!(
        repeat_samples, seed, step, Int32(nsamples),
    )
    @test Array(repeat_samples) == host_samples
    @cuda threads=threads blocks=blocks fill_link_normals_kernel!(
        repeat_samples, seed, step + 1, Int32(nsamples),
    )
    @test Array(repeat_samples) != host_samples

    L, Nensemble = 16, 32768
    n = L * L * Nensemble
    divergence = CUDA.zeros(Float32, L, L, Nensemble, 2)
    blocks = min(cld(n, threads), 264)
    @cuda threads=threads blocks=blocks fill_divergences_kernel!(
        divergence, seed, step, Val(L), Int32(L * L), Int32(n),
    )
    host_divergence = Array(divergence)
    for component in 1:2
        center = Float64.(@view host_divergence[1, 1, :, component])
        right = Float64.(@view host_divergence[2, 1, :, component])
        up = Float64.(@view host_divergence[1, 2, :, component])
        diagonal = Float64.(@view host_divergence[2, 2, :, component])
        @test abs(mean(center)) < 4.0e-2
        @test abs(var(center) - 4.0) < 8.0e-2
        @test abs(cov(center, right) + 1.0) < 5.0e-2
        @test abs(cov(center, up) + 1.0) < 5.0e-2
        @test abs(cov(center, diagonal)) < 5.0e-2
        sums = vec(sum(
            Float64,
            @view(host_divergence[:, :, :, component]);
            dims=(1, 2),
        ))
        @test maximum(abs, sums) < 2.0e-4
    end

    L, Nensemble = 32, 512
    CUDA.seed!(20260726)
    u0 = CUDA.randn(Float32, L, L, Nensemble)
    u = copy(u0)
    buffers = fused_philox_buffers(u; seed)
    dt = 0.002f0
    two_kappa = 0.56f0
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * 0.022f0) - 8.0f0 * 0.28f0
    c3 = 0.088f0
    fused_philox_step!(
        u, buffers, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    copyto!(u, u0)
    seed_fused_philox!(buffers, seed)
    fused_philox_steps!(
        u, buffers, 1000, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    M_drift = maximum(
        abs,
        mean(u; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    @test M_drift < 2.0e-6

    result = Array(u)
    copyto!(u, u0)
    seed_fused_philox!(buffers, seed)
    fused_philox_steps!(
        u, buffers, 1000, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    repeated = Array(u)
    @test repeated == result

    tiled_u = copy(u0)
    tiled_buffers = fused_philox_buffers(tiled_u; seed)
    tiled_philox_step!(
        tiled_u, tiled_buffers, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    copyto!(tiled_u, u0)
    seed_fused_philox!(tiled_buffers, seed)
    tiled_philox_steps!(
        tiled_u, tiled_buffers, 1000, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    @test Array(tiled_u) == result

    standard_u = copy(u0)
    standard_buffers = modelb_fixed_buffers(standard_u; seed)
    modelb_fixed_step!(
        standard_u, standard_buffers, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    copyto!(standard_u, u0)
    seed_modelb_fixed!(standard_buffers, seed)
    for _ in 1:1000
        modelb_fixed_step!(
            standard_u, standard_buffers, dt,
            two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
        )
    end

    overlapped_u = copy(u0)
    overlapped_buffers =
        overlapped_curand_buffers(overlapped_u; seed)
    overlapped_curand_steps!(
        overlapped_u, overlapped_buffers, 1, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    copyto!(overlapped_u, u0)
    seed_overlapped_curand!(overlapped_buffers, seed)
    overlapped_curand_steps!(
        overlapped_u, overlapped_buffers, 1000, dt,
        two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
    )
    CUDA.synchronize()
    @test Array(overlapped_u) == Array(standard_u)

    gpuarrays_u = copy(u0)
    gpuarrays_workspace = gpuarrays_buffers(gpuarrays_u; seed)
    for _ in 1:1000
        modelb_fixed_step!(
            gpuarrays_u, gpuarrays_workspace, dt,
            two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
        )
    end
    CUDA.synchronize()
    gpuarrays_result = Array(gpuarrays_u)
    gpuarrays_M_drift = maximum(
        abs,
        mean(gpuarrays_u; dims=(1, 2)) - mean(u0; dims=(1, 2)),
    )
    @test gpuarrays_M_drift < 2.0e-6
    copyto!(gpuarrays_u, u0)
    seed_gpuarrays!(gpuarrays_workspace, seed)
    for _ in 1:1000
        modelb_fixed_step!(
            gpuarrays_u, gpuarrays_workspace, dt,
            two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
        )
    end
    CUDA.synchronize()
    @test Array(gpuarrays_u) == gpuarrays_result

    direct_u = copy(u0)
    direct_buffers = modelb_fixed_buffers(direct_u; seed)
    for _ in 1:1000
        direct_curand_step!(
            direct_u, direct_buffers, dt,
            two_kappa, m02, c3, 1.0f0, sqrt(2.0f0),
        )
    end
    CUDA.synchronize()
    @test Array(direct_u) == Array(standard_u)

    nnormal = 12_288
    wrapped = CUDA.zeros(Float32, nnormal)
    direct = similar(wrapped)
    padded = CUDA.zeros(Float32, nextpow(2, nnormal))
    wrapped_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    direct_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    padded_rng = CUDA.cuRAND.LibraryRNG(
        CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    )
    Random.seed!(wrapped_rng, seed)
    Random.seed!(direct_rng, seed)
    Random.seed!(padded_rng, seed)
    Random.randn!(wrapped_rng, wrapped)
    direct_curand_fill!(direct_rng, direct)
    Random.randn!(padded_rng, padded)
    CUDA.synchronize()
    @test Array(direct) == Array(wrapped)
    @test Array(@view padded[1:nnormal]) == Array(wrapped)
    Random.randn!(wrapped_rng, wrapped)
    direct_curand_fill!(direct_rng, direct)
    Random.randn!(padded_rng, padded)
    CUDA.synchronize()
    @test Array(direct) != Array(wrapped)
    @test Array(@view padded[1:nnormal]) == Array(wrapped)

    direct_repeat = similar(direct)
    Random.seed!(direct_rng, seed)
    direct_curand_fill!(direct_rng, direct)
    Random.seed!(direct_rng, seed)
    direct_curand_fill!(direct_rng, direct_repeat)
    CUDA.synchronize()
    @test Array(direct) == Array(direct_repeat)
end

main()
