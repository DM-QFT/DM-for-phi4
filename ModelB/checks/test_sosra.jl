using Test
using CUDA
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "ModelB.jl"))

@testset "fixed SOSRA Model B" begin
    CUDA.seed!(20260718)
    u0 = CUDA.randn(Float32, 16, 16, 4)
    u0 .-= mean(u0; dims=(1, 2))
    initial = Array(u0)

    laplacian(field) =
        circshift(field, (1, 0, 0)) + circshift(field, (-1, 0, 0)) +
        circshift(field, (0, 1, 0)) + circshift(field, (0, -1, 0)) - 4f0 * field
    two_kappa = 2f0 * 0.2705f0
    m02 = 2f0 * (1f0 - 2f0 * 0.022f0) - 8f0 * 0.2705f0
    c3 = 4f0 * 0.022f0
    reference_drift = laplacian(-two_kappa * laplacian(initial) + m02 * initial + c3 * initial.^3)
    gpu_drift = similar(u0)
    launch_modelb_drift!(gpu_drift, u0, two_kappa, m02, c3, 1f0)
    CUDA.synchronize()
    @test Array(gpu_drift) ≈ reference_drift rtol=5f-6 atol=2f-5

    xi_x = CUDA.randn(Float32, size(u0))
    xi_y = CUDA.randn(Float32, size(u0))
    gpu_noise = similar(u0)
    launch_conserved_noise!(gpu_noise, xi_x, xi_y)
    CUDA.synchronize()
    noise = Array(gpu_noise)
    xi_x_cpu, xi_y_cpu = Array(xi_x), Array(xi_y)
    reference_noise =
        xi_x_cpu - circshift(xi_x_cpu, (1, 0, 0)) +
        xi_y_cpu - circshift(xi_y_cpu, (0, 1, 0))
    @test noise ≈ reference_noise rtol=2f-6 atol=2f-6
    @test maximum(abs.(vec(sum(noise; dims=(1, 2))))) < 1f-4

    solution = modelB_2d_SDE_prob(
        u0;
        tspan=(0.0f0, 0.02f0),
        saveat=0.02f0,
    )

    final = Array(solution.u[end])
    @test string(solution.retcode) == "Success"
    @test solution.alg isa SOSRA
    @test solution.stats.naccept == 4
    @test solution.stats.nreject == 0
    @test size(final) == (16, 16, 4)
    @test all(isfinite, final)
    initial_means = vec(mean(initial; dims=(1, 2)))
    final_means = vec(mean(final; dims=(1, 2)))
    @test maximum(abs.(final_means - initial_means)) < 1f-5
end
