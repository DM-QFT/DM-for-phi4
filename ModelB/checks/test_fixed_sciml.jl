using CUDA, Statistics, Test

include(joinpath(@__DIR__, "..", "ModelB_fixed_sciml.jl"))
include(joinpath(@__DIR__, "..", "ModelB.jl"))

CUDA.seed!(20260718)
u0 = CUDA.randn(Float32, 128, 128, 4)
u0 .-= mean(u0; dims=(1, 2))
prob = modelB_2d_fixed_problem(u0; tspan=(0.0f0, 0.02f0))
seed = UInt64(0x12345678)

direct = copy(prob.u0)
direct_buffers = modelb_fixed_buffers(direct; seed)
modelB_2d_fixed!(direct, direct_buffers; tspan=prob.tspan, dt=0.002f0)
CUDA.synchronize()

sol = solve(
    prob, ModelBFixedSOSRA();
    dt=0.002f0, saveat=prob.tspan[2], seed,
)
reused_buffers = modelb_fixed_buffers(prob.u0; seed=UInt64(1))
reused_alg = ModelBFixedSOSRA(; buffers=reused_buffers)
sol_reused = solve(prob, reused_alg; dt=0.002f0, seed)
sol_repeated = solve(prob, reused_alg; dt=0.002f0, seed)

initial_mean = Array(mean(prob.u0; dims=(1, 2)))
final_mean = Array(mean(sol.u[end]; dims=(1, 2)))

@testset "ModelBFixedSOSRA SciML interface" begin
    @test MODEL_B_SCIML_BASE.isadaptive(ModelBFixedSOSRA()) == false
    @test MODEL_B_SCIML_BASE.alg_order(ModelBFixedSOSRA()) == 2 // 1
    @test MODEL_B_SCIML_BASE.requires_additive_noise(ModelBFixedSOSRA())
    @test MODEL_B_SCIML_BASE.alg_interpretation(ModelBFixedSOSRA()) ==
          MODEL_B_SCIML_BASE.AlgorithmInterpretation.Ito
    @test MODEL_B_SCIML_BASE.successful_retcode(sol)
    @test sol.t == Float32[0.02f0]
    @test sol.stats.nf == 30
    @test sol.stats.naccept == 10
    @test sol.seed == seed

    # The wrapper must preserve the exact fused backend trajectory.
    @test sol.u[end] == direct
    @test sol_reused.u[end] == direct
    @test sol_repeated.u[end] == sol_reused.u[end]
    @test maximum(abs.(final_mean .- initial_mean)) < 2.0f-6
    @test all(isfinite, Array(sol.u[end]))

    @test_throws AssertionError solve(prob, ModelBFixedSOSRA(); adaptive=true)
    @test_throws AssertionError solve(prob, ModelBFixedSOSRA(); dense=true)
    @test_throws AssertionError solve(
        prob, ModelBFixedSOSRA(); save_everystep=true,
    )
    @test_throws AssertionError solve(
        prob, ModelBFixedSOSRA(); saveat=0.01f0,
    )
    @test_throws MethodError solve(
        prob, ModelBFixedSOSRA(); callback=identity,
    )
end

@testset "fused noise and SOSRA stage 1" begin
    L = 32
    u_values = reshape(
        cos.(0.011f0 .* Float32.(1:(2 * L * L))),
        L, L, 2,
    )
    u = CuArray(u_values)
    buffers = modelb_fixed_buffers(u; seed=UInt64(1))
    values = reshape(
        sin.(0.017f0 .* Float32.(1:length(buffers.links))),
        size(buffers.links),
    )
    copyto!(buffers.links, values)

    dt = 0.002f0
    noise_amplitude = sqrt(2.0f0)
    κ = 0.2705f0
    λ = 0.022f0
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    n = Int32(length(u))
    @cuda threads=768 blocks=cld(length(u), 768) modelb_fixed_noise_stage1_kernel!(
        buffers.h01, buffers.k1, buffers.dW, buffers.q,
        u, buffers.links, sqrt(dt), dt,
        two_kappa, m02, c3, 1.0f0, noise_amplitude,
        buffers.lattice, n,
    )
    CUDA.synchronize()

    wx = @view buffers.links[:, :, :, 1]
    wy = @view buffers.links[:, :, :, 2]
    zx = @view buffers.links[:, :, :, 3]
    zy = @view buffers.links[:, :, :, 4]
    reference_dW = sqrt(dt) .* (
        wx .- circshift(wx, (1, 0, 0)) .+
        wy .- circshift(wy, (0, 1, 0))
    )
    reference_dZ = sqrt(dt) .* (
        zx .- circshift(zx, (1, 0, 0)) .+
        zy .- circshift(zy, (0, 1, 0))
    )
    reference_q = 0.5f0 * noise_amplitude .* (
        reference_dW .+ SOSRA_INV_SQRT3 .* reference_dZ
    )
    reference_k1 = similar(u)
    launch_modelb_drift!(
        reference_k1, u, two_kappa, m02, c3, 1.0f0,
    )
    reference_h01 = u .+ dt * SOSRA_A21 .* reference_k1 .+
                    SOSRA_B21 .* reference_q
    CUDA.synchronize()

    @test buffers.dW == reference_dW
    @test buffers.q == reference_q
    @test buffers.k1 == reference_k1
    @test buffers.h01 == reference_h01
end

@testset "power-of-two lattice sizes" begin
    p = ModelBFixedParameters(1.0f0, 1.0f0, 0.2705f0, 0.022f0)
    two_kappa = 2.0f0 * p.κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * p.λ) - 8.0f0 * p.κ
    c3 = 4.0f0 * p.λ

    lattices = (
        (32, (Int32(31), Int32(5), Int32(10))),
        (64, (Int32(63), Int32(6), Int32(12))),
        (128, (Int32(127), Int32(7), Int32(14))),
    )
    for (L, expected_lattice) in lattices
        values = sin.(0.013f0 .* Float32.(1:(2 * L * L)))
        u = CuArray(reshape(values, L, L, 2))
        fixed_drift = similar(u)
        reference_drift = similar(u)

        modelb_fixed_problem_drift!(fixed_drift, u, p, 0.0f0)
        launch_modelb_drift!(
            reference_drift, u, two_kappa, m02, c3, 1.0f0,
        )
        CUDA.synchronize()

        @test modelb_fixed_lattice(Val(L)) == expected_lattice
        @test fixed_drift == reference_drift

        u .-= mean(u; dims=(1, 2))
        small_prob = modelB_2d_fixed_problem(
            u; tspan=(0.0f0, 0.004f0),
        )
        buffers = modelb_fixed_buffers(small_prob.u0; seed=UInt64(1))
        alg = ModelBFixedSOSRA(; buffers)
        result = solve(small_prob, alg; dt=0.002f0, seed=UInt64(99))

        @test all(isfinite, Array(result.u[end]))
        mean_change = maximum(abs, Array(
            mean(result.u[end]; dims=(1, 2)) -
            mean(small_prob.u0; dims=(1, 2)),
        ))
        @test mean_change < 2.0f-6
    end

    @test_throws AssertionError modelB_2d_fixed_problem(
        CUDA.zeros(Float32, 48, 48, 1),
    )
end
