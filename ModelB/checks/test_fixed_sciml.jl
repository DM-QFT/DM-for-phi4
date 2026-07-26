using CUDA, Statistics, Test

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed_sciml.jl"))

function reference_modelb_drift(u, two_kappa, m02, c3, inv_gamma)
    left = circshift(u, (1, 0, 0))
    right = circshift(u, (-1, 0, 0))
    down = circshift(u, (0, 1, 0))
    up = circshift(u, (0, -1, 0))
    nearest_sum = left .+ right .+ down .+ up
    diagonal_sum =
        circshift(u, (1, 1, 0)) .+ circshift(u, (1, -1, 0)) .+
        circshift(u, (-1, 1, 0)) .+ circshift(u, (-1, -1, 0))
    second_neighbor_sum =
        circshift(u, (2, 0, 0)) .+ circshift(u, (-2, 0, 0)) .+
        circshift(u, (0, 2, 0)) .+ circshift(u, (0, -2, 0))

    minus_biharmonic =
        -20.0f0 .* u .+ 8.0f0 .* nearest_sum .-
        2.0f0 .* diagonal_sum .- second_neighbor_sum
    laplacian_phi = nearest_sum .- 4.0f0 .* u
    laplacian_phi3 =
        left .* left .* left .+ right .* right .* right .+
        down .* down .* down .+ up .* up .* up .-
        4.0f0 .* u .* u .* u
    return inv_gamma .* (
        two_kappa .* minus_biharmonic .+
        m02 .* laplacian_phi .+
        c3 .* laplacian_phi3
    )
end

function direct_checkpoints(prob, seed, dt, steps)
    u = copy(prob.u0)
    buffers = modelb_fixed_buffers(u; seed)
    states = typeof(u)[copy(u)]
    previous_step = 0
    for step in steps
        modelB_2d_fixed!(
            u, buffers;
            γ=prob.p.γ, T=prob.p.T, κ=prob.p.κ, λ=prob.p.λ,
            tspan=(0.0f0, (step - previous_step) * dt), dt=dt,
        )
        push!(states, copy(u))
        previous_step = step
    end
    CUDA.synchronize()
    return states
end

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
    @test sol_reused.t == Float32[0.02f0]
    @test length(sol_reused.u) == 1
    @test maximum(abs.(final_mean .- initial_mean)) < 2.0f-6
    @test all(isfinite, Array(sol.u[end]))

    copied = modelB_2d_fixed(
        prob.u0; tspan=(0.0f0, 0.0f0), seed=UInt64(1),
    )
    @test copied !== prob.u0
    @test copied == prob.u0

    @test_throws AssertionError solve(prob, ModelBFixedSOSRA(); adaptive=true)
    @test_throws AssertionError solve(prob, ModelBFixedSOSRA(); dense=true)
    @test_throws AssertionError solve(
        prob, ModelBFixedSOSRA(); save_everystep=true,
    )
    @test_throws MethodError solve(
        prob, ModelBFixedSOSRA(); callback=identity,
    )
end

@testset "ModelBFixedSOSRA saveat" begin
    dt = 0.002f0
    saved = solve(
        prob, ModelBFixedSOSRA();
        dt, saveat=0.004f0, save_start=true, seed,
    )
    expected_times = Float32[0.0f0, 0.004f0, 0.008f0,
                             0.012f0, 0.016f0, 0.02f0]
    expected_states = direct_checkpoints(prob, seed, dt, [2, 4, 6, 8, 10])

    @test saved.t == expected_times
    @test length(saved.u) == length(expected_states)
    @test MODEL_B_SCIML_BASE.successful_retcode(saved)
    @test saved.stats.nf == 30
    @test saved.stats.naccept == 10
    for i in eachindex(saved.u)
        @test saved.u[i] == expected_states[i]
        @test all(isfinite, saved.u[i])
        mean_change = maximum(abs, Array(
            mean(saved.u[i]; dims=(1, 2)) - mean(prob.u0; dims=(1, 2)),
        ))
        @test mean_change < 2.0f-6
        for j in 1:(i - 1)
            @test saved.u[i] !== saved.u[j]
        end
    end

    without_start = solve(
        prob, ModelBFixedSOSRA(); dt, saveat=0.004f0, seed,
    )
    @test without_start.t == expected_times[2:end]
    @test without_start.u[end] == direct

    selected = solve(
        prob, ModelBFixedSOSRA();
        dt, saveat=Float32[0.01f0, 0.004f0, 0.02f0, 0.004f0], seed,
    )
    @test selected.t == Float32[0.004f0, 0.01f0, 0.02f0]
    @test selected.u[end] == direct

    limited = solve(
        prob, ModelBFixedSOSRA();
        dt, saveat=0.006f0, save_start=true, maxiters=7, seed,
    )
    limited_states = direct_checkpoints(prob, seed, dt, [3, 6, 7])
    @test limited.t == Float32[0.0f0, 0.006f0, 0.012f0, 0.014f0]
    @test limited.retcode == MODEL_B_SCIML_BASE.ReturnCode.MaxIters
    @test limited.stats.nf == 21
    @test limited.stats.naccept == 7
    @test all(limited.u[i] == limited_states[i] for i in eachindex(limited.u))

    @test_throws ArgumentError solve(
        prob, ModelBFixedSOSRA(); dt, saveat=0.003f0, seed,
    )
    @test_throws ArgumentError solve(
        prob, ModelBFixedSOSRA(); dt, saveat=0.0f0, seed,
    )
end

@testset "deterministic fused SOSRA stages" begin
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
    u_initial = copy(u)

    @test !hasproperty(buffers, :dW)
    @test !hasproperty(buffers, :k2)

    dt = 0.002f0
    noise_amplitude = sqrt(2.0f0)
    κ = 0.2705f0
    λ = 0.022f0
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    n = Int32(length(u))
    @cuda threads=768 blocks=cld(length(u), 768) modelb_fixed_stage1_kernel!(
        buffers.h01, buffers.h02, buffers.k1, buffers.q,
        u, buffers.links, sqrt(dt), dt,
        two_kappa, m02, c3, 1.0f0, noise_amplitude,
        buffers.lattice, n,
    )

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
    reference_k1 = reference_modelb_drift(
        u_initial, two_kappa, m02, c3, 1.0f0,
    )
    reference_h01 = u_initial .+ dt * SOSRA_A21 .* reference_k1 .+
                    SOSRA_B21 .* reference_q
    reference_partial_u =
        u_initial .+ dt * SOSRA_ALPHA1 .* reference_k1 .+
        noise_amplitude .* reference_dW
    CUDA.synchronize()

    @test buffers.q == reference_q
    @test buffers.k1 == reference_k1
    @test buffers.h01 == reference_h01
    @test buffers.h02 == reference_partial_u

    reference_k2 = reference_modelb_drift(
        reference_h01, two_kappa, m02, c3, 1.0f0,
    )
    reference_u_after_stage2 =
        reference_partial_u .+ dt * SOSRA_ALPHA2 .* reference_k2
    reference_stage3_input =
        u_initial .+ dt .* (
            SOSRA_A31 .* reference_k1 .+ SOSRA_A32 .* reference_k2
        ) .+ SOSRA_B31_B32 .* reference_q

    @cuda threads=1024 blocks=cld(length(u), 1024) modelb_fixed_stage2_kernel!(
        u, buffers.h02, buffers.h01, buffers.k1, buffers.q, dt,
        two_kappa, m02, c3, 1.0f0,
        buffers.lattice, n,
    )
    CUDA.synchronize()

    @test u == reference_u_after_stage2
    @test buffers.h02 == reference_stage3_input

    reference_k3 = reference_modelb_drift(
        reference_stage3_input, two_kappa, m02, c3, 1.0f0,
    )
    reference_final =
        reference_u_after_stage2 .+ dt * SOSRA_ALPHA3 .* reference_k3
    old_sum_order =
        u_initial .+ dt .* (
            SOSRA_ALPHA1 .* reference_k1 .+
            SOSRA_ALPHA2 .* reference_k2 .+
            SOSRA_ALPHA3 .* reference_k3
        ) .+ noise_amplitude .* reference_dW

    @cuda threads=1024 blocks=cld(length(u), 1024) modelb_fixed_stage3_kernel!(
        u, buffers.h02, dt,
        two_kappa, m02, c3, 1.0f0,
        buffers.lattice, n,
    )
    CUDA.synchronize()

    @test u == reference_final
    old_order_tolerance = 16.0f0 * eps(Float32) *
                          max(1.0f0, maximum(abs, old_sum_order))
    @test maximum(abs, u .- old_sum_order) <= old_order_tolerance
    @test maximum(abs, Array(
        mean(u; dims=(1, 2)) - mean(u_initial; dims=(1, 2)),
    )) < 2.0f-6
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
        reference_drift = reference_modelb_drift(
            u, two_kappa, m02, c3, 1.0f0,
        )

        modelb_fixed_problem_drift!(fixed_drift, u, p, 0.0f0)
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
