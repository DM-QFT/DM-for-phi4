using CUDA
using Statistics
using Test

include(joinpath(@__DIR__, "kernels_variants.jl"))
include(joinpath(@__DIR__, "fused_fixed_sosra.jl"))
using .H100KernelVariants
using .FusedFixedSOSRA128

const DT = 0.002f0

function cpu_packed_increments(links, dt)
    L, _, ensembles, components = size(links)
    @assert L == 128 && components == 4
    dW = zeros(Float32, L, L, ensembles)
    dZ = similar(dW)
    scale = sqrt(dt)
    @inbounds for e in 1:ensembles, y in 1:L, x in 1:L
        xm1 = ((x - 2) & 127) + 1
        ym1 = ((y - 2) & 127) + 1
        dW[x, y, e] = scale * (
            links[x, y, e, 1] - links[xm1, y, e, 1] +
            links[x, y, e, 2] - links[x, ym1, e, 2]
        )
        dZ[x, y, e] = scale * (
            links[x, y, e, 3] - links[xm1, y, e, 3] +
            links[x, y, e, 4] - links[x, ym1, e, 4]
        )
    end
    return dW, dZ
end

function main()
    CUDA.seed!(20260718)
    u0 = CUDA.randn(Float32, 128, 128, 2)
    u0 .-= mean(u0; dims=(1, 2))
    params = ModelBParameters()
    workspace = allocate_workspace(u0)

    # Generate the increments once.  Both implementations below consume these
    # exact arrays, so this is a numerical formula test rather than an RNG test.
    seed_packed_rng!(workspace, 271828)
    fill_packed_normals!(workspace)
    launch_packed_increments!(workspace, DT)
    CUDA.synchronize()

    cpu_dW, cpu_dZ = cpu_packed_increments(Array(workspace.links), DT)
    noise_error_W = maximum(abs, Array(workspace.dW) .- cpu_dW)
    noise_error_Z = maximum(abs, Array(workspace.dZ) .- cpu_dZ)

    # Naive SOSRA formula: three standalone direct-stencil kernels plus one
    # broadcast for each algebraic stage.  No fused stage kernel is used here.
    q = similar(u0)
    k1 = similar(u0)
    k2 = similar(u0)
    k3 = similar(u0)
    h01 = similar(u0)
    h02 = similar(u0)
    u_naive = copy(u0)
    launch_drift_linear128!(
        k1, u_naive, params.two_kappa, params.m02, params.c3,
        params.inv_gamma; threads=1024, blocks=528,
    )
    @. q = 0.5f0 * params.sigma * (
        workspace.dW + inv(sqrt(3.0f0)) * workspace.dZ
    )
    @. h01 = u_naive + DT * FusedFixedSOSRA128.SOSRA_A21 * k1 +
             FusedFixedSOSRA128.SOSRA_B21 * q
    launch_drift_linear128!(
        k2, h01, params.two_kappa, params.m02, params.c3,
        params.inv_gamma; threads=1024, blocks=528,
    )
    @. h02 = u_naive + DT * (
        FusedFixedSOSRA128.SOSRA_A31 * k1 +
        FusedFixedSOSRA128.SOSRA_A32 * k2
    ) + FusedFixedSOSRA128.SOSRA_B31_PLUS_B32 * q
    launch_drift_linear128!(
        k3, h02, params.two_kappa, params.m02, params.c3,
        params.inv_gamma; threads=1024, blocks=528,
    )
    @. u_naive = u_naive + DT * (
        FusedFixedSOSRA128.SOSRA_ALPHA1 * k1 +
        FusedFixedSOSRA128.SOSRA_ALPHA2 * k2 +
        FusedFixedSOSRA128.SOSRA_ALPHA3 * k3
    ) + params.sigma * workspace.dW

    # Fused implementation, with the already-generated dW/dZ.
    u_fused = copy(u0)
    launch_stage1!(u_fused, workspace, DT, params)
    CUDA.synchronize()
    k1_error = maximum(abs, workspace.k1 .- k1)
    h01_error = maximum(abs, workspace.h01 .- h01)
    launch_stage2!(u_fused, workspace, DT, params)
    CUDA.synchronize()
    k2_error = maximum(abs, workspace.k2 .- k2)
    h02_error = maximum(abs, workspace.h02 .- h02)
    launch_stage3!(u_fused, workspace, DT, params)
    CUDA.synchronize()
    final_error = maximum(abs, u_fused .- u_naive)

    dW_sums = vec(Array(sum(workspace.dW; dims=(1, 2))))
    dZ_sums = vec(Array(sum(workspace.dZ; dims=(1, 2))))
    initial_sums = vec(Array(sum(u0; dims=(1, 2))))
    final_sums = vec(Array(sum(u_fused; dims=(1, 2))))
    link_values = Array(workspace.links)
    dW_values = Array(workspace.dW)
    dZ_values = Array(workspace.dZ)

    println("noise_error_W=", noise_error_W)
    println("noise_error_Z=", noise_error_Z)
    println("k1_error=", k1_error)
    println("h01_error=", h01_error)
    println("k2_error=", k2_error)
    println("h02_error=", h02_error)
    println("final_error=", final_error)
    println("max_abs_sum_dW=", maximum(abs, dW_sums))
    println("max_abs_sum_dZ=", maximum(abs, dZ_sums))
    println("max_abs_initial_sum=", maximum(abs, initial_sums))
    println("max_abs_final_sum=", maximum(abs, final_sums))
    println("links_mean=", mean(link_values), " links_var=", var(link_values))
    println("dW_mean=", mean(dW_values), " dW_var=", var(dW_values))
    println("dZ_mean=", mean(dZ_values), " dZ_var=", var(dZ_values))
    println("expected_increment_var=", 4.0f0 * DT)
    println("finite=", all(isfinite, u_fused))

    @testset "packed noise and fused fixed-step SOSRA" begin
        @test noise_error_W <= 2.0f-7
        @test noise_error_Z <= 2.0f-7
        @test k1_error <= 2.0f-5
        @test h01_error <= 2.0f-6
        @test k2_error <= 2.0f-5
        @test h02_error <= 3.0f-6
        @test final_error <= 3.0f-6
        @test maximum(abs, dW_sums) <= 3.0f-6
        @test maximum(abs, dZ_sums) <= 3.0f-6
        @test maximum(abs, final_sums .- initial_sums) <= 2.0f-3
        @test all(isfinite, u_fused)
    end
end

main()
