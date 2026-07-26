using CUDA, Statistics, Test

include(joinpath(@__DIR__, "..", "solver", "ModelB.jl"))

u0 = zeros(Float32, 16, 16, 2)

function endpoint(seed)
    solution = modelB_2d_SDE_prob(
        u0;
        tspan=(0.0f0, 0.004f0),
        saveat=0.004f0,
        dt=0.002f0,
        seed,
    )
    return Array(solution.u[end])
end

first = endpoint(UInt64(12345))
repeated = endpoint(UInt64(12345))
different = endpoint(UInt64(12346))

@testset "generic ModelB explicit GPU RNG" begin
    @test first == repeated
    @test first != different
    @test all(isfinite, first)
    @test maximum(abs, mean(first; dims=(1, 2))) < 2.0f-6
end
