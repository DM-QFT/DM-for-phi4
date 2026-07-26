using CUDA, Printf, Statistics

include(joinpath(@__DIR__, "fused_philox.jl"))

function main()
    L, Nensemble = 256, 128
    u = CUDA.randn(Float32, L, L, Nensemble)
    buffers = fused_philox_buffers(u; seed=UInt64(20260726))
    dt = 0.002f0
    n = Int32(length(u))
    args = (
        buffers.h01,
        buffers.h02,
        buffers.k1,
        buffers.q,
        u,
        sqrt(dt),
        dt,
        0.56f0,
        -0.328f0,
        0.088f0,
        1.0f0,
        sqrt(2.0f0),
        UInt64(20260726),
        UInt64(0),
        Val(L),
        n,
    )
    repetitions = 7
    launches = 1000

    @cuda threads=768 blocks=132 fused_philox_stage1_kernel!(args...)
    CUDA.synchronize()
    for threads in (128, 256, 384, 512, 640, 768)
        for blocks_per_sm in 1:4
            blocks = 132 * blocks_per_sm
            times = Float64[]
            for _ in 1:repetitions
                CUDA.synchronize()
                seconds = CUDA.@elapsed begin
                    for _ in 1:launches
                        @cuda threads=threads blocks=blocks fused_philox_stage1_kernel!(
                            args...,
                        )
                    end
                end
                push!(times, seconds)
            end
            @printf(
                "%d\t%d\t%.6f\n",
                threads,
                blocks,
                1.0e3 * median(times) / launches,
            )
        end
    end
end

main()
