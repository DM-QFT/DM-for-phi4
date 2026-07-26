using CUDA, Printf, Statistics

include(joinpath(@__DIR__, "fused_philox.jl"))

function benchmark_tile(u, buffers, ::Val{TY}, launches, repetitions) where TY
    L = size(u, 1)
    dt = 0.002f0
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
        Val(TY),
    )
    kernel = @cuda launch=false tiled_philox_stage1_kernel!(args...)
    @cuda threads=(32, TY) blocks=(L ÷ 32, L ÷ TY, size(u, 3)) tiled_philox_stage1_kernel!(
        args...,
    )
    times = Float64[]
    for _ in 1:repetitions
        CUDA.synchronize()
        seconds = CUDA.@elapsed begin
            for _ in 1:launches
                @cuda threads=(32, TY) blocks=(L ÷ 32, L ÷ TY, size(u, 3)) tiled_philox_stage1_kernel!(
                    args...,
                )
            end
        end
        push!(times, seconds)
    end
    @printf(
        "%d\t%.6f\t%d\t%d\n",
        TY,
        1.0e3 * median(times) / launches,
        CUDA.registers(kernel),
        CUDA.memory(kernel).local,
    )
end

function main()
    L, Nensemble = 256, 128
    u = CUDA.randn(Float32, L, L, Nensemble)
    buffers = fused_philox_buffers(u; seed=UInt64(20260726))
    for tile_y in (Val(4), Val(8), Val(16))
        benchmark_tile(u, buffers, tile_y, 1000, 7)
    end
end

main()
