using CUDA

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

u = CUDA.randn(Float32, 256, 256, 128)
buffers = modelb_fixed_buffers(u; seed=UInt64(20260720))
args = (
    0.002f0,
    0.56f0,
    2.0f0 * (1.0f0 - 2.0f0 * 0.022f0) - 8.0f0 * 0.28f0,
    0.088f0,
    1.0f0,
    sqrt(2.0f0),
)

modelb_fixed_step!(u, buffers, args...)
CUDA.synchronize()
modelb_fixed_step!(u, buffers, args...)
CUDA.synchronize()
