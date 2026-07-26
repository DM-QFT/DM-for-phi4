using CUDA, Random

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

L, Nensemble = 256, 128
dt = 0.002f0
u = CUDA.randn(Float32, L, L, Nensemble)
buffers = modelb_fixed_buffers(u; seed=UInt64(20260720))
two_kappa = 0.56f0
m02 = 2.0f0 * (1.0f0 - 2.0f0 * 0.022f0) - 8.0f0 * 0.28f0
c3 = 0.088f0
inv_gamma = 1.0f0
noise_amplitude = sqrt(2.0f0)

for _ in 1:20
    modelb_fixed_step!(
        u, buffers, dt, two_kappa, m02, c3, inv_gamma, noise_amplitude,
    )
end
CUDA.synchronize()

profile = CUDA.@profile begin
    for _ in 1:20
        modelb_fixed_step!(
            u, buffers, dt, two_kappa, m02, c3, inv_gamma, noise_amplitude,
        )
    end
    CUDA.synchronize()
end
display(profile)
