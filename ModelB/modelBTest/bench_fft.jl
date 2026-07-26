using CUDA, CUDA.CUFFT, LinearAlgebra, Statistics, Printf

# Is a Fourier-space (exponential / semi-implicit) integrator affordable?
# One ETD step needs one C2R (state -> real space), one R2C (phi^3), and either a
# third R2C for the noise or direct k-space sampling.

bench(f) = (for _ in 1:5; f(); end; CUDA.synchronize();
            median([CUDA.@elapsed(f()) for _ in 1:20]))

@printf("%5s %7s %12s %12s %12s %12s\n",
        "L", "Ne", "elements", "R2C ms", "C2R ms", "3 FFT ms")
for (L, Ne) in ((128, 512), (128, 10240), (256, 128))
    u = CUDA.randn(Float32, L, L, Ne)
    pf = plan_rfft(u, (1, 2))
    uk = pf * u
    pi_ = plan_irfft(uk, L, (1, 2))
    t_f = bench(() -> mul!(uk, pf, u))
    t_i = bench(() -> mul!(u, pi_, uk))
    @printf("%5d %7d %12d %12.4f %12.4f %12.4f\n",
            L, Ne, length(u), 1e3t_f, 1e3t_i, 1e3(2t_f + t_i))
end
