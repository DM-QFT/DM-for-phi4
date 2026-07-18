# Equilibrium axis-averaged structure factor S(k) via MC, for cross-checking Model B.
# Usage: julia mc_sk.jl <outfile> <L> <T> <ntherm> <nmeas> <stride>
using Random, Statistics, Printf, FFTW
include(joinpath(@__DIR__, "mc_common.jl"))

outfile = ARGS[1]
L = parse(Int, ARGS[2])
T = parse(Float64, ARGS[3])
ntherm = parse(Int, ARGS[4])
nmeas = parse(Int, ARGS[5])
stride = parse(Int, ARGS[6])

rng = MersenneTwister(8241)
phi = randn(rng, L, L)
incl = falses(L * L)
stack = Int[]
delta = 1.5
for _ in 1:ntherm
    metrop_sweep!(phi, T, delta, rng)
    wolff!(phi, T, rng, incl, stack)
end
Sacc = zeros(L ÷ 2)
p2 = 0.0
nsamp = 0
for s in 1:nmeas
    metrop_sweep!(phi, T, delta, rng)
    wolff!(phi, T, rng, incl, stack)
    if s % stride == 0
        F = fft(phi)
        for n in 1:L÷2
            Sacc[n] += (abs2(F[n+1, 1]) + abs2(F[1, n+1])) / (2 * L^2)
        end
        global p2 += mean(abs2, phi)
        global nsamp += 1
    end
end
Sacc ./= nsamp
open(outfile, "w") do io
    @printf(io, "# MC L=%d T=%.4f  <phi^2>=%.6f  (axis-averaged S(k))\n", L, T, p2 / nsamp)
    for n in 1:L÷2
        @printf(io, "%.6f  %.6f\n", 2pi * n / L, Sacc[n])
    end
end
@printf("MC S(k) done: L=%d T=%.3f <phi^2>=%.5f S(kmin)=%.4f\n", L, T, p2 / nsamp, Sacc[1])
