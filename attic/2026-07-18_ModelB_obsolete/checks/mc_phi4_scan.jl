# Independent equilibrium MC (Metropolis + embedded Wolff) for the lattice Hamiltonian
# implied by ModelB.jl:  H = sum_x [ (1/2) sum_mu (phi_{x+mu}-phi_x)^2 - phi^2/2 + phi^4/24 ]
# P ∝ exp(-H/T).  Used to cross-check Model B equilibrium and locate T_c via Binder cumulant.
#
# Usage: julia mc_phi4_scan.jl <outfile> <L1,L2,...> <T1,T2,...> <ntherm> <nmeas>
using Random, Printf, Statistics, DelimitedFiles

Vpot(p) = -0.5 * p^2 + p^4 / 24

function metrop_sweep!(phi::Matrix{Float64}, T, delta, rng)
    L = size(phi, 1)
    acc = 0
    @inbounds for j in 1:L, i in 1:L
        ip = i == L ? 1 : i + 1; im = i == 1 ? L : i - 1
        jp = j == L ? 1 : j + 1; jm = j == 1 ? L : j - 1
        s = phi[ip, j] + phi[im, j] + phi[i, jp] + phi[i, jm]
        p0 = phi[i, j]
        p1 = p0 + delta * (2 * rand(rng) - 1)
        dH = 2 * (p1 * p1 - p0 * p0) - (p1 - p0) * s + Vpot(p1) - Vpot(p0)
        if dH <= 0 || rand(rng) < exp(-dH / T)
            phi[i, j] = p1
            acc += 1
        end
    end
    return acc / (L * L)
end

# Embedded-Ising Wolff cluster: bond (x,y) activated with p = 1 - exp(-2 phi_x phi_y / T)
# when phi_x phi_y > 0; cluster spins flip sign. Valid for continuous fields (Brower-Tamayo).
function wolff!(phi::Matrix{Float64}, T, rng, incl::BitVector, stack::Vector{Int})
    L = size(phi, 1)
    N = L * L
    fill!(incl, false)
    empty!(stack)
    seed = rand(rng, 1:N)
    incl[seed] = true
    push!(stack, seed)
    csize = 0
    @inbounds while !isempty(stack)
        idx = pop!(stack)
        csize += 1
        i = (idx - 1) % L + 1
        j = (idx - 1) ÷ L + 1
        pv = phi[idx]
        n1 = (i == L ? idx - L + 1 : idx + 1)
        n2 = (i == 1 ? idx + L - 1 : idx - 1)
        n3 = (j == L ? idx - N + L : idx + L)
        n4 = (j == 1 ? idx + N - L : idx - L)
        for nb in (n1, n2, n3, n4)
            if !incl[nb]
                b = pv * phi[nb] / T
                if b > 0 && rand(rng) < -expm1(-2 * b)
                    incl[nb] = true
                    push!(stack, nb)
                end
            end
        end
    end
    @inbounds for k in 1:N
        incl[k] && (phi[k] = -phi[k])
    end
    return csize
end

function run_point(L, T, ntherm, nmeas, rng)
    phi = randn(rng, L, L)
    incl = falses(L * L)
    stack = Int[]
    delta = 1.5
    accsum = 0.0
    for _ in 1:ntherm
        metrop_sweep!(phi, T, delta, rng)
        wolff!(phi, T, rng, incl, stack)
    end
    m1 = m2 = m4 = p2 = 0.0
    for _ in 1:nmeas
        accsum += metrop_sweep!(phi, T, delta, rng)
        wolff!(phi, T, rng, incl, stack)
        m = mean(phi)
        m1 += abs(m); m2 += m^2; m4 += m^4
        p2 += mean(abs2, phi)
    end
    m1 /= nmeas; m2 /= nmeas; m4 /= nmeas; p2 /= nmeas
    U4 = 1 - m4 / (3 * m2^2)
    chi = L^2 * (m2 - m1^2) / T
    return (m1, m2, U4, chi, p2, accsum / nmeas)
end

outfile = ARGS[1]
Ls = parse.(Int, split(ARGS[2], ","))
Ts = parse.(Float64, split(ARGS[3], ","))
ntherm = parse(Int, ARGS[4])
nmeas = parse(Int, ARGS[5])

rng = MersenneTwister(20260717)
open(outfile, "w") do io
    println(io, "# L  T  <|m|>  <m^2>  U4  chi  <phi^2>  acc")
    for L in Ls, T in Ts
        t = @elapsed res = run_point(L, T, ntherm, nmeas, rng)
        @printf(io, "%d  %.4f  %.6f  %.6f  %.6f  %.4f  %.6f  %.3f\n", L, T, res...)
        flush(io)
        @printf("L=%d T=%.3f done in %.1fs: U4=%.4f <phi^2>=%.4f\n", L, T, t, res[3], res[5])
        flush(stdout)
    end
end
