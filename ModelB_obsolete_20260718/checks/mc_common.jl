# Shared MC routines for the lattice Hamiltonian implied by ModelB.jl:
# H = sum_x [ (1/2) sum_mu (phi_{x+mu}-phi_x)^2 - phi^2/2 + phi^4/24 ],  P ∝ exp(-H/T)

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

# Embedded-Ising Wolff cluster: bond activated with p = 1 - exp(-2 phi_x phi_y / T)
# when phi_x phi_y > 0; cluster flips sign (Brower-Tamayo).
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
