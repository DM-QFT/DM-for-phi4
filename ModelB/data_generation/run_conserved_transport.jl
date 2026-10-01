# Record the total time-integrated current along one coarsening trajectory.
# Reuse the existing SOSRA reference at t=100 and continue with a new noise seed.
using CUDA, JLD2, NPZ, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

@inline function chemical_potential(u, i, lattice, two_kappa, m02, c3)
    mask, shift, _ = modelb_fixed_lattice(lattice)
    x, y, base, row = modelb_fixed_geometry(i, lattice)
    @inbounds begin
        center = Float64(u[i])
        neighbors =
            Float64(u[row + ((x - Int32(1)) & mask) + Int32(1)]) +
            Float64(u[row + ((x + Int32(1)) & mask) + Int32(1)]) +
            Float64(u[base + (((y - Int32(1)) & mask) << shift) + x + Int32(1)]) +
            Float64(u[base + (((y + Int32(1)) & mask) << shift) + x + Int32(1)])
    end
    return -Float64(two_kappa) * (neighbors - 4 * center) +
           Float64(m02) * center + Float64(c3) * center^3
end

function integrate_current!(potential, noise, previous, h01, h02, links,
                            lattice, dt, two_kappa, m02, c3, inv_gamma, amplitude, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        # The existing SOSRA step leaves its two internal stage fields in h01/h02.
        mu1 = chemical_potential(previous, i, lattice, two_kappa, m02, c3)
        mu2 = chemical_potential(h01, i, lattice, two_kappa, m02, c3)
        mu3 = chemical_potential(h02, i, lattice, two_kappa, m02, c3)
        @inbounds begin
            potential[i] += Float64(inv_gamma) * (
                Float64(dt * SOSRA_ALPHA1) * mu1 +
                Float64(dt * SOSRA_ALPHA2) * mu2 +
                Float64(dt * SOSRA_ALPHA3) * mu3
            )
            noise[i] -= amplitude * Float64(links[i])
            noise[n + i] -= amplitude * Float64(links[n + i])
        end
    end
    return nothing
end

function main()
    root = normpath(joinpath(@__DIR__, ".."))
    source = joinpath(root, "results/modelb_reference/quench_tanhgauss_quench_L128_N32.h5")
    output = joinpath(root, "results/coarsening/conserved_transport.npz")
    κ, λ, γ, T, dt = 0.85f0, 1.0f0, 1.0f0, 1.0f0, 0.002f0
    times = Float32[100, 150, 200, 250, 300]
    seed = UInt64(20260920)
    initial = jldopen(source, "r") do archive
        index = findfirst(==(times[1]), archive["times"])
        # Julia reference storage is [y,x,ensemble,time]. Use [x,y] here.
        permutedims(archive["cfgs"][:, :, 1, index], (2, 1))
    end
    L = size(initial, 1)
    u = CuArray(reshape(initial, L, L, 1))
    buffers = modelb_fixed_buffers(u; seed)
    previous = similar(u)
    potential = CUDA.zeros(Float64, size(u))
    noise = CUDA.zeros(Float64, (size(u)..., 2))
    two_kappa, m02, c3 = 2κ, 2 * (1 - 2λ) - 8κ, 4λ
    inv_gamma = 1f0 / γ
    noise_amplitude = sqrt(2f0 * T * inv_gamma)
    amplitude = Float64(noise_amplitude) * Float64(sqrt(dt))
    n = Int32(length(u))

    phi = Array{Float32}(undef, L, L, length(times))
    jx = zeros(Float64, size(phi))
    jy = zeros(Float64, size(phi))
    phi[:, :, 1] = initial
    for index in 2:length(times)
        steps, _ = modelb_fixed_step_count((times[index - 1], times[index]), dt)
        for _ in 1:steps
            copyto!(previous, u)
            modelb_fixed_step!(u, buffers, dt, two_kappa, m02, c3, inv_gamma, noise_amplitude)
            @cuda threads=256 blocks=cld(n, 256) integrate_current!(
                potential, noise, previous, buffers.h01, buffers.h02, buffers.links,
                buffers.lattice, dt, two_kappa, m02, c3, inv_gamma, amplitude, n,
            )
        end
        phi[:, :, index] = Array(u)[:, :, 1]
        mu_integral = Array(potential)[:, :, 1]
        noise_integral = Array(noise)[:, :, 1, :]
        jx[:, :, index] = mu_integral - circshift(mu_integral, (-1, 0)) + noise_integral[:, :, 1]
        jy[:, :, index] = mu_integral - circshift(mu_integral, (0, -1)) + noise_integral[:, :, 2]
        divergence = jx[:, :, index] - circshift(jx[:, :, index], (1, 0)) +
                     jy[:, :, index] - circshift(jy[:, :, index], (0, 1))
        delta = Float64.(phi[:, :, index]) - initial
        residual = delta + divergence
        @assert all(isfinite, delta)
        @assert sqrt(mean(abs2, residual)) < 1e-3
    end
    npzwrite(output, Dict(
        "phi" => phi, "jx" => jx, "jy" => jy, "times" => times,
        "kappa" => κ, "lambda" => λ, "gamma" => γ, "temperature" => T,
        "dt" => dt, "noise_seed" => seed, "source_ensemble" => 0,
    ))
    println("saved $output")
end

main()
