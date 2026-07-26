using CUDA, Statistics, Printf

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

# Roofline anchor: pure streaming copy on this GPU.
function stream_copy!(dst, src)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for k in i:stride:Int32(length(src))
        dst[k] = src[k]
    end
    return nothing
end

function peak_bandwidth(n=1 << 26)
    a = CUDA.rand(Float32, n)
    b = similar(a)
    threads = 1024
    blocks = min(cld(n, threads), 264)
    for _ in 1:3
        @cuda threads=threads blocks=blocks stream_copy!(b, a)
    end
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:20
        t = CUDA.@elapsed @cuda threads=threads blocks=blocks stream_copy!(b, a)
        push!(ts, t)
    end
    bytes = 2 * n * sizeof(Float32)
    return bytes / median(ts) / 1e12   # TB/s
end

function bench_step(L, Ne; steps=200, dt=0.002f0, κ=0.28f0, λ=0.022f0, γ=1.0f0, T=1.0f0)
    u = CUDA.randn(Float32, L, L, Ne)
    buffers = modelb_fixed_buffers(u; seed=UInt64(1))
    inv_gamma = 1.0f0 / γ
    two_kappa = 2.0f0 * κ
    m02 = 2.0f0 * (1.0f0 - 2.0f0 * λ) - 8.0f0 * κ
    c3 = 4.0f0 * λ
    amp = sqrt(2.0f0 * T * inv_gamma)
    for _ in 1:20
        modelb_fixed_step!(u, buffers, dt, two_kappa, m02, c3, inv_gamma, amp)
    end
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:5
        t = CUDA.@elapsed begin
            for _ in 1:steps
                modelb_fixed_step!(u, buffers, dt, two_kappa, m02, c3, inv_gamma, amp)
            end
        end
        push!(ts, t / steps)
    end
    return median(ts)
end

# Bytes DRAM traffic per step, assuming stencil reads hit cache (1 read of u per
# stencil) and every buffer write/read is a full array pass.
function traffic_model(n)
    w = n * sizeof(Float32)
    curand = 4w                       # write wx, wy, zx, zy
    stage1 = 1w + 4w + 4w             # read u stencil, read 4 link fields, write h01/h02/k1/q
    stage2 = 1w + 4w + 2w             # read h01 stencil, read h02/u/k1/q, write u/h02
    stage3 = 1w + 1w + 1w             # read h02 stencil, read u, write u
    return curand + stage1 + stage2 + stage3
end

bw = peak_bandwidth()
@printf("measured stream copy bandwidth: %.2f TB/s\n\n", bw)
@printf("%5s %7s %12s %10s %12s %12s\n", "L", "Ne", "elements", "ms/step", "model GB/s", "frac peak")
for (L, Ne) in ((128, 512), (256, 128), (128, 10240), (64, 2048))
    t = bench_step(L, Ne)
    n = L * L * Ne
    gbs = traffic_model(n) / t / 1e9
    @printf("%5d %7d %12d %10.4f %12.1f %11.2f\n", L, Ne, n, 1e3t, gbs, gbs / (1e3bw))
end
