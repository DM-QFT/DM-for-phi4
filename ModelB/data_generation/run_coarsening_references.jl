# 淬火粗化：无序初态 (M=0) → 破缺相，SOSRA 定步长求解器分段演化，对数时刻测量。
# L(t) 取 sign(φ) 关联函数 C(r) 径向平均的第一零点，免去 S(k) 高 k 热噪声本底的
# 截断选择。S(k) 径向 profile 一并保存供坍缩图。
#   julia data_generation/run_coarsening_references.jl
using CUDA, FFTW, JLD2, Random, Statistics

include(joinpath(@__DIR__, "..", "solver", "ModelB_fixed.jl"))

const OUTPUT = normpath(joinpath(@__DIR__, "..", "results", "coarsening"))
mkpath(OUTPUT)

const λ, T, γ, dt = 1.0f0, 1.0f0, 1.0f0, 0.002f0

# (L, Nens, κ, tmax, seed)
const RUNS = (
    (128, 128, 0.85f0, 10000, UInt64(2026081006)),
    (64, 512, 0.85f0, 5000, UInt64(2026081007)),
)

logtimes(tmax; n=28) = unique(round.(Int, 10 .^ range(0.0, log10(tmax), n)))

# 周期最短距离的整数环平均；返回 (第一零点, 径向 profile)
function radial_first_zero(C, L)
    nr = L ÷ 2
    prof = zeros(nr + 1)
    cnt = zeros(Int, nr + 1)
    for j in 1:L, i in 1:L
        r = round(Int, hypot(min(i - 1, L - i + 1), min(j - 1, L - j + 1)))
        r <= nr || continue
        prof[r+1] += C[i, j]
        cnt[r+1] += 1
    end
    prof ./= cnt
    for r in 1:nr
        prof[r+1] < 0 && return (r - 1) + prof[r] / (prof[r] - prof[r+1]), prof
    end
    return NaN, prof
end

function radial_Sk(S, L)
    nb = L ÷ 2
    prof = zeros(nb + 1)
    cnt = zeros(Int, nb + 1)
    for j in 1:L, i in 1:L
        nx = i - 1 > L ÷ 2 ? i - 1 - L : i - 1
        ny = j - 1 > L ÷ 2 ? j - 1 - L : j - 1
        m = round(Int, hypot(nx, ny))
        m <= nb || continue
        prof[m+1] += S[i, j]
        cnt[m+1] += 1
    end
    prof ./ cnt
end

for (L, Nens, κ, tmax, seed) in RUNS
    tsaves = logtimes(tmax)
    nt = length(tsaves)
    rng = Xoshiro(seed)
    u0 = 0.1f0 .* randn(rng, Float32, L, L, Nens)
    u0 .-= mean(u0; dims=(1, 2))
    u = CuArray(u0)
    buffers = modelb_fixed_buffers(u; seed)

    Lt = fill(NaN, nt)
    Mdrift = zeros(nt)
    Sk = zeros(L ÷ 2 + 1, nt)
    Cr = zeros(L ÷ 2 + 1, nt)
    snaps = Array{Float32}(undef, L, L, nt)

    println("=== L=$L N=$Nens κ=$κ tmax=$tmax ===")
    tprev = 0
    for (i, t) in enumerate(tsaves)
        seconds = @elapsed begin
            modelB_2d_fixed!(u, buffers; γ, T, κ, λ, dt,
                tspan=(0.0f0, Float32(t - tprev)))
            CUDA.synchronize()
        end
        tprev = t
        @assert all(isfinite, u)
        ψ = sign.(u)
        S2 = Array(dropdims(mean(abs2.(fft(ψ, (1, 2))); dims=3); dims=3)) ./ L^2
        C = real.(ifft(S2))
        C ./= C[1, 1]
        Lt[i], Cr[:, i] = radial_first_zero(C, L)
        Sk[:, i] = radial_Sk(S2, L)
        Mdrift[i] = maximum(abs, Array(mean(u; dims=(1, 2))))
        snaps[:, :, i] = Array(@view u[:, :, 1])
        println("  t=$t: L(t)=$(round(Lt[i]; digits=3)) " *
                "max|M|=$(round(Mdrift[i]; sigdigits=3)) ($(round(seconds; digits=1)) s)")
        flush(stdout)
    end

    out = joinpath(OUTPUT, "quench_L$(L)_k$(κ).jld2")
    jldsave(out; times=Float64.(tsaves), Lt, Mdrift, Sk, Cr, snaps,
        L, Nens, kappa=κ, lambda=λ, temperature=T, gamma=γ, dt, seed,
        layout="h5py: Sk/Cr[t, ring m], snaps[t, x, y]; k_m = 2π m / L")
    println("saved $out")
end
