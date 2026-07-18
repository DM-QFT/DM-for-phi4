using CUDA
using LinearAlgebra

function one_gemm_seconds!(C, A, B)
    CUDA.synchronize()
    start_ns = time_ns()
    mul!(C, A, B)
    CUDA.synchronize()
    return Float64(time_ns() - start_ns) * 1.0e-9
end

function run_gemms!(C, A, B, iterations)
    start_ns = time_ns()
    for _ in 1:iterations
        mul!(C, A, B)
        CUDA.synchronize()
    end
    return Float64(time_ns() - start_ns) * 1.0e-9
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8192
    seconds = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 12.0
    A = CUDA.randn(Float32, n, n)
    B = CUDA.randn(Float32, n, n)
    C = similar(A)

    mul!(C, A, B)
    CUDA.synchronize()
    calibration = minimum(one_gemm_seconds!(C, A, B) for _ in 1:3)
    iterations = max(1, ceil(Int, seconds / calibration))
    println("POWER_WINDOW_START epoch=", time())
    flush(stdout)
    elapsed = run_gemms!(C, A, B, iterations)
    println("POWER_WINDOW_END epoch=", time())
    checksum = Array(@view C[1:1])[1]
    println("GEMM_CONTROL n=", n, " iterations=", iterations,
            " elapsed=", elapsed, " checksum=", checksum)
end

main()
