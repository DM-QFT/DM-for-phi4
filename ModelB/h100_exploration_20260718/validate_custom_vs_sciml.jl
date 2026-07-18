include(joinpath(@__DIR__, "solver_kernel_variants.jl"))
include(joinpath(@__DIR__, "rng_variants.jl"))
include(joinpath(@__DIR__, "fused_fixed_sosra.jl"))

using .FusedFixedSOSRA128
using .H100RNGVariants
using Printf

function main()
    ensembles = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
    steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    dt = 0.002f0
    t_end = steps * dt
    rng_seed = 271828

    u0 = make_u0(ensembles=ensembles, seed=20260718)

    custom = copy(u0)
    workspace = allocate_workspace(custom; rng_kind=:library_philox, seed=rng_seed)
    solve_fixed_sosra!(
        custom, workspace, ModelBParameters();
        dt, t_end,
        threads=1024, blocks=264,
        noise_threads=512, noise_blocks=528,
    )
    CUDA.synchronize()

    builder = (t, u) -> packed_noise_process128(
        t, u;
        rng_kind=:library_philox,
        seed=rng_seed,
        synchronize_callbacks=false,
        rswm=DiffEqNoiseProcess.RSWM(adaptivealg=:RSwM1),
    )
    solution = solve_sosra(
        u0;
        drift_launch! = drift_linear128_best!,
        noise_builder=builder,
        synchronize_callbacks=false,
        dt,
        t_end,
    )
    CUDA.synchronize()
    sciml = solution.u[end]

    difference = custom .- sciml
    max_abs = maximum(abs, difference)
    rms = sqrt(sum(abs2, difference) / length(difference))
    exact = all(custom .== sciml)
    @printf(
        "ensembles=%d steps=%d exact=%s max_abs_difference=%.9g rms_difference=%.9g custom_max_mass=%.9g sciml_max_mass=%.9g\n",
        ensembles,
        steps,
        string(exact),
        max_abs,
        rms,
        maximum(abs.(sum(custom; dims=(1, 2)))),
        maximum(abs.(sum(sciml; dims=(1, 2)))),
    )
end

main()
