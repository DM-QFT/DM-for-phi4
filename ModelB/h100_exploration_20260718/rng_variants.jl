module H100RNGVariants

using CUDA
using DiffEqNoiseProcess
using Random

export RNG_CANDIDATES,
       RNG_ORDERINGS,
       make_gpu_rng,
       PackedNoiseCache,
       launch_packed_scaled128!,
       launch_packed_bridge128!,
       packed_noise_process128,
       DualStreamNoiseCache,
       launch_split_scaled128!,
       launch_split_bridge128!,
       dual_stream_noise_process128,
       fill_two_streams!,
       PrefetchPackedNoiseCache,
       prefetch_packed_noise_process128,
       ensemble_ranges,
       run_ensemble_shards

const L128 = 128
const L128_MASK = 127
const L128_SHIFT = 7
const L128_PLANE_SHIFT = 14

"""
RNG implementations worth benchmarking for this workload.

All `library_*` entries use the cuRAND host library. `gpuarrays_philox` uses
CUDA.jl's explicit GPUArrays Philox4x32-10 kernel. `native_philox2` is included
only as an exploratory control; CUDA.jl itself does not recommend NativeRNG
for production.
"""
const RNG_CANDIDATES = (
    :library_default,
    :library_xorwow,
    :library_philox,
    :library_mrg32k3a,
    :library_mtgp32,
    :library_mt19937,
    :gpuarrays_philox,
    :native_philox2,
)

const LIBRARY_RNG_TYPES = Dict(
    :library_default => CUDA.cuRAND.CURAND_RNG_PSEUDO_DEFAULT,
    :library_xorwow => CUDA.cuRAND.CURAND_RNG_PSEUDO_XORWOW,
    :library_philox => CUDA.cuRAND.CURAND_RNG_PSEUDO_PHILOX4_32_10,
    :library_mrg32k3a => CUDA.cuRAND.CURAND_RNG_PSEUDO_MRG32K3A,
    :library_mtgp32 => CUDA.cuRAND.CURAND_RNG_PSEUDO_MTGP32,
    :library_mt19937 => CUDA.cuRAND.CURAND_RNG_PSEUDO_MT19937,
)

const RNG_ORDERINGS = Dict(
    :best => CUDA.cuRAND.CURAND_ORDERING_PSEUDO_BEST,
    :default => CUDA.cuRAND.CURAND_ORDERING_PSEUDO_DEFAULT,
    :seeded => CUDA.cuRAND.CURAND_ORDERING_PSEUDO_SEEDED,
    :legacy => CUDA.cuRAND.CURAND_ORDERING_PSEUDO_LEGACY,
    :dynamic => CUDA.cuRAND.CURAND_ORDERING_PSEUDO_DYNAMIC,
)

"""
    make_gpu_rng(kind, seed; stream=CUDA.stream(), ordering=nothing)

Construct and seed an explicit GPU RNG.  Calling `Random.randn!(rng, A)` with
the returned object is important: `DiffEqNoiseProcess.wiener_randn!` ignores
an explicitly supplied RNG for GPU arrays in DiffEqNoiseProcess 5.33.1.
"""
function make_gpu_rng(
    kind::Symbol,
    seed::Integer;
    stream=CUDA.stream(),
    ordering::Union{Nothing, Symbol}=nothing,
)
    if haskey(LIBRARY_RNG_TYPES, kind)
        rng = CUDA.cuRAND.LibraryRNG(LIBRARY_RNG_TYPES[kind]; stream=stream)
        CUDA.stream!(stream) do
            if !isnothing(ordering)
                haskey(RNG_ORDERINGS, ordering) ||
                    throw(ArgumentError("unknown cuRAND ordering $(repr(ordering))"))
                CUDA.cuRAND.curandSetGeneratorOrdering(rng, RNG_ORDERINGS[ordering])
            end
            Random.seed!(rng, seed)
        end
        return rng
    elseif kind === :gpuarrays_philox
        isnothing(ordering) ||
            throw(ArgumentError("ordering is only valid for cuRAND LibraryRNG"))
        return CUDA.RNG(seed)
    elseif kind === :native_philox2
        isnothing(ordering) ||
            throw(ArgumentError("ordering is only valid for cuRAND LibraryRNG"))
        return CUDA.cuRAND.NativeRNG(seed)
    else
        throw(ArgumentError("unknown GPU RNG $(repr(kind)); choose one of $(RNG_CANDIDATES)"))
    end
end

@inline function check_field128(a)
    @assert ndims(a) == 3 && size(a, 1) == L128 && size(a, 2) == L128
    return nothing
end

@inline function check_packed128(out, xi_xy)
    check_field128(out)
    @assert size(xi_xy) == (L128, L128, size(out, 3), 2)
    return nothing
end

# Direct L=128 addressing.  xi_xy[:,:,:,1] and xi_xy[:,:,:,2] are contiguous
# halves of one dense allocation, so one randn! call fills both vector fields.
@inline function packed_conserved_noise_at128(xi_xy, linear, nfield)
    linear0 = linear - 1
    x0 = linear0 & L128_MASK
    y0 = (linear0 >> L128_SHIFT) & L128_MASK
    ensemble0 = linear0 >> L128_PLANE_SHIFT
    xm1 = (x0 - 1) & L128_MASK
    ym1 = (y0 - 1) & L128_MASK
    row = (ensemble0 << L128_PLANE_SHIFT) + (y0 << L128_SHIFT)
    row_m1 = (ensemble0 << L128_PLANE_SHIFT) + (ym1 << L128_SHIFT)
    x_here = linear
    x_prev = row + xm1 + 1
    y_prev = row_m1 + x0 + 1
    return (
        xi_xy[x_here] - xi_xy[x_prev] +
        xi_xy[nfield + x_here] - xi_xy[nfield + y_prev]
    )
end

function packed_noise_scaled128_kernel!(out, xi_xy, scale)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    nfield = length(out)
    @inbounds for linear in id:stride:nfield
        out[linear] = scale * packed_conserved_noise_at128(xi_xy, linear, nfield)
    end
    return nothing
end

function packed_noise_bridge128_kernel!(out, xi_xy, scale, q, Wh)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    nfield = length(out)
    @inbounds for linear in id:stride:nfield
        out[linear] =
            scale * packed_conserved_noise_at128(xi_xy, linear, nfield) + q * Wh[linear]
    end
    return nothing
end

function launch_packed_scaled128!(
    out, xi_xy, scale;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_packed128(out, xi_xy)
    @cuda threads=threads blocks=blocks packed_noise_scaled128_kernel!(out, xi_xy, scale)
    return nothing
end

function launch_packed_bridge128!(
    out, xi_xy, scale, q, Wh;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_packed128(out, xi_xy)
    @assert size(Wh) == size(out)
    @cuda threads=threads blocks=blocks packed_noise_bridge128_kernel!(
        out, xi_xy, scale, q, Wh
    )
    return nothing
end

struct PackedNoiseCache{A, R}
    xi_xy::A
    rng::R
    synchronize_callbacks::Bool
end

struct PackedNoiseDist{C}
    cache::C
end

struct PackedNoiseBridge{C}
    cache::C
end

function (f::PackedNoiseDist)(rand_vec, W, dt, u, p, t, ignored_rng)
    cache = f.cache
    # Use Random.randn! directly so the selected RNG is honored.
    Random.randn!(cache.rng, cache.xi_xy)
    launch_packed_scaled128!(rand_vec, cache.xi_xy, sqrt(abs(dt)))
    cache.synchronize_callbacks && CUDA.synchronize()
    return nothing
end

function (f::PackedNoiseBridge)(rand_vec, W, W0, Wh, q, h, u, p, t, ignored_rng)
    cache = f.cache
    Random.randn!(cache.rng, cache.xi_xy)
    scale = sqrt((1 - q) * q * abs(h))
    launch_packed_bridge128!(rand_vec, cache.xi_xy, scale, q, Wh)
    cache.synchronize_callbacks && CUDA.synchronize()
    return nothing
end

"""
    packed_noise_process128(t0, template; rng_kind=:library_default,
                            rng_ordering=nothing, seed=20260718,
                            synchronize_callbacks=false)

NoiseProcess candidate that generates both vector-noise components in one
contiguous `randn!` call and fuses divergence with the scale/bridge operation.
It is a drop-in `noise_builder` for `common.jl`.
"""
function packed_noise_process128(
    t0,
    template;
    rng_kind=:library_default,
    rng_ordering=nothing,
    seed=20260718,
    synchronize_callbacks=false,
    rswm=nothing,
)
    check_field128(template)
    xi_xy = similar(template, (size(template)..., 2))
    rng = make_gpu_rng(rng_kind, seed; ordering=rng_ordering)
    cache = PackedNoiseCache(xi_xy, rng, synchronize_callbacks)
    kwargs = isnothing(rswm) ?
        (; save_everystep=false) :
        (; save_everystep=false, rswm=rswm)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        PackedNoiseDist(cache),
        PackedNoiseBridge(cache);
        kwargs...,
    )
end

# -----------------------------------------------------------------------------
# Two independent RNG streams.  This can overlap the xi_x and xi_y fills, but
# the extra events are essential: a producer stream must not overwrite xi while
# the main stream's preceding divergence kernel is still reading it.
# -----------------------------------------------------------------------------

@inline function split_conserved_noise_at128(xi_x, xi_y, linear)
    linear0 = linear - 1
    x0 = linear0 & L128_MASK
    y0 = (linear0 >> L128_SHIFT) & L128_MASK
    ensemble0 = linear0 >> L128_PLANE_SHIFT
    xm1 = (x0 - 1) & L128_MASK
    ym1 = (y0 - 1) & L128_MASK
    row = (ensemble0 << L128_PLANE_SHIFT) + (y0 << L128_SHIFT)
    row_m1 = (ensemble0 << L128_PLANE_SHIFT) + (ym1 << L128_SHIFT)
    return (
        xi_x[linear] - xi_x[row + xm1 + 1] +
        xi_y[linear] - xi_y[row_m1 + x0 + 1]
    )
end

function split_noise_scaled128_kernel!(out, xi_x, xi_y, scale)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:length(out)
        out[linear] = scale * split_conserved_noise_at128(xi_x, xi_y, linear)
    end
    return nothing
end

function split_noise_bridge128_kernel!(out, xi_x, xi_y, scale, q, Wh)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for linear in id:stride:length(out)
        out[linear] =
            scale * split_conserved_noise_at128(xi_x, xi_y, linear) + q * Wh[linear]
    end
    return nothing
end

function launch_split_scaled128!(
    out, xi_x, xi_y, scale;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_field128(out)
    @assert size(xi_x) == size(out) && size(xi_y) == size(out)
    @cuda threads=threads blocks=blocks split_noise_scaled128_kernel!(
        out, xi_x, xi_y, scale
    )
    return nothing
end

function launch_split_bridge128!(
    out, xi_x, xi_y, scale, q, Wh;
    threads=640,
    blocks=min(cld(length(out), threads), 264),
)
    check_field128(out)
    @assert size(xi_x) == size(out) && size(xi_y) == size(out) && size(Wh) == size(out)
    @cuda threads=threads blocks=blocks split_noise_bridge128_kernel!(
        out, xi_x, xi_y, scale, q, Wh
    )
    return nothing
end

struct DualStreamNoiseCache{A, RX, RY, S, E}
    xi_x::A
    xi_y::A
    rng_x::RX
    rng_y::RY
    stream_x::S
    stream_y::S
    ready_x::E
    ready_y::E
    consumed::E
    has_consumed::Base.RefValue{Bool}
    synchronize_callbacks::Bool
end

struct DualStreamNoiseDist{C}
    cache::C
end

struct DualStreamNoiseBridge{C}
    cache::C
end

"""
    fill_two_streams!(cache)

Asynchronously fill `xi_x` and `xi_y` on separate non-blocking streams and
make the calling task's stream wait for both.  The wait is device-side; this
function does not synchronize the host.
"""
function fill_two_streams!(cache::DualStreamNoiseCache)
    main_stream = CUDA.stream()

    if cache.has_consumed[]
        CUDA.wait(cache.consumed, cache.stream_x)
        CUDA.wait(cache.consumed, cache.stream_y)
    end

    CUDA.stream!(cache.stream_x) do
        Random.randn!(cache.rng_x, cache.xi_x)
        CUDA.record(cache.ready_x)
    end
    CUDA.stream!(cache.stream_y) do
        Random.randn!(cache.rng_y, cache.xi_y)
        CUDA.record(cache.ready_y)
    end

    CUDA.wait(cache.ready_x, main_stream)
    CUDA.wait(cache.ready_y, main_stream)
    return nothing
end

@inline function mark_consumed!(cache::DualStreamNoiseCache)
    CUDA.record(cache.consumed, CUDA.stream())
    cache.has_consumed[] = true
    return nothing
end

function (f::DualStreamNoiseDist)(rand_vec, W, dt, u, p, t, ignored_rng)
    cache = f.cache
    fill_two_streams!(cache)
    launch_split_scaled128!(rand_vec, cache.xi_x, cache.xi_y, sqrt(abs(dt)))
    mark_consumed!(cache)
    cache.synchronize_callbacks && CUDA.synchronize()
    return nothing
end

function (f::DualStreamNoiseBridge)(rand_vec, W, W0, Wh, q, h, u, p, t, ignored_rng)
    cache = f.cache
    fill_two_streams!(cache)
    scale = sqrt((1 - q) * q * abs(h))
    launch_split_bridge128!(rand_vec, cache.xi_x, cache.xi_y, scale, q, Wh)
    mark_consumed!(cache)
    cache.synchronize_callbacks && CUDA.synchronize()
    return nothing
end

"""
    dual_stream_noise_process128(t0, template; rng_kind=:library_default,
                                 rng_ordering=nothing, seed=20260718,
                                 synchronize_callbacks=false)

NoiseProcess candidate with independent x/y RNG generators and streams.  This
changes the random-number stream relative to a monolithic solve, so validate
statistics/observables rather than expecting bitwise equality.
"""
function dual_stream_noise_process128(
    t0,
    template;
    rng_kind=:library_default,
    rng_ordering=nothing,
    seed=20260718,
    synchronize_callbacks=false,
    rswm=nothing,
)
    check_field128(template)
    xi_x, xi_y = similar(template), similar(template)
    stream_x = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    stream_y = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    rng_x = make_gpu_rng(rng_kind, seed; stream=stream_x, ordering=rng_ordering)
    rng_y = make_gpu_rng(
        rng_kind, seed + 1; stream=stream_y, ordering=rng_ordering
    )
    event_flags = CUDA.EVENT_DISABLE_TIMING
    cache = DualStreamNoiseCache(
        xi_x,
        xi_y,
        rng_x,
        rng_y,
        stream_x,
        stream_y,
        CUDA.CuEvent(event_flags),
        CUDA.CuEvent(event_flags),
        CUDA.CuEvent(event_flags),
        Ref(false),
        synchronize_callbacks,
    )
    kwargs = isnothing(rswm) ?
        (; save_everystep=false) :
        (; save_everystep=false, rswm=rswm)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        DualStreamNoiseDist(cache),
        DualStreamNoiseBridge(cache);
        kwargs...,
    )
end

# -----------------------------------------------------------------------------
# Double-buffered packed RNG prefetch.  Raw normals do not depend on u, dt, q,
# or h, so the next callback's packed field can be generated while the main
# stream runs the current divergence and later SOSRA kernels.  A consumed event
# prevents the producer from overwriting a buffer still read by the main stream.
# -----------------------------------------------------------------------------

struct PrefetchPackedNoiseCache{A, R, S, E}
    xi_xy::NTuple{2, A}
    rng::R
    producer_stream::S
    ready::NTuple{2, E}
    consumed::NTuple{2, E}
    next_slot::Base.RefValue{Int}
    synchronize_callbacks::Bool
end

struct PrefetchPackedNoiseDist{C}
    cache::C
end

struct PrefetchPackedNoiseBridge{C}
    cache::C
end

function prime_prefetch!(cache::PrefetchPackedNoiseCache)
    CUDA.stream!(cache.producer_stream) do
        for slot in 1:2
            Random.randn!(cache.rng, cache.xi_xy[slot])
            CUDA.record(cache.ready[slot])
        end
    end
    return nothing
end

@inline function acquire_prefetched(cache::PrefetchPackedNoiseCache)
    slot = cache.next_slot[]
    CUDA.wait(cache.ready[slot], CUDA.stream())
    return slot
end

function release_and_refill!(cache::PrefetchPackedNoiseCache, slot::Int)
    main_stream = CUDA.stream()
    CUDA.record(cache.consumed[slot], main_stream)
    CUDA.wait(cache.consumed[slot], cache.producer_stream)
    CUDA.stream!(cache.producer_stream) do
        Random.randn!(cache.rng, cache.xi_xy[slot])
        CUDA.record(cache.ready[slot])
    end
    cache.next_slot[] = 3 - slot
    return nothing
end

function (f::PrefetchPackedNoiseDist)(rand_vec, W, dt, u, p, t, ignored_rng)
    cache = f.cache
    slot = acquire_prefetched(cache)
    launch_packed_scaled128!(rand_vec, cache.xi_xy[slot], sqrt(abs(dt)))
    release_and_refill!(cache, slot)
    cache.synchronize_callbacks && CUDA.synchronize()
    return nothing
end

function (f::PrefetchPackedNoiseBridge)(
    rand_vec, W, W0, Wh, q, h, u, p, t, ignored_rng
)
    cache = f.cache
    slot = acquire_prefetched(cache)
    scale = sqrt((1 - q) * q * abs(h))
    launch_packed_bridge128!(rand_vec, cache.xi_xy[slot], scale, q, Wh)
    release_and_refill!(cache, slot)
    cache.synchronize_callbacks && CUDA.synchronize()
    return nothing
end

"""
    prefetch_packed_noise_process128(t0, template;
        rng_kind=:library_default, rng_ordering=nothing, seed=20260718,
        synchronize_callbacks=false)

Double-buffered version of `packed_noise_process128`. It queues the next raw
normal field on a producer stream while the main task continues the SOSRA
step. This is the most promising RNG/concurrency candidate for raising useful
overlap and power, at the cost of one additional packed field (2*length(u)
Float32 values).
"""
function prefetch_packed_noise_process128(
    t0,
    template;
    rng_kind=:library_default,
    rng_ordering=nothing,
    seed=20260718,
    synchronize_callbacks=false,
    rswm=nothing,
)
    check_field128(template)
    xi_xy = (
        similar(template, (size(template)..., 2)),
        similar(template, (size(template)..., 2)),
    )
    producer_stream = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    rng = make_gpu_rng(
        rng_kind, seed; stream=producer_stream, ordering=rng_ordering
    )
    event_flags = CUDA.EVENT_DISABLE_TIMING
    cache = PrefetchPackedNoiseCache(
        xi_xy,
        rng,
        producer_stream,
        (CUDA.CuEvent(event_flags), CUDA.CuEvent(event_flags)),
        (CUDA.CuEvent(event_flags), CUDA.CuEvent(event_flags)),
        Ref(1),
        synchronize_callbacks,
    )
    prime_prefetch!(cache)
    kwargs = isnothing(rswm) ?
        (; save_everystep=false) :
        (; save_everystep=false, rswm=rswm)
    return NoiseProcess{true}(
        t0,
        zero(template),
        zero(template),
        PrefetchPackedNoiseDist(cache),
        PrefetchPackedNoiseBridge(cache);
        kwargs...,
    )
end

# -----------------------------------------------------------------------------
# Independent-ensemble concurrency.  Time evolution within one trajectory is
# sequential, but ensemble shards can be solved on different task-local streams.
# The return value deliberately remains a vector of shard solutions to avoid a
# large extra concatenation/allocation.
# -----------------------------------------------------------------------------

function ensemble_ranges(nensemble::Integer, nshards::Integer)
    1 <= nshards <= nensemble ||
        throw(ArgumentError("require 1 <= nshards <= nensemble"))
    width, extra = divrem(nensemble, nshards)
    ranges = UnitRange{Int}[]
    first_index = 1
    for shard in 1:nshards
        count = width + (shard <= extra)
        push!(ranges, first_index:(first_index + count - 1))
        first_index += count
    end
    return ranges
end

"""
    run_ensemble_shards(solve_one, u0; nshards=2, seed=20260718)

Run independent ensemble slices concurrently. `solve_one(shard, shard_seed)`
must solve one dense CuArray shard and return its solution.  Each Julia task
uses an explicit non-blocking CUDA stream and a distinct seed.  Start with two
shards; four is worth testing, while more shards usually add launch and memory
pressure without useful overlap.
"""
function run_ensemble_shards(
    solve_one,
    u0;
    nshards::Integer=2,
    seed::Integer=20260718,
)
    ndims(u0) == 3 || throw(ArgumentError("u0 must be a 3D (L,L,Nensemble) array"))
    ranges = ensemble_ranges(size(u0, 3), nshards)
    tasks = map(enumerate(ranges)) do (shard_id, range)
        @async begin
            shard_stream = CUDA.CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
            CUDA.stream!(shard_stream) do
                shard_seed = seed + shard_id - 1
                # Seed all task-local CUDA RNG paths for builders that still use
                # CUDA.randn!/DiffEqNoiseProcess.wiener_randn! internally.
                CUDA.seed!(shard_seed)
                # `copy(view(...))` stays device-to-device for a CuArray. Calling
                # `CuArray` directly on a CuSubArray can take the generic
                # AbstractArray constructor and stage through a CPU Array.
                shard_copy = copy(@view u0[:, :, range])
                shard = shard_copy isa CUDA.AnyCuArray ?
                    shard_copy : CUDA.CuArray(shard_copy)
                result = solve_one(shard, shard_seed)
                CUDA.synchronize(shard_stream)
                return result
            end
        end
    end
    results = fetch.(tasks)
    # Also covers producer streams owned by a shard's prefetch noise process.
    CUDA.device_synchronize()
    return results
end

end # module H100RNGVariants
