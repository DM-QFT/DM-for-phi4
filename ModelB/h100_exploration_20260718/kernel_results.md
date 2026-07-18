# Kernel and library screening

All kernel comparisons use `L=128`, 1024 ensembles, `Float32`, synchronized
warm medians. The direct/shared/fused candidates were checked against the
unchanged production kernels on eight independent inputs; all eight drift and
noise comparisons had zero maximum elementwise error.

## Drift stencil

| Implementation | Best launch | Median |
|---|---:|---:|
| production CartesianIndices/periodic branches | 896 x 132 | 0.306658 ms |
| direct L128 bit-mask linear addressing | 1024 x 528 | **0.093888 ms** |
| shared tile 32 x 8 | `(32,8)` tile | 0.104001 ms |
| shared tile 16 x 16 | `(16,16)` tile | 0.129632 ms |
| exact-symbol cuFFT spectral finite-difference drift | cuFFT plans | 0.781652 ms |

Direct addressing is about 3.27 times faster than production. Shared memory
does not win because neighboring x loads are already coalesced/cache-friendly
and the halo plus barriers cost more than the saved global reads. The cuFFT
operator agrees to `4.58e-5` maximum and `4.25e-6` RMS (`~3e-7` relative RMS)
but is about 2.56 times slower than production and 8.3 times slower than the
direct kernel.

## Conserved noise

| Operation | Median |
|---|---:|
| production divergence only | 0.146113 ms |
| production divergence plus separate scale broadcast | 0.298673 ms |
| direct divergence fused with scale | **0.075489 ms** |

The fused noise operation is about 3.96 times faster than the original
divergence-plus-scale pair. Its per-ensemble zero mode is unchanged up to
Float32 reduction roundoff.

## End-to-end contribution before RNG/SOSRA specialization

| General SciML path | Warm median `t=2` |
|---|---:|
| production sync + RSwM3 | 5.164691 s |
| production callbacks async + RSwM3 | 4.971002 s |
| production callbacks async + RSwM1 | 4.308709 s |
| direct drift + fused noise + RSwM1 | **3.271561 s** |
| shared drift + fused noise + RSwM1 | 3.297306 s |

The ten-step same-seed validation is in `trajectory_validation.md`: all these
fixed-step changes were exactly equal to production element by element. The
later formal sustained-power timings differ slightly because they use a full
same-shape prewarm and a longer repeated marker window; the ranking is
unchanged.
