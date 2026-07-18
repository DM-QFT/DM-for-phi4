# Register-cap and launch-shape sweep

Target: `L=128`, 1024 ensembles, `Float32`, one drift evaluation after warm-up.
Every output was compared with the direct-reference drift and had zero maximum
absolute error. Raw output is in `maxregs_raw.log`; reproduce with
`benchmark_maxregs.jl`.

## Production Cartesian-index kernel

| Threads x blocks | maxregs | Median drift time |
|---:|---:|---:|
| 896 x 132 | natural (72 registers) | 0.308034 ms |
| 1024 x 264 | 64 | **0.306466 ms** |
| 1024 x 264 | 56 | 0.346978 ms |
| 1024 x 264 | 48 | 0.355170 ms |
| 1024 x 264 | 40 | 0.494435 ms |
| 1024 x 264 | 32 | 1.080358 ms |

The uncapped kernel cannot launch 1024 threads because `72*1024=73728`
registers exceeds the 65536-register block limit. Capping at 64 permits that
launch but improves the existing 896-thread result by only about 0.5%, within
the scale of run-to-run variation. Lower caps spill increasingly heavily;
forcing 32 registers makes the kernel about 3.5 times slower.

## L=128 direct bit-mask kernel

| Threads x blocks | maxregs | Median drift time |
|---:|---:|---:|
| 1024 x 528 | natural (32 registers) | 0.096096 ms |
| 1024 x 528 | 32 | **0.094593 ms** |
| 1024 x 528 | 28 | 0.095425 ms |
| 1024 x 528 | 24 | 0.105664 ms |

This kernel already uses only 32 registers per thread. An explicit cap of 32
is effectively neutral; 28 is also within about 1%, while 24 is slower. The
direct kernel remains about 3.25 times faster than the best production-kernel
register/launch combination, so indexing simplification—not register forcing—
is the material optimization.

Conclusion: do not apply an aggressive `maxregs` cap. Use the natural
32-register direct/fused kernels; register spilling cannot manufacture useful
H100 utilization.
