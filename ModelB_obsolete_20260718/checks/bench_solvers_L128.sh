#!/bin/zsh
# Systematic solver benchmark for Model B at L=128.
# Round 1 (equilibrium accuracy vs cost): T=8, t=600, <phi^2> vs MC reference.
# Round 2 (production quench): T=1, t=200.
cd /work/gw97/w97000/project/DM/ModelB/checks
R() { julia -O2 modelB_runner.jl "$@" ; }

# --- Round 1: T=8, L=128, t=600 ---
R b128_EMref_    128 600 8.0 0.001  EM        0.01 # fine-dt reference
R b128_EM25_     128 600 8.0 0.0025 EM        0.01
R b128_EM50_     128 600 8.0 0.005  EM        0.01
R b128_EM100_    128 600 8.0 0.01   EM        0.01
R b128_EM200_    128 600 8.0 0.02   EM        0.01 # stability probe
R b128_EMZ50_    128 600 8.0 0.005  EMZ       0.01 # Z-overhead probe
R b128_Heun50_   128 600 8.0 0.005  EulerHeun 0.01
R b128_Heun100_  128 600 8.0 0.01   EulerHeun 0.01
R b128_LambaEM_  128 600 8.0 0.005  LambaEM   0.05
R b128_SOSRA5_   128 600 8.0 0.001  SOSRA     0.5
R b128_SOSRA2_   128 600 8.0 0.001  SOSRA     0.2
R b128_SRA1_     128 600 8.0 0.001  SRA1      0.5
R b128_SRA3_     128 600 8.0 0.001  SRA3      0.5

# --- Round 2: T=1 quench, L=128, t=200 ---
R q128_EM50_     128 200 1.0 0.005  EM        0.01
R q128_Heun100_  128 200 1.0 0.01   EulerHeun 0.01
R q128_SOSRA5_   128 200 1.0 0.001  SOSRA     0.5
echo BENCH_DONE
