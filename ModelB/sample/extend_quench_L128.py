"""把 L=128 淬火链从现有 t=1000 终态续跑到 t=5000（ML4PS Fig.3a 需要 t=2000、5000）。"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sample_modelB_chain as chain

RUN = Path(__file__).resolve().parents[1] / "runs/modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss"
OLD = RUN / "samples/quench_chain_ep10000_L128_N32_t1000.h5"
T0 = RUN / "samples/quench_t1000_L128_N32.npz"
OUT = RUN / "samples/quench_chain_ep10000_L128_N32_t1010-5000.h5"

with h5py.File(OLD) as f:
    assert f["times"][-1] == 1000.0
    np.savez(T0, phi_gen=f["phi_gen"][-1])
    seed0 = int(f["seeds"][-1]) + 1  # 旧链 seed = 2026082710+k，续用不重叠

chain.TIMES = [1000.0 + 10.0 * (k + 1) for k in range(400)]
chain.SEEDS = [seed0 + k for k in range(400)]
chain.BATCH = 32
chain.NUM_STEPS = 2000
chain.SCHEDULE = "log"
chain.main([
    "--checkpoint", str(RUN / "models/epoch=10000.ckpt"),
    "--input", str(T0),
    "--output", str(OUT),
    "--compile_mode", "default",  # reduce-overhead 在 L=128 有 interpolate bug，旧链也是 default
])
