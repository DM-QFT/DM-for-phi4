"""Generate training/test pairs. Usage: python solver/make_pairs.py reaction|brownian|single
reaction: A<->2A, brownian: same initial states without reactions, single: one free particle per box.
Initial states cycle through uniform, sinusoidal modulation and clusters/voids (kind 0,1,2)."""
import os
import sys
from multiprocessing import Pool
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from config import LAM, GAM
from simulate import trajectory, future, write_pairs, read_pairs

# training sizes L = 8, 10, ..., 32 with trajectories ~ 1/L^2 so every size holds the same number of particles. L = 48, 64 are extrapolation only
L_TRAIN = list(range(8, 33, 2))
SIZES = [(L, round(131072 / L ** 2), round(32768 / L ** 2)) for L in L_TRAIN] + [(48, 0, 64), (64, 0, 64)]
K = 4                                 # pairs per trajectory
N_MULTI_INIT, N_MULTI_FUT = 16, 64    # structured test initial states with many independent futures, for conditional statistics
OUT = os.path.join(PROJECT_DIR, "trainingdata")

tag = sys.argv[1]
lam, gam = {"reaction": (LAM, GAM), "brownian": (0.0, 0.0), "single": (0.0, 0.0)}[tag]
kind = (lambda s: 3) if tag == "single" else (lambda s: s % 3)
os.makedirs(OUT, exist_ok=True)
pool = Pool(64)
seed0 = 0
for L, n_train, n_test in SIZES:
    for split, n in (("train", n_train), ("test", n_test)):
        if n == 0:
            continue
        snaps = pool.map(trajectory, [(L, seed0 + s, lam, gam, K, kind(s)) for s in range(n)])
        seed0 += n
        x0s = [sn[k] for sn in snaps for k in range(K)]
        x10s = [sn[k + 1] for sn in snaps for k in range(K)]
        write_pairs(f"{OUT}/{tag}_L{L}_{split}.h5", x0s, x10s, L, [t for t in range(n) for _ in range(K)],
                    [k for _ in range(n) for k in range(K)], [kind(t) for t in range(n) for _ in range(K)])
for L in (16, 32):
    x0s_test, _, _, _, _, init = read_pairs(f"{OUT}/{tag}_L{L}_test.h5")
    inits = x0s_test[0::K][:N_MULTI_INIT]   # the initial states themselves, where the structure is strongest
    jobs = [(x0, L, 10_000_000 + 1000 * a + b, lam, gam) for a, x0 in enumerate(inits) for b in range(N_MULTI_FUT)]
    x10s = pool.map(future, jobs)
    write_pairs(f"{OUT}/{tag}_L{L}_multi.h5", [x0 for x0 in inits for _ in range(N_MULTI_FUT)], x10s, L,
                [a for a in range(len(inits)) for _ in range(N_MULTI_FUT)], [0] * (len(inits) * N_MULTI_FUT),
                [init[0::K][a] for a in range(len(inits)) for _ in range(N_MULTI_FUT)])
