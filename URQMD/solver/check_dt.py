"""Time-step convergence of the simulator from uniform states at L=24: events per particle, std(N10-N0), propagator G(k) and g01(0)."""
import os
import sys
import numpy as np
from multiprocessing import Pool
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from config import D, LAM, GAM, DT_PAIR
from simulate import initial_state, evolve
from observables import propagator, cross_correlation

L = 24
N_TRAJ = 192
DTS = [0.5, 0.2, 0.1, 0.05, 0.02]
OUT = os.path.join(PROJECT_DIR, "results", "solver")


def run(args):
    seed, dt = args
    rng = np.random.default_rng(seed)
    x0 = initial_state(L, 0, rng)
    x10, nm, nb = evolve(x0, L, DT_PAIR, dt, rng, LAM, GAM)
    return x0, x10, nm / len(x0), nb / len(x0)


os.makedirs(OUT, exist_ok=True)
pool = Pool(64)
rows = []
k_ref = 2 * np.pi * np.sqrt(np.array([1, 2, 4])) / L
print(f"reference: branchings per particle {LAM * DT_PAIR:.3f}, G(k) {np.round(np.exp(-(LAM + D * k_ref ** 2) * DT_PAIR), 3)} at |n|^2 = 1, 2, 4")
for dt in DTS:
    res = pool.map(run, [(s, dt) for s in range(N_TRAJ)])
    x0s, x10s = [r[0] for r in res], [r[1] for r in res]
    merges, branches = np.array([r[2] for r in res]), np.array([r[3] for r in res])
    dn = np.array([len(b) - len(a) for a, b in zip(x0s, x10s)])
    _, G = propagator(x0s, x10s, L, n_max=2)
    _, g01 = cross_correlation(x0s, x10s, L, rho_max=2.0, n_bins=4)
    rows.append((dt, merges.mean(), branches.mean(), dn.std(), G[0], G[1], G[3], g01[0]))
    print(f"dt {dt:<5} merges/N0 {merges.mean():.4f} branch/N0 {branches.mean():.4f} std(N10-N0) {dn.std():5.1f}  "
          f"G {G[0]:.3f} {G[1]:.3f} {G[3]:.3f}  g01(rho<0.5) {g01[0]:.3f}", flush=True)
np.savetxt(f"{OUT}/dt_check_L{L}.tsv", np.array(rows), header="dt merges_per_N0 branch_per_N0 std_dN G1 G2 G4 g01_0", fmt="%.6g", delimiter="\t")
