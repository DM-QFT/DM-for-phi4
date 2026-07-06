"""
Export the diagonal momentum-space propagator G(k) (HMC + DM epochs) as 3-col
`.dat` files for Mathematica plotting.

Mirrors the calculation in `compare_dm_hmc_propagator.py`:
  diagonal modes (n, n) for n = 1..L/2
  k² = 8 sin²(π n / L),   G(k) = <|φ̃(n,n)|²> / V
  bootstrap mean / std with 200 resamples.

Output:
  <run_dir>/mathematica_data/
      G_k_diag_HMC.dat                   # k, G, dG
      G_k_diag_DM_<method>_epoch=XXXX.dat
      xi.dat                             # name  N  xi  G(k_min)  dG(k_min)

Usage (from 2Dphi4/):
  python analysis/export_diagonal_propagator_dat.py \
      --k 0.28 --L 128 --run_suffix _sigma640 --method em --sde_steps 2000
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

PROJ = Path(__file__).resolve().parent.parent


def diagonal_propagator(cfgs, n_boot=200, seed=0):
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V

    n_diag = np.arange(1, L // 2 + 1)
    g_per_cfg = pk[:, n_diag, n_diag]

    k2 = 8.0 * np.sin(np.pi * n_diag / L) ** 2
    k_lat = np.sqrt(k2)

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(n_diag)))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k2, G, n_low=5):
    if len(k2) < n_low:
        return float("nan")
    a, b = np.polyfit(k2[:n_low], 1.0 / G[:n_low], 1)
    if a <= 0 or b <= 0:
        return float("nan")
    return 1.0 / np.sqrt(b / a)


def load_dm(path):
    return np.load(path).transpose(2, 0, 1).astype(np.float64)


def load_hmc(path):
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs


def write_dat(path, k, G, dG):
    np.savetxt(path, np.column_stack([k, G, dG]),
               fmt=("%.10e", "%.10e", "%.10e"),
               header="k  G(k)  dG(k)   (diagonal modes n_x=n_y, n=1..L/2)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--run_suffix", default="_sigma640")
    p.add_argument("--method", choices=["em", "ode"], default="em")
    p.add_argument("--sde_steps", type=int, default=2000)
    p.add_argument("--ode_steps", type=int, default=400)
    p.add_argument("--epochs", default=None,
                   help="comma list; default = every epoch found for this method/steps")
    p.add_argument("--out_subdir", default="mathematica_data")
    args = p.parse_args()

    steps = args.sde_steps if args.method == "em" else args.ode_steps
    run_dir = PROJ / f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.run_suffix}"
    data_dir = run_dir / "data"
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    hmc_path = PROJ / f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"

    if args.epochs:
        eps = [e.strip() for e in args.epochs.split(",") if e.strip()]
    else:
        prefix = f"samples_{args.method}_steps{steps}_epoch="
        eps = sorted(
            f.name[len(prefix):-len(".npy")]
            for f in data_dir.glob(f"{prefix}*.npy")
        )

    print(f"run_dir : {run_dir}")
    print(f"out_dir : {out_dir}")
    print(f"hmc     : {hmc_path}")
    print(f"epochs  : {eps}")
    print()

    summary = []  # (name, N, xi, G[0], dG[0])

    hmc = load_hmc(str(hmc_path))
    k_lat, G_hmc, dG_hmc = diagonal_propagator(hmc)
    xi_hmc = fit_xi(k_lat ** 2, G_hmc)
    write_dat(out_dir / "G_k_diag_HMC.dat", k_lat, G_hmc, dG_hmc)
    summary.append(("HMC", hmc.shape[0], xi_hmc, G_hmc[0], dG_hmc[0]))
    print(f"{'HMC':>22}  N={hmc.shape[0]:5d}  xi={xi_hmc:7.2f}  "
          f"G(k_min)={G_hmc[0]:8.4f} +/- {dG_hmc[0]:6.4f}")
    del hmc

    for ep in eps:
        npy = data_dir / f"samples_{args.method}_steps{steps}_epoch={ep}.npy"
        if not npy.is_file():
            print(f"  [skip] {npy.name}")
            continue
        x = load_dm(npy)
        _, G, dG = diagonal_propagator(x)
        xi = fit_xi(k_lat ** 2, G)
        out_name = f"G_k_diag_DM_{args.method}_epoch={ep}.dat"
        write_dat(out_dir / out_name, k_lat, G, dG)
        summary.append((f"DM_ep={ep}", x.shape[0], xi, G[0], dG[0]))
        print(f"{'DM ep=' + ep:>22}  N={x.shape[0]:5d}  xi={xi:7.2f}  "
              f"G(k_min)={G[0]:8.4f} +/- {dG[0]:6.4f}")
        del x

    with open(out_dir / "xi.dat", "w") as f:
        f.write("# name           N      xi      G(k_min)     dG(k_min)\n")
        for name, N, xi, g0, dg0 in summary:
            f.write(f"{name:<14s}  {N:6d}  {xi:7.3f}  {g0:12.6e}  {dg0:12.6e}\n")

    print(f"\nwrote {len(summary)} files + xi.dat to {out_dir}")


if __name__ == "__main__":
    main()
