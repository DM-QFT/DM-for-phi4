"""
Merge per-node MALA production shards into one ensemble per kappa.

Concatenates data_crossL_mala/prod512f_2000_shard{r}/final.npy (r=0..7)
into data_crossL_mala/prod512f_2000/final.npy, the location the analysis
pipeline (--mala_tag prod512f_2000) expects. Also reports per-shard
acceptance from traces.npz.

Run from 3Dphi4/:
    python scripts/merge_mala_shards.py
"""

import argparse
from pathlib import Path
import numpy as np

ALL_SPECS = {"0.18": "_sigma200", "0.1923": "", "0.2": "_sigma400"}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", type=str, default="prod512f_2000")
ap.add_argument("--kappas", type=str, nargs="+",
                default=["0.18", "0.1923", "0.2"])
ap.add_argument("--n_shards", type=int, default=8)
args = ap.parse_args()

SPECS = [(k, ALL_SPECS[k]) for k in args.kappas]
N_SHARDS = args.n_shards
TAG = args.tag

for k, suf in SPECS:
    rd = Path(f"runs/phi4_3d_Lmulti4-8-16-32_k{k}_l0.9_ncsnpp{suf}")
    base = rd / "data_crossL_mala"
    parts, accs = [], []
    for r in range(N_SHARDS):
        d = base / f"{TAG}_shard{r}"
        f = d / "final.npy"
        if not f.exists():
            print(f"[warn] kappa={k}: missing shard {r} ({f})")
            continue
        parts.append(np.load(f))                       # (L,L,L,n_r)
        tr = np.load(d / "traces.npz")
        accs.append(float(tr["accept_total"].mean() / tr["n_done"]))
    if not parts:
        print(f"[skip] kappa={k}: no shards found")
        continue
    merged = np.concatenate(parts, axis=-1)
    out_dir = base / TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "final.npy", merged)
    print(f"kappa={k}: merged {len(parts)} shards -> {merged.shape}  "
          f"acc(per-shard)={np.round(accs, 3).tolist()}")
    print(f"  saved {out_dir / 'final.npy'}")
