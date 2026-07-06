"""Merge 8 PC-scan shards for one corrector setting into a 512-sample .npy."""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pc", type=int, required=True)
ap.add_argument("--k", type=str, default="0.1923")
ap.add_argument("--n_shards", type=int, default=8)
ap.add_argument("--tag", type=str, default="pcscan",
                help="filename stem: samples_{tag}_pc{PC}_shard{r}.npy")
args = ap.parse_args()

base = Path(f"runs/phi4_3d_Lmulti4-8-16-32_k{args.k}_l0.9_ncsnpp/data_crossL")
parts = []
for r in range(args.n_shards):
    f = base / f"samples_{args.tag}_pc{args.pc}_shard{r}.npy"
    if not f.exists():
        print(f"[warn] missing {f}")
        continue
    parts.append(np.load(f))                      # (L,L,L,64)
if not parts:
    raise SystemExit(f"no shards for pc={args.pc}")
merged = np.concatenate(parts, axis=-1)
out = base / f"samples_{args.tag}_pc{args.pc}_merged.npy"
np.save(out, merged)
print(f"pc={args.pc}: merged {len(parts)} shards -> {merged.shape}  saved {out}")
