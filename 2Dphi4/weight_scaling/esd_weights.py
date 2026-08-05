"""Per-layer weight spectra (ESD) and entry statistics across checkpoints.

Conv (Co,Ci,3,3) matricized to (Co, Ci*9). alpha_hill = CCDF tail index of the
eigenvalues s^2 (Martin-Mahoney ESD exponent ~ alpha_hill + 1).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def hill(lam, frac=0.25):
    lam = np.sort(lam)[::-1]
    k = max(4, int(len(lam) * frac))
    if k >= len(lam):
        return np.nan
    return k / np.sum(np.log(lam[:k] / lam[k]))


def layer_rows(sd, run, epoch, ema, npz_store):
    rows = []
    for name, w in sd.items():
        if w.ndim not in (2, 4) or "norm" in name:
            continue
        W = w.reshape(w.shape[0], -1).numpy().astype(np.float64)
        sv = np.linalg.svd(W, compute_uv=False)
        lam = sv**2
        entries = W.ravel()
        m2 = np.mean((entries - entries.mean()) ** 2)
        kurt = np.mean((entries - entries.mean()) ** 4) / m2**2 - 3.0
        rows.append((run, epoch, ema, name, W.shape[0], W.shape[1],
                     np.sum(lam), lam[0], np.sum(lam) / lam[0],
                     hill(lam), entries.std(), kurt))
        npz_store[f"{run}/{epoch}/{ema}/{name}"] = sv.astype(np.float32)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out_prefix", default="results/esd")
    args = p.parse_args()

    summary, npz_store = [], {}
    for run_dir in args.runs:
        run_dir = Path(run_dir)
        run = re.search(r"k(0\.\d+)", run_dir.name).group(1)
        for ckpt in sorted(run_dir.glob("models/epoch=*.ckpt")):
            epoch = int(re.search(r"epoch=(\d+)", ckpt.name).group(1))
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = {k.replace("score_model._orig_mod.", "").replace(
                "score_model.", ""): v for k, v in ck["state_dict"].items()}
            summary += layer_rows(sd, run, epoch, 0, npz_store)
            shadow = ck["ema_state_dict"]["shadow_params"]
            assert len(shadow) == len(sd)
            ema_sd = {k: s for (k, v), s in zip(sd.items(), shadow)
                      if s.shape == v.shape}
            assert len(ema_sd) == len(sd)  # order mismatch would break shapes
            summary += layer_rows(ema_sd, run, epoch, 1, npz_store)
            print(f"{run} epoch={epoch}")

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = ("kappa\tepoch\tema\tlayer\tnrow\tncol\tfrob2\tspec2\tstable_rank\t"
           "alpha_hill\tentry_std\tentry_kurt")
    with open(f"{out}_summary.tsv", "w") as f:
        f.write("# " + hdr + "\n")
        for r in summary:
            f.write("\t".join(str(x) for x in r) + "\n")
    np.savez_compressed(f"{out}_svals.npz", **npz_store)
    print(f"wrote {out}_summary.tsv ({len(summary)} rows), {out}_svals.npz")


if __name__ == "__main__":
    main()
