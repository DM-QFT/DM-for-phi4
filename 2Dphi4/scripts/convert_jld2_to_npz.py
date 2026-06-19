#!/usr/bin/env python
"""Convert 2Dphi4 training .jld2 (JLD2/HDF5) configs to uncompressed .npz.

Mirrors data.py's read path: cfgs = np.array(f["cfgs"]).  Julia stores cfgs
column-major as (L, L, n_samples); h5py reverses the axes so it comes out as
(n_samples, L, L) = (N, L, L), exactly what FieldDataModule expects.

cfgs are kept at their native float64 (lossless).  Any scalar metadata stored
alongside (kappa, lambda, N, D, n_samples, abs_mag, ...) is carried over too,
so each .npz is self-describing.

Output: one .npz per .jld2, same basename, under trainingdata/npz/.
"""
import os
import glob
import numpy as np
import h5py

SRC = "/work/gw97/w97000/project/DM/2Dphi4/trainingdata"
DST = os.path.join(SRC, "npz")


def load_cfgs(obj):
    """Read cfgs as a single ndarray.

    Handles both layouts: a single HDF5 dataset (these files), and the JLD2
    streaming format where cfgs is a group of per-config datasets (cfgs/000001
    ...) — stacked along a new leading axis for that case.
    """
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj)
    names = [k for k in obj.keys() if not k.startswith("_")]

    def keyfn(k):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    names.sort(key=keyfn)
    return np.stack([np.asarray(obj[k]) for k in names], axis=0)


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.jld2")))
    print(f"Found {len(files)} .jld2 files in {SRC}")

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(DST, base + ".npz")
        if os.path.exists(out):
            print(f"[skip] {os.path.basename(out)} already exists")
            continue

        with h5py.File(path, "r") as f:
            top = [k for k in f.keys() if not k.startswith("_")]
            print(f"\n=== {os.path.basename(path)}")
            print(f"    top-level keys: {top}")

            payload = {}
            cfgs = np.ascontiguousarray(load_cfgs(f["cfgs"]), dtype=np.float64)
            payload["cfgs"] = cfgs
            print(f"    cfgs -> shape={cfgs.shape} dtype={cfgs.dtype}")

            for k in top:
                if k == "cfgs":
                    continue
                obj = f[k]
                if not isinstance(obj, h5py.Dataset):
                    print(f"    [skip meta] {k} (not a dataset)")
                    continue
                try:
                    payload[k] = np.asarray(obj)
                    print(f"    {k} -> {payload[k].shape or payload[k].item()}")
                except Exception as e:  # noqa: BLE001 - just skip unreadable meta
                    print(f"    [warn] could not read {k}: {e}")

        np.savez(out, **payload)
        print(f"[ok] wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")

    print("\nDONE")


if __name__ == "__main__":
    main()
