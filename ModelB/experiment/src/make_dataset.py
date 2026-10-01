"""Parse the Zenodo 13908799 colloid tracks into data/trajectories.h5 with per-file fringe parameters."""

from pathlib import Path

import h5py
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
DATA = EXP_DIR / "data"
PIXEL, DT_S = 0.24, 0.04            # um per pixel, s per frame (instruction.pdf)
FOV = (1280 * PIXEL, 1024 * PIXEL)   # Mako U-130B sensor
WINDOW = 15000                       # training window = first 10 min
LAM_GRID = np.arange(4.100, 4.1801, 0.001)
TILT_GRID = np.arange(-0.0010, 0.00101, 0.0001)


def read_file(path):
    a = np.fromfile(path, sep=" ").reshape(-1, 4)
    a = a[np.lexsort((a[:, 2], a[:, 3]))]
    n_id, T = int(a[-1, 3]), int(a[:, 2].max())
    assert a.shape[0] == n_id * T
    assert (a[:, 2].reshape(n_id, T) == np.arange(1, T + 1)).all()
    return (a[:, :2] * PIXEL).astype(np.float32).reshape(n_id, T, 2)


def fit_fringe(pos):
    """Period, tilt and offset of the light-field fringes from the training window."""
    x = pos[:, :WINDOW:5, 0].ravel().astype(np.float64)
    y = pos[:, :WINDOW:5, 1].ravel().astype(np.float64)
    best = (0.0,)
    for lam in LAM_GRID:
        Z = np.exp(2j * np.pi * (x[None] / lam - TILT_GRID[:, None] * y[None])).mean(1)
        j = np.argmax(np.abs(Z))
        if abs(Z[j]) > best[0]:
            best = (abs(Z[j]), lam, TILT_GRID[j], Z[j])
    Zabs, lam, tilt, Z = best
    assert Zabs > 0.4, Zabs
    # density maximum (well bottom) at u = (x - x0)/lam - tilt*y = 0 mod 1
    x0 = lam * np.angle(Z) / (2 * np.pi)
    return lam, tilt, x0, Zabs


def main():
    files = sorted(DATA.glob("*.dat"))
    groups = {"eq": [f for f in files if "speed000um" in f.name],
              "driven": [f for f in files if "speed0053um" in f.name]}
    with h5py.File(DATA / "trajectories.h5", "w") as h:
        h.attrs.update(pixel_um=PIXEL, dt_s=DT_S, window_frames=WINDOW, fov_um=FOV,
                       reflect="x -> fov_x - x for driven b files")
        for group, paths in groups.items():
            print(f"== {group}: {len(paths)} files")
            print("file   n_id     T  lam[um]  tilt[cyc/um]  x0[um]   |Z|   D1[um^2/s]  <dx>(10s)[um]  dir")
            pos, file_index, params = [], [], []
            for k, path in enumerate(paths):
                p = read_file(path)
                b_file = path.name.split("_speed")[0][-1] == "b"
                if b_file:
                    p[..., 0] = FOV[0] - p[..., 0]
                direction = 0 if group == "eq" else (1 if b_file else -1)
                lam, tilt, x0, Zabs = fit_fringe(p)
                D1 = np.diff(p[:, :WINDOW], axis=1).var(axis=(0, 1)).mean() / (2 * DT_S)
                dx10 = (p[:, 250:WINDOW, 0] - p[:, :WINDOW - 250, 0]).mean()
                print(f"{path.name[:6]:6s} {p.shape[0]:4d} {p.shape[1]:6d}  {lam:.4f}  {tilt:+.4f}"
                      f"      {x0:+.3f}  {Zabs:.3f}  {D1:.4f}      {dx10:+.4f}      {direction:+d}", flush=True)
                pos.append(p)
                file_index.append(np.full(p.shape[0], k, np.int16))
                params.append((lam, tilt, x0, Zabs, direction))
            pos = np.concatenate(pos)
            g = h.create_group(group)
            g.create_dataset("pos", data=pos, chunks=(1, WINDOW, 2))
            g["file_index"] = np.concatenate(file_index)
            g["files"] = np.array([p.name for p in paths], dtype=h5py.string_dtype())
            par = np.array(params)
            for name, col in zip(("lam", "tilt", "x0", "Zabs"), par[:, :4].T):
                g[name] = col
            g["direction"] = par[:, 4].astype(np.int8)
            print(f"{group}: N={pos.shape[0]} T={pos.shape[1]} {pos.nbytes / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
