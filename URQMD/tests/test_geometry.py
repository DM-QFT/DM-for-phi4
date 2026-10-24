"""Geometry checks against scipy: minimum image, periodic radius pairs across events, grids, partition-of-unity weights."""
import os
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry import min_image, cutoff, radius_pairs, grids, build_geometry

torch.manual_seed(0)
rng = np.random.default_rng(0)
dt = torch.float64


def pairset(i, j):
    return set(map(tuple, torch.stack([i, j], 1).tolist()))


# minimum image
L = 7.0
d = (torch.rand(10000, 2, dtype=dt) - 0.5) * 6 * L
m = min_image(d, L)
assert (m.abs() <= L / 2 + 1e-12).all()
assert ((d - m) / L - torch.round((d - m) / L)).abs().max() < 1e-9
print("min_image ok")

# radius pairs of several events at once vs cKDTree per event, including pairs across the boundary, cells of size > R,
# and boxes with only two cells per axis (L = 8 and 10 at R = 4), where the 3x3 stencil must not revisit a cell
Ls = [16.0, 12.0, 20.0, 12.0, 8.0, 10.0]
xs = [torch.tensor(rng.random((n, 2)) * L, dtype=dt) for n, L in zip((400, 200, 600, 50, 80, 120), Ls)]
pos = torch.cat(xs)
batch = torch.cat([torch.full((len(x),), b) for b, x in enumerate(xs)])
Lt = torch.tensor(Ls, dtype=dt)
for R in (1.0, 2.5, 4.0):
    i, j, d = radius_pairs(pos, batch, pos, batch, Lt, R, True)
    assert (batch[i] == batch[j]).all()
    assert (d - min_image(pos[j] - pos[i], Lt[batch[i]][:, None])).abs().max() < 1e-12
    assert (d.norm(dim=-1) < R).all()
    off = 0
    for b, x in enumerate(xs):
        ref = cKDTree(x.numpy(), boxsize=Ls[b]).query_pairs(R, output_type="ndarray")
        exp = set(map(tuple, ref.tolist())) | set(map(tuple, ref[:, ::-1].tolist()))
        sel = batch[i] == b
        assert pairset(i[sel] - off, j[sel] - off) == exp, (b, R)
        off += len(x)
    print(f"radius_pairs R={R}: {len(i)} directed edges over {len(Ls)} events match cKDTree")

# bipartite pairs between two point sets
ys = [torch.tensor(rng.random((n, 2)) * L, dtype=dt) for n, L in zip((300, 100, 200, 40, 60, 90), Ls)]
pos_b = torch.cat(ys)
batch_b = torch.cat([torch.full((len(y),), b) for b, y in enumerate(ys)])
ia, ib, dd = radius_pairs(pos, batch, pos_b, batch_b, Lt, 1.5, False)
offa = offb = 0
for b, (x, y) in enumerate(zip(xs, ys)):
    ref = cKDTree(x.numpy(), boxsize=Ls[b]).query_ball_tree(cKDTree(y.numpy(), boxsize=Ls[b]), 1.5)
    sel = batch[ia] == b
    assert pairset(ia[sel] - offa, ib[sel] - offb) == {(a, c) for a, cs in enumerate(ref) for c in cs}
    offa, offb = offa + len(x), offb + len(y)
print("bipartite pairs ok")

# grids: right count per event, inside the box, correct spacing and shift
shift = torch.tensor(rng.random((len(Ls), 2)) * 2, dtype=dt)
g, bg = grids(Ls, 2.0, shift)
assert g.shape == (sum((int(L) // 2) ** 2 for L in Ls), 2) and (g >= 0).all()
for b, L in enumerate(Ls):
    gb = g[bg == b]
    assert len(gb) == (int(L) // 2) ** 2 and (gb < L).all()
    assert torch.allclose(torch.sort(gb[:, 0]).values[::int(L) // 2], (torch.arange(int(L) // 2, dtype=dt) * 2 + shift[b, 0]) % L)
print("grids ok")

# partition of unity: weights sum to one per fine point and every fine point reaches at least four grid points
geo = build_geometry(pos, batch, Ls, torch.tensor(rng.random((len(Ls), 2)), dtype=dt), 1.0, 1.0, 2.0, 2.0, 4.0)
for key, n_f in (("f2m", pos.shape[0]), ("m2c", geo["n_m"])):
    fi, gg, w = geo[key]
    tot = torch.zeros(n_f, dtype=dt).index_add_(0, fi, w)
    assert torch.allclose(tot, torch.ones_like(tot)), (key, tot.min(), tot.max())
    assert (w >= 0).all()
    assert torch.bincount(fi, minlength=n_f).min() >= 4, key
print("assignment weights ok, mid points", geo["n_m"], "coarse points", geo["n_c"])
assert geo["n_m"] == sum(int(L) ** 2 for L in Ls) and geo["n_c"] == sum((int(L) // 2) ** 2 for L in Ls)
assert (geo["batch_m"][geo["mid"][0]] == geo["batch_m"][geo["mid"][1]]).all()
assert (geo["batch_c"][geo["coarse"][0]] == geo["batch_c"][geo["coarse"][1]]).all()
assert torch.bincount(geo["mid"][0], minlength=geo["n_m"]).min() == 8   # spacing 1, strict radius 2: 4 axial + 4 diagonal
print("grid graphs ok")

# cutoff vanishes continuously at R
rho = torch.tensor([0.0, 0.5, 1.0 - 1e-6, 1.0, 1.5], dtype=dt)
c = cutoff(rho, 1.0)
assert c[0] == 1 and abs(c[1] - 0.5) < 1e-12 and c[2] < 1e-10 and c[3] == 0 and c[4] == 0
print("cutoff ok")
print("ALL GEOMETRY TESTS PASSED")
