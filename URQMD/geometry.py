"""Periodic geometry for the particle U-Net, vectorised over all events of a batch. Lengths in units of r=1."""
import torch


def min_image(d, L):
    return d - L * torch.round(d / L)


def cutoff(rho, R):
    # cosine cutoff, C^1 at rho=R, used both as edge envelope and as assignment kernel
    return 0.5 * (1 + torch.cos(torch.pi * rho / R)) * (rho < R)


def cells(pos, batch, m, h):
    c = torch.floor(pos / h[batch][:, None]).long()
    return torch.minimum(c, (m[batch] - 1)[:, None])   # pos/h can round up to m at the box edge


def radius_pairs(pos_a, batch_a, pos_b, batch_b, L, R, exclude_self):
    # all (a, b) pairs in the same event with |pos_b - pos_a|_min < R, via a cell list of side >= R.
    # returns (ia, ib, d) with d = pos_b[ib] - pos_a[ia] minimum image. needs R <= L/2 so every pair has one image inside R
    m = torch.floor(L / R).long()
    h = L / m
    cell_off = torch.cat([m.new_zeros(1), (m * m).cumsum(0)])
    cb = cells(pos_b, batch_b, m, h)
    key_b = cell_off[batch_b] + cb[:, 0] * m[batch_b] + cb[:, 1]
    order = torch.argsort(key_b)
    count = torch.bincount(key_b, minlength=int(cell_off[-1]))
    start = count.cumsum(0) - count
    ca, ma = cells(pos_a, batch_a, m, h), m[batch_a]
    dxy = torch.tensor([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], device=pos_a.device)
    c = (ca[None] + dxy[:, None]) % ma[None, :, None]                      # (9, n_a, 2)
    k9 = cell_off[batch_a][None] + c[..., 0] * ma[None] + c[..., 1]
    # with fewer than 3 cells per axis (L < 3R) several stencil offsets hit the same cell, visit each cell once
    ks, o9 = torch.sort(k9, dim=0)
    dup = torch.zeros_like(k9, dtype=torch.bool).scatter_(0, o9[1:], ks[1:] == ks[:-1])
    k = k9.reshape(-1)
    cnt = count[k] * ~dup.reshape(-1)
    i = torch.repeat_interleave(torch.arange(pos_a.shape[0], device=pos_a.device).repeat(9), cnt)
    offs = torch.arange(int(cnt.sum()), device=pos_a.device) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    j = order[torch.repeat_interleave(start[k], cnt) + offs]
    d = min_image(pos_b[j] - pos_a[i], L[batch_a[i]][:, None])
    keep = (d * d).sum(-1) < R * R
    if exclude_self:
        keep &= i != j
    return i[keep], j[keep], d[keep]


def grids(Ls, h, shift):
    # periodic square grids of spacing h for every event, L/h integer, shift (B,2) in [0,h)^2. grouped by distinct L
    pts, bid = [], []
    ev = torch.arange(len(Ls), device=shift.device)
    for L in sorted(set(Ls)):
        n = int(round(L / h))
        ax = torch.arange(n, device=shift.device, dtype=shift.dtype) * h
        g = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
        sel = ev[[l == L for l in Ls]]
        pts.append(((g[None] + shift[sel][:, None]) % L).reshape(-1, 2))
        bid.append(sel.repeat_interleave(n * n))
    return torch.cat(pts), torch.cat(bid)


def assign(pos_f, batch_f, pos_g, batch_g, L, R):
    # partition-of-unity weights from fine points to grid points: w = cutoff / sum_g cutoff, so sum_g w = 1 per fine point
    i, g, d = radius_pairs(pos_f, batch_f, pos_g, batch_g, L, R, False)
    w = cutoff(d.norm(dim=-1), R)
    norm = torch.zeros(pos_f.shape[0], device=w.device, dtype=w.dtype).index_add_(0, i, w)
    return i, g, w / norm[i]


def build_geometry(pos, batch, Ls, shift, R_fine, h_mid, R_mid, h_coarse, R_coarse):
    # everything the network needs that depends only on positions. shift: (B,2) grid phase per event
    L = torch.tensor(Ls, device=pos.device, dtype=pos.dtype)
    geo = {}
    geo["fine"] = radius_pairs(pos, batch, pos, batch, L, R_fine, True)
    pos_m, batch_m = grids(Ls, h_mid, shift)
    pos_c, batch_c = grids(Ls, h_coarse, shift)
    geo["batch_m"], geo["batch_c"] = batch_m, batch_c
    geo["mid"] = radius_pairs(pos_m, batch_m, pos_m, batch_m, L, R_mid, True)
    geo["coarse"] = radius_pairs(pos_c, batch_c, pos_c, batch_c, L, R_coarse, True)
    geo["f2m"] = assign(pos, batch, pos_m, batch_m, L, 1.5 * h_mid)
    geo["m2c"] = assign(pos_m, batch_m, pos_c, batch_c, L, 1.5 * h_coarse)
    geo["n_m"], geo["n_c"] = pos_m.shape[0], pos_c.shape[0]
    return geo
