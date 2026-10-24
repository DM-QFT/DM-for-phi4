"""Particle U-Net noise predictor: PaiNN-style scalar/vector message passing in 2D on particles plus two periodic auxiliary grids."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry import build_geometry, cutoff
from config import R_FINE, H_MID, R_MID, H_COARSE, R_COARSE

N_RBF = 16
T_DIM = 32
C_LEVELS = (32, 64, 128)
V_LEVELS = (8, 16, 32)
N_BLOCKS = 1


def rbf(rho, R):
    centers = torch.linspace(0, R, N_RBF, device=rho.device, dtype=rho.dtype)
    return torch.exp(-((rho[:, None] - centers) * ((N_RBF - 1) / R)) ** 2)


def mlp(din, dh, dout, zero=False):
    # zero: last layer starts at zero so a residual block starts as the identity
    m = nn.Sequential(nn.Linear(din, dh), nn.SiLU(), nn.Linear(dh, dout))
    if zero:
        nn.init.zeros_(m[2].weight)
        nn.init.zeros_(m[2].bias)
    return m


def vlin(lin, v):
    # channel mixing of vector features (n, V, 2) without touching the spatial index
    return lin(v.transpose(1, 2)).transpose(1, 2)


def vnorm(v):
    # equivariant per-node normalisation of the vector channels (rms norm over channels), identity for small v.
    # without it the unnormalised residual stream overflowed float32 after ~3000 steps with 2 blocks per level on dense cluster events
    return v * torch.rsqrt(v.pow(2).sum(-1).mean(-1, keepdim=True)[..., None] + 1)


class TimeEmbedding(nn.Module):
    # tau = log sigma spans about 8 units. fixed geometric periods from 0.5 to 20 plus tau/10 as a raw input
    def __init__(self):
        super().__init__()
        self.register_buffer("freq", 2 * torch.pi / torch.tensor(np.geomspace(0.5, 20, 8), dtype=torch.float32))
        self.net = mlp(17, T_DIM, T_DIM)

    def forward(self, sigma):
        tau = torch.log(sigma)[:, None]
        return self.net(torch.cat([(tau * self.freq).sin(), (tau * self.freq).cos(), tau / 10], -1))


class Block(nn.Module):
    def __init__(self, C, V, n_edge):
        super().__init__()
        self.C, self.V = C, V
        self.filt = mlp(n_edge, C, C + 2 * V)
        self.ln1 = nn.LayerNorm(C)
        self.phi = mlp(C + 2 + T_DIM, C, C + 2 * V, zero=True)
        self.U = nn.Linear(V, V, bias=False)
        self.W = nn.Linear(V, V, bias=False)
        self.ln2 = nn.LayerNorm(C)
        self.upd = mlp(C + V + 2 + T_DIM, C, C + 2 * V, zero=True)
        self.proj = nn.Linear(V, C)

    def forward(self, s, v, c, temb, i, j, d, eattr, env):
        C, V = self.C, self.V
        # the envelope keeps every message C^1 at the cutoff radius. the displacement d (not d/|d|) keeps it continuous at coincidence
        m = self.phi(torch.cat([self.ln1(s), c, temb], -1))[j] * self.filt(eattr) * env[:, None]
        s = s + torch.zeros_like(s).index_add_(0, i, m[:, :C])
        mv = m[:, C:C + V, None] * d[:, None, :] + m[:, C + V:, None] * vnorm(v)[j]
        v = v + torch.zeros_like(v).index_add_(0, i, mv)
        vn = vnorm(v)
        Uv, Wv = vlin(self.U, vn), vlin(self.W, vn)
        a = self.upd(torch.cat([self.ln2(s), Wv.norm(dim=-1), c, temb], -1))
        s = s + a[:, :C] + self.proj(a[:, C:C + V] * (Uv * Wv).sum(-1))
        v = v + a[:, C + V:, None] * Uv
        return s, v


class Down(nn.Module):
    # norm: expected total assignment weight per grid node at density 1 per type (a constant, not a graph statistic)
    def __init__(self, C_in, V_in, C_out, V_out, norm):
        super().__init__()
        self.ls = nn.Linear(C_in, C_out)
        self.lv = nn.Linear(V_in, V_out, bias=False)
        self.norm = norm

    def forward(self, s, v, cnt, i, g, w, n_g):
        s_g = s.new_zeros(n_g, self.ls.out_features).index_add_(0, g, w[:, None] * self.ls(s)[i]) / self.norm
        v_g = v.new_zeros(n_g, self.lv.out_features, 2).index_add_(0, g, w[:, None, None] * vlin(self.lv, v)[i]) / self.norm
        cnt_g = cnt.new_zeros(n_g, cnt.shape[1]).index_add_(0, g, w[:, None] * cnt[i])
        return s_g, v_g, cnt_g


class Up(nn.Module):
    def __init__(self, C_in, V_in, C_out, V_out):
        super().__init__()
        self.ls = nn.Linear(C_in, C_out)
        self.lv = nn.Linear(V_in, V_out, bias=False)
        self.ms = nn.Linear(2 * C_out, C_out)
        self.mv = nn.Linear(2 * V_out, V_out, bias=False)

    def forward(self, s_g, v_g, s_skip, v_skip, i, g, w):
        s = s_skip.new_zeros(s_skip.shape[0], self.ls.out_features).index_add_(0, i, w[:, None] * self.ls(s_g)[g])
        v = v_skip.new_zeros(v_skip.shape[0], self.lv.out_features, 2).index_add_(0, i, w[:, None, None] * vlin(self.lv, v_g)[g])
        return self.ms(torch.cat([s, s_skip], -1)), vlin(self.mv, torch.cat([v, v_skip], 1))


def fine_inputs(typ, geo, dtype):
    i, j, d = geo["fine"]
    rho = d.norm(dim=-1)
    onehot = F.one_hot(typ, 2).to(dtype)
    env = cutoff(rho, R_FINE)
    eattr = torch.cat([rbf(rho, R_FINE), onehot[i], onehot[j]], -1)
    cnt = torch.log1p(onehot.new_zeros(onehot.shape).index_add_(0, i, env[:, None] * onehot[j]))
    return onehot, env, eattr, cnt


BUCKET = 2048


def bucket(n):
    # padding granularity grows with n (at most 8 distinct shapes per octave) so large sets recompile rarely and waste < 1/8
    return max(BUCKET, 1 << max(n.bit_length() - 4, 0))


def pad_geometry(typ, batch, geo):
    # static shapes for torch.compile: nodes, fine edges and particle-grid pairs are padded to multiples of a bucket.
    # the dummy node (last index, type 0) only receives self-loop messages and zero-weight assignments, so real nodes never see it
    n = typ.shape[0]
    n_pad = (n // bucket(n) + 1) * bucket(n)
    dummy = n_pad - 1
    pad = lambda t, k, value: torch.cat([t, t.new_full((k, *t.shape[1:]), value)])
    i, j, d = geo["fine"]
    k_e = (-i.shape[0]) % (8 * bucket(i.shape[0]))
    a, g, w = geo["f2m"]
    k_a = (-a.shape[0]) % (8 * bucket(a.shape[0]))
    out = dict(geo, fine=(pad(i, k_e, dummy), pad(j, k_e, dummy), pad(d, k_e, 0)), f2m=(pad(a, k_a, dummy), pad(g, k_a, 0), pad(w, k_a, 0)))
    return pad(typ, n_pad - n, 0), pad(batch, n_pad - n, 0), out, n


class ParticleUNet(nn.Module):
    def __init__(self, C=C_LEVELS, V=V_LEVELS, n_blocks=N_BLOCKS):
        super().__init__()
        self.embed = nn.Embedding(2, C[0])
        self.temb = TimeEmbedding()
        self.enc_f = nn.ModuleList([Block(C[0], V[0], N_RBF + 4) for _ in range(n_blocks)])
        self.down_fm = Down(C[0], V[0], C[1], V[1], 2.0)
        self.enc_m = nn.ModuleList([Block(C[1], V[1], N_RBF) for _ in range(n_blocks)])
        self.down_mc = Down(C[1], V[1], C[2], V[2], 4.0)
        self.coarse = nn.ModuleList([Block(C[2], V[2], N_RBF) for _ in range(n_blocks)])
        self.up_cm = Up(C[2], V[2], C[1], V[1])
        self.dec_m = nn.ModuleList([Block(C[1], V[1], N_RBF) for _ in range(n_blocks)])
        self.up_mf = Up(C[1], V[1], C[0], V[0])
        self.dec_f = nn.ModuleList([Block(C[0], V[0], N_RBF + 4) for _ in range(n_blocks)])
        self.out = nn.Linear(V[0], 1, bias=False)

    def forward(self, pos, typ, batch, Ls, sigma, shift):
        # pos (n,2) concatenated events, typ 0 = condition X0 / 1 = noisy Z, batch (n,), Ls list of floats, sigma (B,), shift (B,2).
        # geometry stays eager (data-dependent shapes), core runs on padded static shapes and may be torch.compiled
        geo = build_geometry(pos, batch, Ls, shift, R_FINE, H_MID, R_MID, H_COARSE, R_COARSE)
        typ_p, batch_p, geo_p, n = pad_geometry(typ, batch, geo)
        return self.core(typ_p, batch_p, sigma, geo_p)[:n][typ == 1]

    def core(self, typ, batch, sigma, geo):
        temb = self.temb(sigma)
        onehot, env_f, e_f, c_f = fine_inputs(typ, geo, self.embed.weight.dtype)
        i, j, d = geo["fine"]
        t_f = temb[batch]
        s, v = self.embed(typ), onehot.new_zeros(typ.shape[0], self.out.in_features, 2)
        for blk in self.enc_f:
            s, v = blk(s, v, c_f, t_f, i, j, d, e_f, env_f)
        s_skip_f, v_skip_f = s, v

        fi, fg, fw = geo["f2m"]
        s_m, v_m, cnt_m = self.down_fm(s, v, onehot, fi, fg, fw, geo["n_m"])
        c_m, t_m = torch.log1p(cnt_m), temb[geo["batch_m"]]
        im, jm, dm = geo["mid"]
        env_m, e_m = cutoff(dm.norm(dim=-1), R_MID), rbf(dm.norm(dim=-1), R_MID)
        for blk in self.enc_m:
            s_m, v_m = blk(s_m, v_m, c_m, t_m, im, jm, dm, e_m, env_m)
        s_skip_m, v_skip_m = s_m, v_m

        mi, mg, mw = geo["m2c"]
        s_c, v_c, cnt_c = self.down_mc(s_m, v_m, cnt_m, mi, mg, mw, geo["n_c"])
        c_c, t_c = torch.log1p(cnt_c), temb[geo["batch_c"]]
        ic, jc, dc = geo["coarse"]
        env_c, e_c = cutoff(dc.norm(dim=-1), R_COARSE), rbf(dc.norm(dim=-1), R_COARSE)
        for blk in self.coarse:
            s_c, v_c = blk(s_c, v_c, c_c, t_c, ic, jc, dc, e_c, env_c)

        s_m, v_m = self.up_cm(s_c, v_c, s_skip_m, v_skip_m, mi, mg, mw)
        for blk in self.dec_m:
            s_m, v_m = blk(s_m, v_m, c_m, t_m, im, jm, dm, e_m, env_m)

        s, v = self.up_mf(s_m, v_m, s_skip_f, v_skip_f, fi, fg, fw)
        for blk in self.dec_f:
            s, v = blk(s, v, c_f, t_f, i, j, d, e_f, env_f)
        return self.out(v.transpose(1, 2)).squeeze(-1)


class ParticleGNN(nn.Module):
    # single-scale baseline with the same local block
    def __init__(self, C=C_LEVELS[0], V=V_LEVELS[0], n_blocks=6):
        super().__init__()
        self.embed = nn.Embedding(2, C)
        self.temb = TimeEmbedding()
        self.blocks = nn.ModuleList([Block(C, V, N_RBF + 4) for _ in range(n_blocks)])
        self.out = nn.Linear(V, 1, bias=False)

    def forward(self, pos, typ, batch, Ls, sigma, shift):
        geo = build_geometry(pos, batch, Ls, shift, R_FINE, H_MID, R_MID, H_COARSE, R_COARSE)
        typ_p, batch_p, geo_p, n = pad_geometry(typ, batch, geo)
        return self.core(typ_p, batch_p, sigma, geo_p)[:n][typ == 1]

    def core(self, typ, batch, sigma, geo):
        temb = self.temb(sigma)
        onehot, env_f, e_f, c_f = fine_inputs(typ, geo, self.embed.weight.dtype)
        i, j, d = geo["fine"]
        t_f = temb[batch]
        s, v = self.embed(typ), onehot.new_zeros(typ.shape[0], self.out.in_features, 2)
        for blk in self.blocks:
            s, v = blk(s, v, c_f, t_f, i, j, d, e_f, env_f)
        return self.out(v.transpose(1, 2)).squeeze(-1)


def collate(x0s, zs):
    # concatenate events on the default device, condition nodes first then noisy nodes within each event. inputs are numpy arrays or tensors
    pos = torch.cat([torch.as_tensor(x) for pair in zip(x0s, zs) for x in pair])
    n0, nz = torch.tensor([len(x) for x in x0s]), torch.tensor([len(z) for z in zs])
    typ = torch.repeat_interleave(torch.tensor([0, 1]).repeat(len(x0s)), torch.stack([n0, nz], 1).reshape(-1))
    batch = torch.repeat_interleave(torch.arange(len(x0s)), n0 + nz)
    return pos, typ, batch
