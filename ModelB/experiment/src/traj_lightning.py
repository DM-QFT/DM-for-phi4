"""Conditional VE-SDE kernel p(dr | r) for single-particle tracks in the periodic light field."""

import contextlib
import math

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from diffusion_lightning import DiffusionModel


def cond_features(r, x0, tilt, lam, fov=None, grid=4):
    """Fringe phase of r = (x, y) [um], well bottoms at u = 0 mod 1.
    With fov, also the centre of the grid x grid cell of the field of view that holds r, in [0, 1]: coarse enough
    that a cell holds ~75 tracks, so the beam envelope is resolved but no single track is."""
    u = (r[:, 0] - x0) / lam - tilt * r[:, 1]
    c = [torch.cos(2 * torch.pi * u), torch.sin(2 * torch.pi * u)]
    if fov is not None:
        c += [(torch.floor(r[:, k] / fov[k] * grid).clamp(0, grid - 1) + 0.5) / grid for k in (0, 1)]
    return torch.stack(c, 1)


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreMLP(nn.Module):
    """Score of the noised displacement given the fringe phase and the diffusion time."""

    def __init__(self, marginal_prob_std_fn, n_cond=2, embed_dim=128, width=256, depth=4, embed_logstd=False):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim),
                                   nn.Linear(embed_dim, embed_dim), nn.SiLU())
        layers = [nn.Linear(2 + n_cond + embed_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        # optional: feed the Fourier features ln std(t) mapped onto [0, 1] instead of t, so the noise levels
        # between the well width (t ~ 3e-3) and the period are resolved as finely as the large-noise end
        self.embed_logstd = embed_logstd
        lo, hi = (float(torch.log(marginal_prob_std_fn(torch.tensor(t, dtype=torch.float64)))) for t in (1e-5, 1.0))
        self.log_std_lo, self.log_std_range = lo, hi - lo

    def forward(self, x, cond, t):
        s = (torch.log(self.marginal_prob_std(t)) - self.log_std_lo) / self.log_std_range if self.embed_logstd else t
        h = self.net(torch.cat([x, cond, self.embed(s)], 1))
        return h / self.marginal_prob_std(t)[:, None]


class TrajKernel(DiffusionModel):
    """Score of dr = r(t+dt) - r(t), normalised by norm_max, conditioned on the fringe phase of r(t)."""

    def __init__(self, score_model, sigma=12.0, lr=1e-3, lr_min=1e-5, ema_decay=0.9999,
                 norm_min=None, norm_max=None, group="eq", dt_frames=250, window_frames=15000,
                 data_path=None, run_dir=None, t_log=False, embed_logstd=False, cond_xy=False, fov=None, grid=4):
        super().__init__(score_model, sigma=sigma, lr=lr, ema_decay=ema_decay,
                         norm_min=norm_min, norm_max=norm_max)
        self.lr_min = lr_min
        self.t_log = t_log
        self.fov = fov if cond_xy else None

    def forward(self, x, cond, t):
        return self.score_model(x, cond, t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=self.lr_min)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def loss_fn(self, dr, cond, eps=1e-5):
        v = torch.rand(dr.shape[0], device=dr.device)
        t = torch.exp(v * math.log(eps)) if self.t_log else v * (1. - eps) + eps   # log-uniform or uniform on [eps, 1]
        z = torch.randn_like(dr)
        std = self.marginal_prob_std_fn(t)[:, None]
        score = self(dr + z * std, cond, t)
        return (score * std + z).square().sum(1).mean()

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(*batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def sample(self, cond, num_steps=2000, eps=1e-5, schedule="log", use_ema=True):
        """Euler-Maruyama reverse SDE, one EM step captured as a CUDA graph; returns dr in um."""
        self.eval()
        device = self.device
        cond = cond.to(device=device, dtype=torch.float32)
        ema_ctx = self.ema.average_parameters() if use_ema else contextlib.nullcontext()
        with ema_ctx:
            times = self._build_time_steps(num_steps, eps, schedule, device)
            dt = times[:-1] - times[1:]
            x = torch.randn(cond.shape[0], 2, device=device) * self.marginal_prob_std_fn(times[0])
            x0 = x.clone()
            batch_t = torch.empty(cond.shape[0], device=device)
            t_i, dt_i = torch.zeros((), device=device), torch.zeros((), device=device)

            def em_step():
                batch_t.fill_(t_i)
                g = self.diffusion_coeff_fn(t_i)
                mean = x + g.square() * self(x, cond, batch_t) * dt_i
                x.copy_(mean + g * torch.sqrt(dt_i) * torch.randn_like(x))
                return mean

            t_i.fill_(times[0])
            dt_i.fill_(dt[0])
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    em_step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mean_x = em_step()
            x.copy_(x0)
            for i in range(num_steps):
                t_i.fill_(times[i])
                dt_i.fill_(dt[i])
                graph.replay()
        # the EM chain diverges for about one sample in 1e6 (a rare noise sequence at small t); redraw those rows
        bad = ~torch.isfinite(mean_x).all(1)
        if bad.any():
            mean_x[bad] = self.sample(cond[bad], num_steps, eps, schedule, use_ema) / self.norm_max
        return mean_x * self.norm_max


class PairSampler:
    """Random (track, t0) pairs at lag dt, drawn on the GPU every step."""

    def __init__(self, pos, par, A_res, dt, batch_size, steps_per_epoch, fov=None):
        self.pos, self.par, self.A_res, self.dt, self.fov = pos, par, A_res, dt, fov
        self.batch_size, self.steps_per_epoch = batch_size, steps_per_epoch

    def __iter__(self):
        N, W = self.pos.shape[:2]
        for _ in range(self.steps_per_epoch):
            i = torch.randint(N, (self.batch_size,), device=self.pos.device)
            t0 = torch.randint(W - self.dt, (self.batch_size,), device=self.pos.device)
            r0 = self.pos[i, t0]
            dr = (self.pos[i, t0 + self.dt] - r0) / self.A_res
            yield dr, cond_features(r0, *(p[i] for p in self.par), self.fov)

    def __len__(self):
        return self.steps_per_epoch


class TrajPairData(pl.LightningDataModule):
    """Training window of every track, resident on the GPU."""

    def __init__(self, h5_path, group, dt, window, batch_size, steps_per_epoch, device="cuda:0"):
        super().__init__()
        self.h5_path, self.group, self.dt, self.window = h5_path, group, dt, window
        self.batch_size, self.steps_per_epoch = batch_size, steps_per_epoch
        self.device = torch.device(device)
        self.A_res = None

    def setup(self, stage=None):
        if self.A_res is not None:
            return
        with h5py.File(self.h5_path, "r") as h:
            g = h[self.group]
            self.pos = torch.from_numpy(g["pos"][:, :self.window]).to(self.device)
            fi = torch.from_numpy(g["file_index"][:].astype(np.int64))
            self.par = [torch.from_numpy(g[k][:].astype(np.float32))[fi].to(self.device)
                        for k in ("x0", "tilt", "lam")]
            self.fov = torch.tensor(h.attrs["fov_um"], dtype=torch.float32, device=self.device)
        dr = self.pos[:, self.dt:] - self.pos[:, :-self.dt]
        self.A_res = float(dr.abs().max())
        # diameter of the normalised target cloud = max over directions of the projected width
        d2 = dr.reshape(-1, 2) / self.A_res
        self.D_max_norm = max(float(p.max() - p.min()) for p in
                              (d2[:, 0] * np.cos(th) + d2[:, 1] * np.sin(th) for th in np.linspace(0, np.pi, 720)))
        std = d2.std(0)
        print(f"{self.group}: N={self.pos.shape[0]} window={self.window} lag={self.dt} "
              f"A_res={self.A_res:.4f} um, normalised std=({std[0]:.4f}, {std[1]:.4f}), "
              f"D_max_norm={self.D_max_norm:.3f}", flush=True)

    def train_dataloader(self):
        return PairSampler(self.pos, self.par, self.A_res, self.dt,
                           self.batch_size, self.steps_per_epoch)
