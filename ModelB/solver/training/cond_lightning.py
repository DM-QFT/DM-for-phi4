"""Conditional VE-SDE diffusion for the Model B kernel p(dphi | phi0)."""

import contextlib
import os
import sys

MODEL_B_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DM_DIR = os.path.dirname(MODEL_B_DIR)
sys.path.insert(0, DM_DIR)

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

from diffusion_lightning import DiffusionModel
from networks import NCSNpp2D


class NCSNppCond2D(NCSNpp2D):
    """NCSNpp2D with input channels [noised residual, clean condition]."""

    def __init__(self, marginal_prob_std_fn, in_channels=2,
                 channels=[16, 32, 64, 128], embed_dim=256,
                 use_attention=False, periodic=True):
        super().__init__(marginal_prob_std_fn, channels=channels,
                         embed_dim=embed_dim, use_attention=use_attention,
                         periodic=periodic)
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(
            in_channels, channels[0], 3, padding=1,
            padding_mode="circular" if periodic else "zeros",
        )


class ConditionalDiffusionModel(DiffusionModel):
    """Learn the score of residual dphi conditioned on the clean phi0."""

    def __init__(self, score_model, sigma=250.0, lr=1e-3, L=128,
                 ema_decay=0.9999, ema_start_epoch=0,
                 norm_min=None, norm_max=None, zero_mode_loss_weight=0.0,
                 cond_norm_min=None, cond_norm_max=None,
                 kappa=None, lam=None, horizon=None,
                 cond_data_path=None, target_data_path=None, run_dir=None,
                 in_channels=2, z2_aug=True):
        super().__init__(
            score_model, sigma=sigma, lr=lr, L=L,
            ema_decay=ema_decay, ema_start_epoch=ema_start_epoch,
            norm_min=norm_min, norm_max=norm_max, spatial_dims=2,
            zero_mode_loss_weight=zero_mode_loss_weight,
        )
        self.cond_norm_min = cond_norm_min
        self.cond_norm_max = cond_norm_max
        self.z2_aug = z2_aug
        self._ema_parameters = tuple(self.score_model.parameters())

    def loss_fn(self, x, eps=1e-5):
        if self.z2_aug and self.training:
            # Exact symmetry p(dphi|phi0) = p(-dphi|-phi0).
            sign = torch.randint(
                0, 2, (x.shape[0], 1, 1, 1),
                device=x.device, dtype=x.dtype,
            ) * 2 - 1
            x = x * sign
        x_res, x_cond = x[:, :1], x[:, 1:]
        random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
        noise = torch.randn_like(x_res)
        std = self.marginal_prob_std_fn(random_t)
        perturbed = x_res + noise * self._bcast(std)
        score = self(torch.cat([perturbed, x_cond], dim=1), random_t)
        error = score * self._bcast(std) + noise
        dsm, k0, _ = self._dsm_loss_components(error)
        loss = (dsm + self.zero_mode_loss_weight * k0).mean()
        return loss, k0.detach().mean()

    def training_step(self, batch, batch_idx):
        loss, k0 = self.loss_fn(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("train_loss_k0_epoch", k0, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def _update_ema_foreach(self):
        decay = self.ema.decay
        if self.ema.num_updates is not None:
            self.ema.num_updates += 1
            decay = min(
                decay,
                (1 + self.ema.num_updates) / (10 + self.ema.num_updates),
            )
        torch._foreach_lerp_(
            self.ema.shadow_params,
            self._ema_parameters,
            1.0 - decay,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch >= self.ema_start_epoch:
            self._update_ema_foreach()

    @torch.no_grad()
    def sample(self, cond, num_steps=500, eps=1e-5,
               schedule="log", use_ema=True):
        """Euler-Maruyama reverse SDE in the zero-mean residual subspace."""
        if (cond.ndim != 4 or cond.shape[1] != 1
                or cond.shape[-2] != cond.shape[-1]
                or cond.shape[-1] % 8 != 0):
            raise ValueError(
                "cond must have shape (B,1,L,L) with L divisible by 8, "
                f"got {tuple(cond.shape)}"
            )
        self.eval()
        device = self.device
        cond = cond.to(device=device, dtype=torch.float32)
        batch_size = cond.shape[0]
        lattice_size = cond.shape[-1]
        ema_ctx = self.ema.average_parameters() if use_ema else contextlib.nullcontext()
        with ema_ctx, torch.autocast(device.type, dtype=torch.bfloat16):
            times = self._build_time_steps(num_steps, eps, schedule, device)
            dt = times[:-1] - times[1:]
            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(
                batch_size, 1, lattice_size, lattice_size, device=device
            ) * init_std
            x -= x.mean(dim=(-2, -1), keepdim=True)
            batch_t = torch.empty(batch_size, device=device)
            for i in tqdm(range(num_steps), desc="Sampling (cond EM)"):
                batch_t.copy_(times[i])
                g = self.diffusion_coeff_fn(times[i])
                score = self(torch.cat([x, cond], dim=1), batch_t).float()
                score -= score.mean(dim=(-2, -1), keepdim=True)
                mean_x = x + g.square() * score * dt[i]
                noise = torch.randn_like(x)
                noise -= noise.mean(dim=(-2, -1), keepdim=True)
                x = mean_x + g * torch.sqrt(dt[i]) * noise
        mean_x -= mean_x.mean(dim=(-2, -1), keepdim=True)
        return mean_x


class ShuffledTensorLoader:
    """Shuffle and batch a tensor already resident on one device."""

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        indices = torch.randperm(len(self.data), device=self.data.device)
        for start in range(0, len(self.data), self.batch_size):
            yield self.data[indices[start:start + self.batch_size]]

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class MixedLShuffledTensorLoader:
    """Random-L batches, with one lattice size in each batch."""

    def __init__(self, data_by_L, batch_size):
        self.data_by_L = data_by_L
        self.batch_size = batch_size
        self.Ls = sorted(data_by_L)
        sizes = np.array([len(data_by_L[L]) for L in self.Ls], dtype=np.float64)
        self.probabilities = sizes / sizes.sum()
        self.n_steps = int(sizes.sum() // batch_size)

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.n_steps):
            L = self.Ls[rng.choice(len(self.Ls), p=self.probabilities)]
            data = self.data_by_L[L]
            indices = torch.randint(
                len(data), (self.batch_size,), device=data.device
            )
            yield data[indices]

    def __len__(self):
        return self.n_steps


class ModelBPairDataModule(pl.LightningDataModule):
    """Load every paired field and normalize [dphi, phi0] to [-1, 1]."""

    def __init__(self, cond_path, target_path, batch_size=64, device="cuda:0",
                 expected_L=None, chunk_size=256):
        super().__init__()
        if isinstance(cond_path, (str, os.PathLike)):
            cond_path = [cond_path]
        if isinstance(target_path, (str, os.PathLike)):
            target_path = [target_path]
        self.cond_paths = [os.path.realpath(os.path.abspath(p)) for p in cond_path]
        self.target_paths = [os.path.realpath(os.path.abspath(p)) for p in target_path]
        if len(self.cond_paths) != len(self.target_paths):
            raise ValueError("cond_path and target_path must have the same length")
        self.cond_path = self.cond_paths[0] if len(self.cond_paths) == 1 else self.cond_paths
        self.target_path = (
            self.target_paths[0] if len(self.target_paths) == 1 else self.target_paths
        )
        self.batch_size = batch_size
        self.device = torch.device(device)
        if expected_L is None:
            self.expected_L = None
        elif np.isscalar(expected_L):
            self.expected_L = {int(expected_L)}
        else:
            self.expected_L = set(map(int, expected_L))
        self.chunk_size = chunk_size
        self.A_res = None
        self.A_cond = None
        self.num_pairs = None
        self.num_pairs_by_L = None
        self.data_on_gpu = None
        self.data_by_L = None
        self._setup_complete = False

    def setup(self, stage=None):
        if self._setup_complete:
            return
        for path in self.cond_paths + self.target_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        A_res = A_cond = 0.0
        pairs = []
        seen_L = set()
        for cond_path, target_path in zip(self.cond_paths, self.target_paths):
            with h5py.File(cond_path, "r") as cond_file, \
                    h5py.File(target_path, "r") as target_file:
                if "cfgs" not in cond_file or "cfgs" not in target_file:
                    raise KeyError("both HDF5 files must contain 'cfgs'")
                cond_ds, target_ds = cond_file["cfgs"], target_file["cfgs"]
                if cond_ds.shape != target_ds.shape or cond_ds.ndim != 3:
                    raise ValueError(
                        f"cfgs must have the same (N,L,L) shape, got "
                        f"{cond_ds.shape} and {target_ds.shape}"
                    )
                n, lx, ly = cond_ds.shape
                if n == 0 or lx != ly or lx in seen_L:
                    raise ValueError(f"invalid or duplicate cfgs shape {cond_ds.shape}")
                seen_L.add(lx)
                for dataset in (cond_ds, target_ds):
                    if (not np.issubdtype(dataset.dtype, np.number)
                            or np.issubdtype(dataset.dtype, np.complexfloating)):
                        raise TypeError(
                            f"cfgs must be real numeric, got {dataset.dtype}"
                        )
                pairs.append((cond_path, target_path, n, lx))

                for start in range(0, n, self.chunk_size):
                    stop = min(start + self.chunk_size, n)
                    cond = cond_ds[start:stop]
                    target = target_ds[start:stop]
                    residual = target - cond
                    cond_max = float(np.abs(cond).max())
                    residual_max = float(np.abs(residual).max())
                    if not np.isfinite((cond_max, residual_max)).all():
                        raise ValueError(f"non-finite cfgs in rows {start}:{stop}")
                    A_cond = max(A_cond, cond_max)
                    A_res = max(A_res, residual_max)

        if self.expected_L is not None and seen_L != self.expected_L:
            raise ValueError(
                f"data lattice sizes {sorted(seen_L)} do not match "
                f"expected {sorted(self.expected_L)}"
            )
        if A_res <= 0.0 or A_cond <= 0.0:
            raise ValueError(
                f"invalid normalization A_res={A_res}, A_cond={A_cond}"
            )

        self.data_by_L = {}
        self.num_pairs_by_L = {}
        for cond_path, target_path, n, L in pairs:
            with h5py.File(cond_path, "r") as cond_file, \
                    h5py.File(target_path, "r") as target_file:
                cond_ds, target_ds = cond_file["cfgs"], target_file["cfgs"]
                stacked = np.empty((n, 2, L, L), dtype=np.float32)
                for start in range(0, n, self.chunk_size):
                    stop = min(start + self.chunk_size, n)
                    cond = cond_ds[start:stop]
                    residual = target_ds[start:stop] - cond
                    np.divide(
                        residual, A_res, out=stacked[start:stop, 0],
                        casting="unsafe",
                    )
                    np.divide(
                        cond, A_cond, out=stacked[start:stop, 1],
                        casting="unsafe",
                    )
            self.data_by_L[L] = torch.from_numpy(stacked).to(self.device)
            self.num_pairs_by_L[L] = n

        self.A_res, self.A_cond = A_res, A_cond
        self.num_pairs = sum(self.num_pairs_by_L.values())
        if len(self.data_by_L) == 1:
            self.data_on_gpu = next(iter(self.data_by_L.values()))
        sizes = ", ".join(
            f"L={L}: {self.num_pairs_by_L[L]}" for L in sorted(self.data_by_L)
        )
        total_gb = sum(x.nbytes for x in self.data_by_L.values()) / 1e9
        print(
            f"Pairs: {self.num_pairs} training ({sizes}), "
            f"A_res={A_res:.17g}, A_cond={A_cond:.17g}"
        )
        print(f"Pair data loaded to {self.device} ({total_gb:.2f} GB)")
        self._setup_complete = True

    def train_dataloader(self):
        if len(self.data_by_L) == 1:
            return ShuffledTensorLoader(self.data_on_gpu, self.batch_size)
        return MixedLShuffledTensorLoader(self.data_by_L, self.batch_size)
