"""
EDM (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models") Lightning module.

Wraps an existing score network (constructed with marginal_prob_std_fn=trivial_std
so its internal /std division becomes /1) in EDM preconditioning. Training uses
log-normal sigma sampling; inference uses 2nd-order Heun sampling.

Reuses the existing NCSNpp2D / ScoreNet backbones without modification.
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm


def trivial_std(t):
    """marginal_prob_std_fn that returns ones. Pass to NCSNpp2D etc. to bypass
    their internal h / marginal_prob_std(t) division, so EDM preconditioning
    can apply its own sigma-aware scaling externally."""
    return torch.ones_like(t)


class EDMPrecond(nn.Module):
    """EDM preconditioning wrapper (Karras 2022, Eq. 7).

        D_theta(x; sigma) = c_skip(sigma) * x
                          + c_out(sigma) * F_theta(c_in(sigma) * x, c_noise(sigma))

    where F_theta is the underlying score network.
    """

    def __init__(self, score_network, sigma_data):
        super().__init__()
        self.net = score_network
        self.sigma_data = float(sigma_data)

    def _coeffs(self, sigma):
        sd = self.sigma_data
        s2 = sigma ** 2
        denom = torch.sqrt(s2 + sd ** 2)
        c_in = 1.0 / denom
        c_out = sigma * sd / denom
        c_skip = sd ** 2 / (s2 + sd ** 2)
        c_noise = 0.25 * torch.log(sigma)
        return c_in, c_out, c_skip, c_noise

    def forward(self, x_noisy, sigma):
        """D_theta(x_noisy; sigma): denoised estimate. Used in sampling."""
        c_in, c_out, c_skip, c_noise = self._coeffs(sigma)
        b = lambda v: v.view(-1, 1, 1, 1)
        F = self.net(b(c_in) * x_noisy, c_noise)
        return b(c_skip) * x_noisy + b(c_out) * F


class EDMDiffusionModel(pl.LightningModule):
    """EDM Lightning module: training (log-normal sigma + effective-target loss)
    and Heun sampling.

    Args:
        score_network: NCSNpp2D / ScoreNet etc., constructed with
            marginal_prob_std_fn=trivial_std (see top of file).
        sigma_data: per-site std of training data in [-1,1] norm space.
        P_mean, P_std: log-normal sigma sampling parameters for training.
            Karras defaults (-1.2, 1.2) are tuned for image data with
            sigma_data ~= 0.5; physics data with different sigma_data may
            want different values, but EDM is fairly robust.
        sigma_min, sigma_max: sigma range used in the Heun sampler schedule.
        norm_min, norm_max: data normalisation bounds (stored for inference).
    """

    def __init__(self, score_network, sigma_data, lr=1e-3, L=32,
                 P_mean=-1.2, P_std=1.2,
                 sigma_min=0.002, sigma_max=80.0,
                 ema_decay=0.9999, ema_start_epoch=0,
                 norm_min=None, norm_max=None):
        super().__init__()
        self.save_hyperparameters(ignore=['score_network'])
        self.precond = EDMPrecond(score_network, sigma_data)
        self.sigma_data = float(sigma_data)
        self.lr = lr
        self.L = L
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ema_start_epoch = ema_start_epoch
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.ema = ExponentialMovingAverage(self.precond.parameters(),
                                            decay=ema_decay)

    def forward(self, x, sigma):
        """Denoiser D_theta(x; sigma)."""
        return self.precond(x, sigma)

    def loss_fn(self, x):
        """Karras Eq. 8 with preconditioning absorbed: loss reduces to
        ||F_theta(c_in*x_noisy, c_noise) - (x - c_skip*x_noisy)/c_out||^2
        with NO explicit weight (lambda(sigma) * c_out(sigma)^2 = 1)."""
        B = x.shape[0]
        log_sigma = self.P_mean + self.P_std * torch.randn(B, device=x.device)
        sigma = log_sigma.exp()
        eps = torch.randn_like(x)
        b = lambda v: v.view(-1, 1, 1, 1)
        x_noisy = x + b(sigma) * eps

        c_in, c_out, c_skip, c_noise = self.precond._coeffs(sigma)
        F = self.precond.net(b(c_in) * x_noisy, c_noise)
        target = (x - b(c_skip) * x_noisy) / b(c_out)
        per_sample = (F - target).pow(2).mean(dim=(1, 2, 3))
        return per_sample.mean(), per_sample.detach(), sigma.detach()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, per_sample, sigma = self.loss_fn(x)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Loss bins by log sigma: small sigma = UV/high-SNR, large sigma = IR/low-SNR.
        log_sigma = sigma.log()
        bins = [
            (-1e9, -2.0, 'loss_high_snr'),   # sigma < e^-2 ~ 0.135
            (-2.0,  0.5, 'loss_mid'),         # 0.135 <= sigma < 1.65
            (0.5,  1e9,  'loss_low_snr'),    # sigma >= 1.65
        ]
        for lo, hi, name in bins:
            mask = (log_sigma >= lo) & (log_sigma < hi)
            if mask.any():
                self.log(name, per_sample[mask].mean(),
                         on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # EMA hooks ----------
    def on_fit_start(self):
        self.ema.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch >= self.ema_start_epoch:
            self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Handle torch.compile's _orig_mod. prefix mismatch (mirrors DiffusionModel).
        sd = checkpoint.get('state_dict', {})
        model_keys = set(self.state_dict().keys())
        ckpt_has_orig = any('._orig_mod.' in k for k in sd)
        model_has_orig = any('._orig_mod.' in k for k in model_keys)
        if ckpt_has_orig and not model_has_orig:
            sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
        elif not ckpt_has_orig and model_has_orig:
            sd = {k.replace('precond.net.', 'precond.net._orig_mod.', 1): v
                  for k, v in sd.items()}
        checkpoint['state_dict'] = sd

        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

    # ------- 2nd-order Heun sampler (Karras Algorithm 1) -------
    @torch.no_grad()
    def sample_heun(self, num_samples=64, num_steps=18, rho=7.0,
                    S_churn=0.0, S_min=0.0, S_max=float('inf'), S_noise=1.0,
                    progress=True):
        """Karras 2nd-order Heun sampler. Defaults match the EDM paper.

        With S_churn=0 this is fully deterministic (ODE). Set S_churn>0 for
        stochastic sampling.
        """
        device = self.device
        self.eval()
        with self.ema.average_parameters():
            step_indices = torch.arange(num_steps, device=device, dtype=torch.float32)
            sigma_max = float(self.sigma_max)
            sigma_min = float(self.sigma_min)
            inv_rho = 1.0 / rho
            sigmas = (sigma_max ** inv_rho
                      + step_indices / (num_steps - 1)
                        * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
            sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # final sigma = 0

            x = torch.randn(num_samples, 1, self.L, self.L, device=device) * sigmas[0]
            iterator = range(num_steps)
            if progress:
                iterator = tqdm(iterator, desc="Sampling (Heun)")
            for i in iterator:
                t_cur = sigmas[i]
                t_next = sigmas[i + 1]

                # Optional stochastic churn
                gamma = 0.0
                if S_churn > 0 and (S_min <= t_cur <= S_max):
                    gamma = min(S_churn / num_steps, np.sqrt(2.0) - 1.0)
                t_hat = t_cur + gamma * t_cur
                if gamma > 0:
                    x = x + torch.sqrt(t_hat ** 2 - t_cur ** 2) * S_noise * torch.randn_like(x)

                sigma_b = t_hat.expand(num_samples)
                denoised = self.precond(x, sigma_b)
                d_cur = (x - denoised) / t_hat
                x_next = x + (t_next - t_hat) * d_cur

                # 2nd-order correction (skip when t_next == 0)
                if i < num_steps - 1:
                    sigma_b = t_next.expand(num_samples)
                    denoised_next = self.precond(x_next, sigma_b)
                    d_next = (x_next - denoised_next) / t_next
                    x_next = x + (t_next - t_hat) * 0.5 * (d_cur + d_next)
                x = x_next

        return x

    def denormalize(self, x_norm):
        """Convert from [-1,1] back to original physical scale."""
        if self.norm_min is None or self.norm_max is None:
            return x_norm
        return (x_norm / 2 + 0.5) * (self.norm_max - self.norm_min) + self.norm_min
