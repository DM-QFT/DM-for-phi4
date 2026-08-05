"""Shared loaders/estimators for betaconfig verification scripts."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT))
OUT = HERE.parent / "results" / "verification"      # 产物出目录，与代码分离
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

from phi4_action import phi4_beta_conf, phi4_beta_conf_terms  # noqa: E402


def load_hmc(k, L, n=10240):
    with h5py.File(ROOT / "2Dphi4" / "trainingdata" /
                   f"cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2", "r") as f:
        cfgs = np.array(f["cfgs"][:n], dtype=np.float64)
    assert cfgs.shape[1:] == (L, L)
    return cfgs


def beta_terms_phys(p, k, l, chunk=2048):
    """(term1, term2) per config for physical-space cfgs p (N,L,L) torch."""
    t1, t2 = [], []
    for i in range(0, p.shape[0], chunk):
        # phi_min=-1, phi_max=1 makes the internal normalisation map the identity
        a, b = phi4_beta_conf_terms(p[i:i + chunk, None], k, l, -1.0, 1.0)
        t1.append(a)
        t2.append(b)
    return torch.cat(t1), torch.cat(t2)


def beta_conf_phys(p, k, l, chunk=2048):
    out = [phi4_beta_conf(p[i:i + chunk, None], k, l, -1.0, 1.0)
           for i in range(0, p.shape[0], chunk)]
    return torch.cat(out)


def blocked_mean_sem(x, nb=64):
    """Mean and blocked SEM; blocks tame HMC autocorrelation."""
    x = np.asarray(x, dtype=np.float64)
    n = (len(x) // nb) * nb
    b = x[:n].reshape(nb, -1).mean(axis=1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb))


def grad_phys(p, k, l):
    """g_x = -2k*sum_nb(phi) + 2(1-2l)phi + 4l phi^3 for (N,L,L) torch."""
    nb = (torch.roll(p, 1, 1) + torch.roll(p, -1, 1)
          + torch.roll(p, 1, 2) + torch.roll(p, -1, 2))
    return -2 * k * nb + 2 * (1 - 2 * l) * p + 4 * l * p ** 3


def action_phys(p, k, l):
    """S per config for (N,L,L) torch."""
    nb = torch.roll(p, 1, 1) + torch.roll(p, 1, 2)
    return (-2 * k * p * nb + (1 - 2 * l) * p ** 2 + l * p ** 4).sum(dim=(1, 2))


def phys_obs(p):
    """Z2-even observables of (N,L,L) torch cfgs; per-config arrays for jackknife."""
    N, L, _ = p.shape
    m = p.mean(dim=(1, 2))
    fk = torch.fft.fft2(p.double())
    # G(k) = <|phi_k|^2>/V; average the two k_min orientations for isotropy
    Fk1 = (fk[:, 0, 1].abs() ** 2 + fk[:, 1, 0].abs() ** 2).cpu().numpy() / (2 * L * L)
    Fk2 = (fk[:, 0, 2].abs() ** 2 + fk[:, 2, 0].abs() ** 2).cpu().numpy() / (2 * L * L)
    return dict(
        phi2=(p ** 2).mean(dim=(1, 2)).cpu().numpy(),
        m2=(m ** 2).cpu().numpy(),
        m4=(m ** 4).cpu().numpy(),
        absm=m.abs().cpu().numpy(),
        Fk1=Fk1, Fk2=Fk2, L=L,
    )


def derived_obs(means, L):
    chi = L * L * means["m2"]
    # broken phase: <m^2> is condensate-dominated; xi needs the connected part
    chi_c = L * L * (means["m2"] - means["absm"] ** 2)
    F = means["Fk1"]
    xi = np.sqrt(max(chi_c / F - 1.0, 0.0)) / (2 * np.sin(np.pi / L))
    return dict(phi2=means["phi2"], chi=chi, chi_c=chi_c, xi=xi, Fk1=F,
                Fk2=means["Fk2"], binder=means["m4"] / means["m2"] ** 2)


def jackknife_obs(raw, nb=32):
    """Blocked jackknife over configs for all derived observables."""
    L = raw["L"]
    keys = [k for k in raw if k != "L"]
    n = (len(raw["m2"]) // nb) * nb
    blocks = {k: raw[k][:n].reshape(nb, -1).mean(axis=1) for k in keys}
    full = {k: blocks[k].mean() for k in keys}
    center = derived_obs(full, L)
    jk = []
    for i in range(nb):
        m = {k: (nb * full[k] - blocks[k][i]) / (nb - 1) for k in keys}
        jk.append(derived_obs(m, L))
    out = {}
    for k in center:
        vals = np.array([j[k] for j in jk])
        out[k] = (center[k], float(np.sqrt((nb - 1) / nb) * np.linalg.norm(vals - vals.mean())))
    return out


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def em_reverse(model, n, steps, schedule, eps, bf16=True, seed=0):
    """EM reverse SDE with the model's EMA score; returns normalised samples."""
    dev = model.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ts = model._build_time_steps(steps, eps, schedule, dev)
    dts = ts[:-1] - ts[1:]
    with model.ema.average_parameters():
        init_std = model.marginal_prob_std_fn(torch.tensor(1.0, device=dev))
        x = torch.randn(*model._sample_shape(n), device=dev) * init_std
        bt = torch.empty(n, device=dev)
        ctx = torch.autocast(dev.type, dtype=torch.bfloat16) if bf16 else _NullCtx()
        with ctx:
            for i in range(steps):
                bt.fill_(ts[i].item())
                g = model.diffusion_coeff_fn(ts[i])
                mean_x = x + g ** 2 * model(x, bt) * dts[i]
                x = mean_x + g * torch.sqrt(dts[i]) * torch.randn_like(x)
    # production sampler returns mean_x: the last kick is not part of the sample
    return mean_x.float()


def norm_to_phys(x, norm_min, norm_max):
    return (x + 1) / 2 * (norm_max - norm_min) + norm_min
