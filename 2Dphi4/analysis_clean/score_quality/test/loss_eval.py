"""Score quality diagnostics for phi^4 score models.

This version evaluates the model score and the exact phi^4 score in the same
coordinate system.

Conventions
-----------
S[p] = sum_x [-2 k p_x sum_{+mu} p_{x+mu} + (1 - 2 l) p_x^2 + l p_x^4]
score = grad log p = - grad S

If ``phi_min`` and ``phi_max`` are given, the input field is assumed to be the
normalised variable x in [-1, 1], with p = (x + 1) / 2 * (phi_max - phi_min)
+ phi_min.  The returned exact score is then -dS/dx, not -dS/dp.

If ``phi_min`` or ``phi_max`` is None, the input field is treated as the
physical field p itself, and the exact score is -dS/dp.  This reproduces the
old unnormalised convention.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor
Number = Union[float, int]


def load_reference_data(
    data_path: str,
    num_samples: int,
    norm_min: Number,
    norm_max: Number,
    device: Union[str, torch.device],
    key: str = "cfgs",
    seed: int = 42,
) -> Tensor:
    """Load a fixed validation batch and normalise it to [-1, 1]."""
    import h5py

    with h5py.File(data_path, "r") as f:
        cfgs = np.asarray(f[key])

    rng = np.random.default_rng(seed)
    idx = rng.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs_norm = ((cfgs[idx] - norm_min) / (norm_max - norm_min) - 0.5) * 2.0

    x = torch.as_tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[1] != 1:
        # Interpret as (N, L, L, L) for 3D if the caller forgot the channel dim.
        # For ordinary 2D data this branch should not be triggered.
        pass
    return x


def _as_float(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().flatten()[0].cpu())
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0])
    return float(v)


def _extract_k_l(param=None, k=None, l=None) -> Tuple[float, float]:
    """Get scalar kappa and lambda from explicit values or a param tensor."""
    if k is not None and l is not None:
        return _as_float(k), _as_float(l)
    if param is None:
        raise ValueError("Need either param or both k and l.")

    if isinstance(param, torch.Tensor):
        p = param.detach()
        return _as_float(p.reshape(-1)[0]), _as_float(p.reshape(-1)[1])

    p = np.asarray(param)
    return _as_float(p.reshape(-1)[0]), _as_float(p.reshape(-1)[1])


def _time_like(x: Tensor, t_eval: Number) -> Tensor:
    return torch.full(
        (x.shape[0],),
        float(t_eval),
        dtype=x.dtype,
        device=x.device,
    )


def _call_score_model(score_model, param, x: Tensor, t: Tensor) -> Tensor:
    """Support both model(param, x, t) and model(x, t)."""
    if param is None:
        return score_model(x, t)

    try:
        return score_model(param, x, t)
    except TypeError as first_error:
        try:
            return score_model(x, t)
        except TypeError:
            raise first_error


def _field_from_input(
    phi: Tensor,
    phi_min: Optional[Number],
    phi_max: Optional[Number],
) -> Tuple[Tensor, float]:
    """Return physical field p and dp/dx scale for the input coordinate."""
    if phi.ndim < 4:
        raise ValueError("phi should have shape (N, 1, L, L) or (N, 1, L, L, L).")

    x = phi[:, 0]
    if phi_min is None or phi_max is None:
        return x, 1.0

    scale = (_as_float(phi_max) - _as_float(phi_min)) / 2.0
    shift = (_as_float(phi_max) + _as_float(phi_min)) / 2.0
    p = scale * x + shift
    return p, scale


def phi4_action(
    phi: Tensor,
    k: Number,
    l: Number,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
) -> Tensor:
    """Compute the per-sample phi^4 action.

    If phi_min/phi_max are provided, ``phi`` is the normalised field x.  If not,
    ``phi`` is the physical field p.
    """
    p, _ = _field_from_input(phi, phi_min, phi_max)
    k = _as_float(k)
    l = _as_float(l)

    neighbor_sum = sum(torch.roll(p, shifts=1, dims=d) for d in range(1, spatial_dims + 1))
    sum_axes = tuple(range(1, spatial_dims + 1))
    return torch.sum(
        -2.0 * k * p * neighbor_sum + (1.0 - 2.0 * l) * p.pow(2) + l * p.pow(4),
        dim=sum_axes,
    )


def phi4_grad_S(
    phi: Tensor,
    k: Number,
    l: Number,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
) -> Tensor:
    """Return dS/dx in the current input coordinate.

    If phi_min/phi_max are provided, x is the normalised field and the chain
    rule factor dp/dx = (phi_max - phi_min) / 2 is included.  If not, x is the
    physical field and the result is simply dS/dp.
    """
    p, scale = _field_from_input(phi, phi_min, phi_max)
    k = _as_float(k)
    l = _as_float(l)

    nb = sum(
        torch.roll(p, shifts=1, dims=d) + torch.roll(p, shifts=-1, dims=d)
        for d in range(1, spatial_dims + 1)
    )
    dS_dp = -2.0 * k * nb + 2.0 * (1.0 - 2.0 * l) * p + 4.0 * l * p.pow(3)
    return (dS_dp * scale).unsqueeze(1)


def phi4_score(
    phi: Tensor,
    k: Number,
    l: Number,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
) -> Tensor:
    """Return the exact target score, score = -dS/dx."""
    return -phi4_grad_S(phi, k, l, phi_min, phi_max, spatial_dims)


def get_drift(
    phi: Union[Tensor, np.ndarray],
    k: Number,
    l: Number,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
) -> Union[Tensor, np.ndarray]:
    """Backward-compatible exact score/drift helper.

    For torch input it returns a torch tensor.  For numpy input it returns numpy.
    """
    if isinstance(phi, np.ndarray):
        device = torch.device("cpu")
        x = torch.as_tensor(phi, dtype=torch.float32, device=device)
        out = phi4_score(x, k, l, phi_min, phi_max, spatial_dims)
        return out.cpu().numpy()
    return phi4_score(phi, k, l, phi_min, phi_max, spatial_dims)


def _per_sample_metrics(
    score: Tensor,
    target: Tensor,
    denom_eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    s_m = score.reshape(score.shape[0], -1)
    s_t = target.reshape(target.shape[0], -1)

    dot = (s_m * s_t).sum(dim=1)
    norm_m = torch.linalg.vector_norm(s_m, dim=1)
    norm_t = torch.linalg.vector_norm(s_t, dim=1)

    cos_sim = dot / (norm_m * norm_t + denom_eps)
    rel_mse = (s_m - s_t).pow(2).sum(dim=1) / (s_t.pow(2).sum(dim=1) + denom_eps)
    mag_ratio = norm_m / (norm_t + denom_eps)
    return cos_sim, rel_mse, mag_ratio


def _mean_or_mean_sem(v: Tensor, return_sem: bool):
    mean = v.mean().item()
    if not return_sem:
        return mean
    n = v.numel()
    sem = (v.std(unbiased=True) / math.sqrt(n)).item() if n > 1 else 0.0
    return mean, sem


def score_quality_each(
    phi_torch: Tensor,
    score_model,
    param=None,
    eps: Number = 1e-4,
    k: Optional[Number] = None,
    l: Optional[Number] = None,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
    grad_S_fn: Optional[Callable[[Tensor], Tensor]] = None,
    denom_eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return per-sample (cos_sim, relative_mse, magnitude_ratio)."""
    x = phi_torch
    t = _time_like(x, eps)

    with torch.no_grad():
        score = _call_score_model(score_model, param, x, t)
        if grad_S_fn is None:
            kk, ll = _extract_k_l(param, k, l)
            target = phi4_score(x, kk, ll, phi_min, phi_max, spatial_dims)
        else:
            target = -grad_S_fn(x)
        return _per_sample_metrics(score, target, denom_eps=denom_eps)


def consine_similarity(
    phi_torch: Tensor,
    score_model,
    param=None,
    eps: Number = 1e-4,
    k: Optional[Number] = None,
    l: Optional[Number] = None,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
    grad_S_fn: Optional[Callable[[Tensor], Tensor]] = None,
    return_sem: bool = False,
):
    """Mean cosine similarity.  Kept misspelled for old scripts."""
    cos_sim, _, _ = score_quality_each(
        phi_torch, score_model, param, eps, k, l, phi_min, phi_max,
        spatial_dims, grad_S_fn,
    )
    return _mean_or_mean_sem(cos_sim, return_sem)


def cosine_similarity(*args, **kwargs):
    """Correctly spelled alias for consine_similarity."""
    return consine_similarity(*args, **kwargs)


def relative_MSE(
    phi_torch: Tensor,
    score_model,
    param=None,
    eps: Number = 1e-4,
    k: Optional[Number] = None,
    l: Optional[Number] = None,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
    grad_S_fn: Optional[Callable[[Tensor], Tensor]] = None,
    return_sem: bool = False,
):
    """Mean relative MSE against the exact phi^4 score."""
    _, rel_mse, _ = score_quality_each(
        phi_torch, score_model, param, eps, k, l, phi_min, phi_max,
        spatial_dims, grad_S_fn,
    )
    return _mean_or_mean_sem(rel_mse, return_sem)


def relative_mse(*args, **kwargs):
    return relative_MSE(*args, **kwargs)


def magnitude_ratio(
    phi_torch: Tensor,
    score_model,
    param=None,
    eps: Number = 1e-4,
    k: Optional[Number] = None,
    l: Optional[Number] = None,
    phi_min: Optional[Number] = None,
    phi_max: Optional[Number] = None,
    spatial_dims: int = 2,
    grad_S_fn: Optional[Callable[[Tensor], Tensor]] = None,
    return_sem: bool = False,
):
    """Mean norm ratio ||score_model|| / ||score_exact||."""
    _, _, mag_ratio = score_quality_each(
        phi_torch, score_model, param, eps, k, l, phi_min, phi_max,
        spatial_dims, grad_S_fn,
    )
    return _mean_or_mean_sem(mag_ratio, return_sem)


def one_score_quality(
    model,
    grad_S_fn: Callable[[Tensor], Tensor],
    x_ref: Tensor,
    t_eval: Number,
    denom_eps: float = 1e-12,
):
    """Return {'metric': (mean, SEM)} at current model weights.

    This matches the interface in the snippet you sent, using model(x, t) and
    s_true = -grad_S_fn(x).
    """
    device = getattr(model, "device", x_ref.device)
    x_ref = x_ref.to(device)
    batch_t = torch.full((x_ref.shape[0],), float(t_eval), device=device, dtype=x_ref.dtype)

    with torch.no_grad():
        s_model = model(x_ref, batch_t)
        s_true = -grad_S_fn(x_ref)
        cos_sim, rel_mse, mag_ratio = _per_sample_metrics(s_model, s_true, denom_eps)

    out = {}
    for name, value in (
        ("cos_sim", cos_sim),
        ("rel_mse", rel_mse),
        ("mag_ratio", mag_ratio),
    ):
        out[name] = _mean_or_mean_sem(value, return_sem=True)
    return out
