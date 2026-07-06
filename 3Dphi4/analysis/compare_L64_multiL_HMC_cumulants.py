"""
3D phi^4 L=64 multi-L DM samples vs HMC reference.

Compares three multi-L training configurations against HMC, all using
EM/linear schedule, ep=10000, steps=2000:
  - multiL_4-8       (cross-L from L_train=8  to L=64)
  - multiL_4-8-16    (cross-L from L_train=16 to L=64)
  - multiL_4-8-16-32 (cross-L from L_train=32 to L=64)

For kappa=0.2 the multiL_4-8-16-32 sigma=400 variant is used (as in
export_L64_crossL_3d_propagator_k018_k02.py); the smaller multi-L runs
have only the default sigma=800 dir at kappa=0.2.

Pipeline (all observables on per-cfg magnetisation M = <phi>_V):
  1. Load raw cfgs (HMC and DM, in physical units).
  2. Z_2-symmetrise in raw space:  cat([s, -s]).
     Applied to BOTH HMC and DM. This is the physical phi -> -phi symmetry
     of the action; after concatenation <M> = 0 exactly.
  3. Compute statistics on the raw, Z_2-symmetrised cfgs (no pm1 transform).

Statistics:
  - Order parameter:        <|M|>                              (DM.tex eq.)
  - Magnetic susceptibility: chi = V*(<|M|^2> - <|M|>^2)        (DM.tex eq.)
  - Binder cumulant:         U_4 = 1 - <M^4>/(3<M^2>^2)         (DM.tex eq.)
  - Connected cumulants kappa_2..kappa_8 via the recursion of
    DiaaEddinH/On-learning-higher-order-cumulants-in-diffusion-models
    (bin/utils.py::nth_cumulants):
        kappa_n = mu_n - sum_{i=1..n-1} C(n-1, i-1) * kappa_i * mu_{n-i}
    Z_2 symmetrisation in raw space ensures <M> = 0 exactly, so the
    recursion holds without manual centering. Odd cumulants
    (kappa_3, kappa_5, kappa_7) vanish identically by Z_2; only
    kappa_2, kappa_4, kappa_6, kappa_8 carry physics.

Bootstrap errors over cfg index (n_boot=400 by default).

Run from 3Dphi4/:
  python analysis/compare_L64_multiL_HMC_cumulants.py
  python analysis/compare_L64_multiL_HMC_cumulants.py --kappas 0.18 0.1923 0.2
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import csv
import json
import numpy as np
import h5py

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _layout import to_NLLL


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_dm_3d(path: Path) -> np.ndarray:
    return to_NLLL(np.load(path)).astype(np.float64)


def load_hmc_3d(path: Path, max_n: int | None = None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    cfgs = to_NLLL(cfgs)
    return cfgs if max_n is None else cfgs[:max_n]


def hmc_path(k: float, lam: float, L: int = 64) -> Path:
    p = Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l={lam}_{L}^3.jld2")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def hmc_norm_range(k: float, lam: float, L: int = 64) -> tuple[float, float]:
    p = Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l={lam}_{L}^3.jld2.norm.json")
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.loads(p.read_text())
    return float(d["cfgs_min"]), float(d["cfgs_max"])


def normalize_pm1(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def filter_oor(cfgs: np.ndarray, lo: float, hi: float,
               max_abs_M: float | None = None
               ) -> tuple[np.ndarray, int, int]:
    """Drop DM failure-mode cfgs by two physically motivated criteria:

      (a) Any spatial site lies outside the HMC training range [lo, hi]
          (equivalently outside [-1, +1] in pm1-normalised space) — the
          reverse-SDE drifted into a region the network never saw during
          training, so the score is pure extrapolation.

      (b) (Optional, when ``max_abs_M`` is given) The per-cfg magnetisation
          |M| = |<phi>_V| exceeds the maximum |M| in the HMC ensemble —
          the order parameter has fallen outside its physical range,
          typically an artefact of late-training Z_2 mode collapse on a
          single broken-phase branch.

    Returns ``(filtered_cfgs, n_dropped_total, n_dropped_amplitude_only)``.
    """
    spatial_axes = tuple(range(1, cfgs.ndim))
    in_amp = (cfgs.min(axis=spatial_axes) >= lo) & \
             (cfgs.max(axis=spatial_axes) <= hi)
    keep = in_amp.copy()
    if max_abs_M is not None:
        in_M = np.abs(cfgs.mean(axis=spatial_axes)) <= max_abs_M
        keep = keep & in_M
    return cfgs[keep], int((~keep).sum()), int((~in_amp).sum())


# Cross-L L=64 sample paths. Two filename conventions exist (kappa=0.1923 was
# saved with a different tag than 0.18 / 0.2). Each returned path is a
# candidate list; we take the first that exists.
def dm_path_multiL(L_train: int, k: float, lam: float,
                   schedule: str, num_steps: int, ep: int) -> Path:
    L_label = {8: "4-8", 16: "4-8-16", 32: "4-8-16-32"}[L_train]
    # sigma override for the L_train=32 multi-L: tuned per kappa via the
    # std(t=1) ~ 2 D_max empirical rule. The default ncsnpp dir uses sigma=800.
    sigma_tag = ""
    if L_train == 32:
        if abs(k - 0.2)  < 1e-9: sigma_tag = "_sigma400"
        if abs(k - 0.18) < 1e-9: sigma_tag = "_sigma200"
    base = Path(f"runs/phi4_3d_Lmulti{L_label}_k{k}_l{lam}_ncsnpp{sigma_tag}")
    crossL = base / "data_crossL"

    # Filename conventions seen in repo:
    #  k=0.1923 (tagged):  samples_crossL64_multiL{label}_ep{ep}_em_{sched}_steps{S}_512.npy
    #                      and for L_train=32 only: samples_crossL64_multiL_ep{ep}_em_{sched}_steps{S}_512.npy
    #  k=0.18 / 0.2:       samples_crossL_train{L_train}_sample64_em_{sched}_steps{S}_ep{ep}.npy
    cands = [
        crossL / f"samples_crossL_train{L_train}_sample64_em_{schedule}_steps{num_steps}_ep{ep}.npy",
        crossL / f"samples_crossL64_multiL{L_label}_ep{ep}_em_{schedule}_steps{num_steps}_512.npy",
        crossL / f"samples_crossL64_multiL_ep{ep}_em_{schedule}_steps{num_steps}_512.npy",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of: {[str(p) for p in cands]}")


# In-distribution L=64 single-L samples. Available for the three kappas where
# a sigma-tuned L=64 model has been trained
# (kappa=0.18 sigma=2000, kappa=0.1923 sigma=2048, kappa=0.2 sigma=2760), all
# at epoch=20000. The filename matches sample_phi4.py, which uses
# tag = f"em_{schedule}_steps{N}_epoch={ep}" since the linear-schedule refactor.
_L64_SINGLE: dict[float, tuple[int, int]] = {
    0.18:   (2000, 20000),
    0.1923: (2048, 20000),
    0.2:    (2760, 20000),
}


def dm_path_L64single(k: float, lam: float, schedule: str,
                      num_steps: int) -> tuple[Path, int] | None:
    """Return (path, ep) for the L=64 single-L sample, or None if not configured.

    sample_phi4.py saves to runs/{dir}/data/samples_{method}_steps{N}_{ep}.npy
    (no schedule tag in filename); we tolerate a couple of historical variants
    too (samples_em_linear_steps... and samples_em_steps..._epoch=...).
    """
    cfg = next(((s, e) for kk, (s, e) in _L64_SINGLE.items()
                if abs(k - kk) < 1e-9), None)
    if cfg is None:
        return None
    sigma, ep = cfg
    data = Path(f"runs/phi4_3d_L64_k{k}_l{lam}_ncsnpp_sigma{sigma}/data")
    cands = [
        data / f"samples_em_steps{num_steps}_{ep}.npy",                # current sample_phi4.py
        data / f"samples_em_{schedule}_steps{num_steps}_epoch={ep}.npy",
        data / f"samples_em_steps{num_steps}_epoch={ep}.npy",
    ]
    for p in cands:
        if p.exists():
            return p, ep
    return None


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def magnetisation(cfgs: np.ndarray) -> np.ndarray:
    """M_i = mean of phi over the volume of cfg i. Shape: (N,)."""
    return cfgs.mean(axis=tuple(range(1, cfgs.ndim)))


def per_site_field(cfgs: np.ndarray) -> np.ndarray:
    """Flatten (N_cfg, L, L, L) into (N_cfg*V,), treating every (cfg, site)
    pair as one sample of the single-site field phi(x).

    Following Aarts et al. (arXiv:2410.21212v2), eq. (5.11):
    by translation invariance, moments mu_n(x) = E[phi^n(x)] are
    x-independent and can be estimated by pooling over all spacetime points
    and configurations. The resulting cumulants kappa_n^{(site)} describe
    the single-point field distribution and are O(1) numerically (very
    different from kappa_n(M_bar), which is suppressed by 1/V^{n-1}).
    """
    return cfgs.reshape(-1)


def nth_cumulants(data: np.ndarray, order: int) -> np.ndarray:
    """
    Connected cumulants kappa_1..kappa_order of `data`, returned as a 1D array.

    Standard moment-to-cumulant relation expressed in *central* moments:
        kappa_n = mu_n - sum_{i=2..n-1} C(n-1, i-1) * kappa_i * mu_{n-i},
    valid for any input regardless of whether <data>=0; the cumulant
    kappa_n (n>=2) is invariant under translation, so we work with the
    centred series Y = X - <X> for which kappa_1(Y) = 0 and the i=1 term
    in the recursion vanishes by construction.

    The reference (DiaaEddinH/On-learning-higher-order-cumulants-in-diffusion-
    models, bin/utils.py::nth_cumulants) writes the same recursion but lets
    i run from 1, inserting cum[0] = <X> into the i=1 term. That introduces
    a spurious -<X>*mu_{n-1} contribution whenever <X> != 0; the bias is 0
    on Z_2-symmetric input. We skip i=1 explicitly so the function is
    correct on any input.
    """
    from math import comb
    cum = [float(np.mean(data))]              # kappa_1 = <X>
    dev = data - cum[0]
    mu = {k: float(np.mean(dev ** k)) for k in range(1, order + 1)}   # mu_1=0
    for n in range(2, order + 1):
        kappa = mu[n]
        for i in range(2, n):                 # i=1 term skipped (kappa_1(Y)=0)
            kappa -= comb(n - 1, i - 1) * cum[i - 1] * mu[n - i]
        cum.append(kappa)
    return np.array(cum)


def stat_block(M: np.ndarray, V: int) -> dict[str, float]:
    """Single point estimates from the per-cfg magnetisation M of shape (N,).

    Computed in raw (physical) field space. After cat([phi, -phi])
    Z_2-symmetrisation, <M> = 0 exactly and the odd cumulants kappa_3,
    kappa_5, kappa_7 vanish identically; this docstring previously
    referred to normalised space but the pipeline writes raw cfgs.
    """
    abs_M = float(np.mean(np.abs(M)))
    M2 = float(np.mean(M ** 2))                  # <M^2> = <|M|^2>
    M4 = float(np.mean(M ** 4))
    chi = V * (M2 - abs_M ** 2)                  # DM.tex definition
    U4 = 1.0 - M4 / (3.0 * M2 ** 2) if M2 > 0 else float("nan")
    cums = nth_cumulants(M, order=8)
    return {
        "mean_M": float(np.mean(M)),
        "abs_M":  abs_M,
        "chi":    chi,
        "U4":     U4,
        "k2": cums[1], "k3": cums[2], "k4": cums[3], "k5": cums[4],
        "k6": cums[5], "k7": cums[6], "k8": cums[7],
    }


def _moments_to_cumulants_from_raw(m: np.ndarray) -> np.ndarray:
    """Convert raw moments m[k] = <X^k> for k=1..8 (1-indexed via m[0]=<X>) to
    cumulants kappa_1..kappa_8 via the standard moment-cumulant recursion
    on RAW moments:
        kappa_n = m_n - sum_{i=1..n-1} C(n-1, i-1) kappa_i m_{n-i},  m_0 = 1.
    Unlike the central-moment form, this works on shifted data (kappa_1 is
    correctly accounted for as <X>, not 0).
    """
    from math import comb
    cum = [m[0]]                           # kappa_1 = <X>
    for n in range(2, 9):
        kappa = m[n - 1]                   # m_n
        for i in range(1, n):
            kappa -= comb(n - 1, i - 1) * cum[i - 1] * m[n - i - 1]
        cum.append(kappa)
    return np.array(cum)


def bootstrap_stats_per_site(cfgs: np.ndarray, n_boot: int = 400, seed: int = 0,
                             n_raw: int | None = None):
    """Bootstrap means/errors of single-site cumulants kappa_n^{(site)}.

    Pre-computes per-cfg spatial averages of phi^k for k=1..8, then
    bootstrap-resamples cfg indices and averages those (linear combination)
    to build the pooled raw moments. Equivalent to pooling every
    (resampled cfg, site) and computing moments — exact, just much faster.
    Paired-bootstrap mode (n_raw given) keeps the Z_2 partner.
    """
    rng = np.random.default_rng(seed)
    N_cfg = cfgs.shape[0]
    if n_raw is not None and 2 * n_raw != N_cfg:
        raise ValueError(
            f"paired bootstrap: expected cfgs of length 2*n_raw={2*n_raw}, "
            f"got {N_cfg}")
    flat = cfgs.reshape(N_cfg, -1)             # (N_cfg, V) — view
    # per-cfg <phi^k>_x for k=1..8 and <|phi|>_x.
    # Compute one cfg at a time to avoid allocating a (N_cfg, V)-sized
    # temporary for every power (would peak ~ 8 * N_cfg * V * 8 bytes =
    # tens of GB on the full HMC ensemble).
    V = flat.shape[1]
    pcm = np.empty((N_cfg, 8), dtype=np.float64)
    abs_pcm = np.empty(N_cfg, dtype=np.float64)
    for i in range(N_cfg):
        x = flat[i]                            # (V,)
        x2 = x * x
        x4 = x2 * x2
        pcm[i, 0] = x.mean()
        pcm[i, 1] = x2.mean()
        pcm[i, 2] = (x2 * x).mean()
        pcm[i, 3] = x4.mean()
        pcm[i, 4] = (x4 * x).mean()
        pcm[i, 5] = (x4 * x2).mean()
        pcm[i, 6] = (x4 * x2 * x).mean()
        pcm[i, 7] = (x4 * x4).mean()
        abs_pcm[i] = np.abs(x).mean()

    keys = ["mean_phi", "abs_phi", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
            "U4_site"]
    accum = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        if n_raw is not None:
            idx_raw = rng.integers(0, n_raw, size=n_raw)
            idx = np.concatenate([idx_raw, idx_raw + n_raw])
        else:
            idx = rng.integers(0, N_cfg, size=N_cfg)
        m = pcm[idx].mean(axis=0)                  # pooled raw moments m_1..m_8
        cums = _moments_to_cumulants_from_raw(m)
        accum["mean_phi"][b] = m[0]
        accum["abs_phi"][b]  = float(abs_pcm[idx].mean())
        accum["k2"][b] = cums[1]; accum["k3"][b] = cums[2]
        accum["k4"][b] = cums[3]; accum["k5"][b] = cums[4]
        accum["k6"][b] = cums[5]; accum["k7"][b] = cums[6]
        accum["k8"][b] = cums[7]
        accum["U4_site"][b] = (1.0 - m[3] / (3.0 * m[1] ** 2)) if m[1] > 0 else float("nan")
    mean = {k: float(np.mean(accum[k])) for k in keys}
    err  = {k: float(np.std(accum[k]))  for k in keys}
    return mean, err


def bootstrap_stats(M: np.ndarray, V: int, n_boot: int = 400, seed: int = 0,
                    n_raw: int | None = None):
    """Bootstrap means/errors of stat_block on M (shape (N,)).

    If `n_raw` is given, M must be the Z_2-symmetrised array of shape
    (2*n_raw,) with originals at indices [0, n_raw) and the negated copies
    at [n_raw, 2*n_raw).  The resampler then draws n_raw indices with
    replacement from the originals and emits both halves, which keeps
    every bootstrap sample exactly Z_2-symmetric (<M_b> = 0 strictly,
    odd cumulants vanish for each draw).  This eliminates the residual
    Z_2-breaking noise that the default independent-index bootstrap leaves
    on kappa_3, kappa_5, kappa_7 and slightly shrinks the bootstrap
    variance of every observable.

    If `n_raw` is None, falls back to the standard independent-index
    bootstrap (legacy behaviour, retained for cross-checks).
    """
    rng = np.random.default_rng(seed)
    N = len(M)
    keys = ["mean_M", "abs_M", "chi", "U4",
            "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
    accum = {k: np.empty(n_boot) for k in keys}
    if n_raw is not None and 2 * n_raw != N:
        raise ValueError(
            f"paired bootstrap: expected M of length 2*n_raw={2*n_raw}, got {N}")
    for b in range(n_boot):
        if n_raw is not None:
            idx_raw = rng.integers(0, n_raw, size=n_raw)
            idx = np.concatenate([idx_raw, idx_raw + n_raw])
        else:
            idx = rng.integers(0, N, size=N)
        s = stat_block(M[idx], V)
        for kname in keys:
            accum[kname][b] = s[kname]
    mean = {k: float(np.mean(accum[k])) for k in keys}
    err  = {k: float(np.std(accum[k]))  for k in keys}
    return mean, err


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fmt(v, e, w=10, p=4):
    if not np.isfinite(v):
        return f"{'nan':>{w*2+1}}"
    return f"{v:>{w}.{p}f}±{e:<{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+",
                    default=[0.18, 0.1923, 0.2])
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--ep",  type=int,   default=10000)
    ap.add_argument("--num_steps", type=int, default=2000)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hmc_max", type=int, default=None,
                    help="cap on HMC cfgs (default: use all)")
    ap.add_argument("--paired_bootstrap", action="store_true",
                    help="resample raw cfg indices and reuse both halves of "
                         "the Z_2-symmetrised array (default: independent-index "
                         "bootstrap on the full 2*N_raw array)")
    ap.add_argument("--per_site", action="store_true",
                    help="ALSO print single-site cumulants kappa_n^{(site)} "
                         "following Aarts et al. (arXiv:2410.21212v2) eq. (5.11): "
                         "pool every (cfg, site) into one sample of phi(x). "
                         "Bootstrap still resamples cfgs (the independent unit).")
    ap.add_argument("--no_filter_dm_oor", action="store_true",
                    help="disable the default DM out-of-range filter (drops "
                         "any DM cfg with a site outside the HMC training "
                         "range [lo, hi]); HMC ensembles are never filtered.")
    args = ap.parse_args()

    L = 64
    V = L ** 3
    schedule = "linear"

    print(f"\n3D phi^4 L={L}  multi-L {{4-8, 4-8-16, 4-8-16-32}} DM  vs  HMC")
    print(f"  kappas={args.kappas}  lam={args.lam}  ep={args.ep}  "
          f"schedule={schedule}  steps={args.num_steps}  n_boot={args.n_boot}")
    print(f"  Z_2 symmetrisation in RAW space (cat[s, -s]),"
          f" applied to BOTH HMC and DM (physical phi -> -phi)")
    print(f"  bootstrap mode: "
          f"{'PAIRED (n_raw indices, both halves used)' if args.paired_bootstrap else 'independent index'}")
    print(f"  DM filter:      "
          f"{'OFF (--no_filter_dm_oor)' if args.no_filter_dm_oor else 'ON (drops cfgs with phi outside HMC range OR |M| beyond HMC max)'}\n")

    rows: list[dict] = []
    for k in args.kappas:
        try:
            f_hmc = hmc_path(k, args.lam, L)
            lo, hi = hmc_norm_range(k, args.lam, L)
        except FileNotFoundError as e:
            print(f"[skip] kappa={k}: {e}")
            continue

        hmc = load_hmc_3d(f_hmc, max_n=args.hmc_max)
        # Pipeline: Z_2 symmetrise in raw space (physical phi -> -phi).
        hmc_z2 = np.concatenate([hmc, -hmc], axis=0)

        # Reference scale for the |M| failure-mode filter applied to DM cfgs.
        max_abs_M_hmc = float(np.abs(hmc.mean(axis=tuple(range(1, hmc.ndim)))).max())

        # Determine DM cfg count (all multi-L runs use the same N=512 here, but
        # detect from the first available file to stay general).
        dm_loaded: list[tuple[int, np.ndarray, str]] = []
        for L_train in (8, 16, 32):
            try:
                f_dm = dm_path_multiL(L_train, k, args.lam,
                                      schedule, args.num_steps, args.ep)
            except FileNotFoundError as e:
                print(f"[skip] kappa={k}  multiL_4..{L_train}: {e}")
                continue
            dm_raw = load_dm_3d(f_dm)
            if not args.no_filter_dm_oor:
                dm_filt, n_drop, n_drop_amp = filter_oor(
                    dm_raw, lo, hi, max_abs_M=max_abs_M_hmc)
                if n_drop:
                    print(f"  [filter] multiL_4..{L_train}: dropped {n_drop}/{dm_raw.shape[0]} "
                          f"({n_drop_amp} by phi-OOR + {n_drop - n_drop_amp} by |M|>{max_abs_M_hmc:.3f})")
                dm_raw = dm_filt
            dm_loaded.append((L_train, dm_raw, str(f_dm)))
        dm_N_raw = dm_loaded[0][1].shape[0] if dm_loaded else 0

        # HMC subsampled to a fixed reference count (default 512) — matches the
        # *unfiltered* DM cfg count, so the same N is used across all kappas
        # for table consistency, independent of how many cfgs the OOR filter
        # dropped from any particular DM source.
        sources: list[tuple[str, np.ndarray, str]] = [("HMC", hmc_z2, str(f_hmc))]
        sub_N = 512
        if sub_N < hmc.shape[0]:
            rng_sub = np.random.default_rng(args.seed + 1000)
            idx = rng_sub.choice(hmc.shape[0], sub_N, replace=False)
            hmc_sub = hmc[idx]
            hmc_sub_z2 = np.concatenate([hmc_sub, -hmc_sub], axis=0)
            sources.append(
                (f"HMC_subN{sub_N}", hmc_sub_z2, str(f_hmc) + f" (subsample {sub_N})")
            )
        for L_train, dm, path_str in dm_loaded:
            dm_z2 = np.concatenate([dm, -dm], axis=0)
            label = {8: "DM_multiL4-8", 16: "DM_multiL4-8-16",
                     32: "DM_multiL4-8-16-32"}[L_train]
            sources.append((label, dm_z2, path_str))

        # In-distribution L=64 single-L (only at kappa=0.1923, 0.2).
        single = dm_path_L64single(k, args.lam, schedule, args.num_steps)
        if single is not None:
            f_dm64, ep_dm64 = single
            dm64 = load_dm_3d(f_dm64)
            if not args.no_filter_dm_oor:
                dm64_filt, n_drop, n_drop_amp = filter_oor(
                    dm64, lo, hi, max_abs_M=max_abs_M_hmc)
                if n_drop:
                    print(f"  [filter] DM_L64single ep={ep_dm64}: dropped {n_drop}/{dm64.shape[0]} "
                          f"({n_drop_amp} by phi-OOR + {n_drop - n_drop_amp} by |M|>{max_abs_M_hmc:.3f})")
                dm64 = dm64_filt
            dm64_z2 = np.concatenate([dm64, -dm64], axis=0)
            sources.append(
                (f"DM_L64single_ep{ep_dm64}", dm64_z2,
                 str(f_dm64) + f"  (in-distribution L=64, ep={ep_dm64})")
            )

        print(f"--- kappa={k}, lam={args.lam}  (raw range from HMC: [{lo:.4f}, {hi:.4f}]) ---")
        for name, cfgs, path in sources:
            print(f"  {name:>20}  N={cfgs.shape[0]:>5}   {path}")
        # block 1: <|M|>, chi, U4 (DM.tex definitions)
        h1 = (f"  {'source':>20}  {'<|M|>':>23}  "
              f"{'chi=V(<|M|^2>-<|M|>^2)':>26}  {'U4':>23}")
        print(h1); print("  " + "-" * (len(h1) - 2))
        block_rows = []
        for name, cfgs, path in sources:
            M = magnetisation(cfgs)
            n_raw = cfgs.shape[0] // 2 if args.paired_bootstrap else None
            mean, err = bootstrap_stats(M, V, n_boot=args.n_boot,
                                        seed=args.seed, n_raw=n_raw)
            block_rows.append((name, cfgs.shape[0], mean, err))
            print(f"  {name:>20}  "
                  f"{fmt(mean['abs_M'], err['abs_M'], 10, 5)}  "
                  f"{fmt(mean['chi'],   err['chi'],   12, 4)}  "
                  f"{fmt(mean['U4'],    err['U4'],    10, 5)}")

        # block 2: cumulants k2..k8 (kappa_n of the per-cfg M = <phi>_V)
        print()
        h2 = (f"  {'source':>20}  {'k2':>23}  {'k3':>23}  {'k4':>23}  "
              f"{'k5':>23}  {'k6':>23}  {'k7':>23}  {'k8':>23}")
        print(h2); print("  " + "-" * (len(h2) - 2))
        for name, N, mean, err in block_rows:
            print(f"  {name:>20}  "
                  f"{fmt(mean['k2'], err['k2'], 10, 6)}  "
                  f"{fmt(mean['k3'], err['k3'], 10, 6)}  "
                  f"{fmt(mean['k4'], err['k4'], 10, 6)}  "
                  f"{fmt(mean['k5'], err['k5'], 10, 6)}  "
                  f"{fmt(mean['k6'], err['k6'], 10, 6)}  "
                  f"{fmt(mean['k7'], err['k7'], 10, 6)}  "
                  f"{fmt(mean['k8'], err['k8'], 10, 6)}")
            rows.append(dict(kappa=k, lo=lo, hi=hi, source=name, N=N, V=V,
                             **mean,
                             **{f"d_{kk}": err[kk] for kk in mean}))
        print()

        # Optional: single-site cumulants (Aarts et al. convention).
        if args.per_site:
            print(f"  --- per-site cumulants  kappa_n^(site)  "
                  f"[Aarts et al. 2410.21212v2, eq. (5.11)] ---")
            h3 = (f"  {'source':>20}  {'<|phi|>':>14}  {'k2(phi)':>14}  "
                  f"{'k4(phi)':>14}  {'k6(phi)':>14}  {'k8(phi)':>14}  "
                  f"{'U4(phi)':>14}")
            print(h3); print("  " + "-" * (len(h3) - 2))
            for (name, cfgs, _), (_, _, mean, _err) in zip(sources, block_rows):
                n_raw = cfgs.shape[0] // 2 if args.paired_bootstrap else None
                m_s, e_s = bootstrap_stats_per_site(
                    cfgs, n_boot=args.n_boot, seed=args.seed, n_raw=n_raw)
                print(f"  {name:>20}  "
                      f"{m_s['abs_phi']:>6.4f}±{e_s['abs_phi']:<6.4f}  "
                      f"{m_s['k2']:>6.4f}±{e_s['k2']:<6.4f}  "
                      f"{m_s['k4']:>+6.4f}±{e_s['k4']:<6.4f}  "
                      f"{m_s['k6']:>+6.4f}±{e_s['k6']:<6.4f}  "
                      f"{m_s['k8']:>+6.4f}±{e_s['k8']:<6.4f}  "
                      f"{m_s['U4_site']:>+6.4f}±{e_s['U4_site']:<6.4f}")
                # store per-site numbers in the same row as per-V for CSV/TeX
                for r in rows:
                    if r["kappa"] == k and r["source"] == name:
                        r.update({f"site_{kk}": m_s[kk] for kk in m_s})
                        r.update({f"d_site_{kk}": e_s[kk] for kk in e_s})
                        break
            print()

    if rows:
        stem = (f"results/L{L}_multiL_HMC_cumulants_raw_z2"
                f"_em_{schedule}_steps{args.num_steps}_ep{args.ep}")
        out_csv = Path(stem + ".csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["kappa", "lo", "hi", "source", "N", "V",
                "mean_M", "d_mean_M", "abs_M", "d_abs_M",
                "chi", "d_chi", "U4", "d_U4",
                "k2", "d_k2", "k3", "d_k3", "k4", "d_k4", "k5", "d_k5",
                "k6", "d_k6", "k7", "d_k7", "k8", "d_k8"]
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
        print(f"Saved CSV: {out_csv}")

        out_tex = Path(stem + ".tex")
        write_latex_table(rows, out_tex, ep=args.ep, schedule=schedule)
        print(f"Saved LaTeX table: {out_tex}")


# ---------------------------------------------------------------------------
# LaTeX table emission
# ---------------------------------------------------------------------------

def _val_err(v: float, e: float, sig_v: int = 3, sig_e: int = 1) -> str:
    """
    Render value(error) compactly in LaTeX. Uses scientific notation for very
    small / large magnitudes; otherwise fixed-point with errors in parentheses
    counted in the last quoted digit (e.g. 0.05157(45)).
    """
    if not np.isfinite(v):
        return r"$\mathrm{nan}$"
    av = abs(v)
    # scientific: |v| < 1e-3 or >= 1e3
    if av != 0 and (av < 1e-3 or av >= 1e3):
        # round v to sig_v significant figures
        from math import floor, log10
        exp = int(floor(log10(av)))
        scale = 10.0 ** exp
        vv = v / scale
        ee = e / scale
        return rf"${vv:.{max(sig_v-1,1)}f}({ee*10**(max(sig_v-1,1)):.0f})\!\times\!10^{{{exp}}}$"
    # fixed-point: choose decimals from error
    if e <= 0 or not np.isfinite(e):
        return rf"${v:.4g}$"
    from math import floor, log10
    e_exp = int(floor(log10(e))) if e > 0 else -4
    decimals = max(0, -(e_exp - (sig_e - 1)))
    val_str = f"{v:.{decimals}f}"
    # err in last decimals digits
    err_int = round(e * 10 ** decimals)
    return rf"${val_str}({err_int})$"


def _src_label(name: str) -> str:
    if name == "HMC":
        return r"HMC"
    if name.startswith("HMC_subN"):
        n = name[len("HMC_subN"):]
        return rf"HMC ($N{{=}}{n}$)"
    if name == "DM_multiL4-8":
        return r"DM\,$\{4,8\}$"
    if name == "DM_multiL4-8-16":
        return r"DM\,$\{4,8,16\}$"
    if name == "DM_multiL4-8-16-32":
        return r"DM\,$\{4,8,16,32\}$"
    if name.startswith("DM_L64single"):
        return r"DM\,$L{=}64$"
    return name.replace("_", r"\_")


def write_latex_table(rows: list[dict], out: Path, ep: int, schedule: str):
    """
    Single table* with two horizontal blocks:
      Block A: <|M|>, chi, U_4
      Block B: kappa_2, kappa_4, kappa_6, kappa_8 (odd ones vanish by Z_2)
    Rows grouped by kappa, separated by midrules.
    """
    kappas = sorted({r["kappa"] for r in rows})
    by_k = {k: [r for r in rows if r["kappa"] == k] for k in kappas}

    lines = []
    lines.append(r"% auto-generated by 3Dphi4/analysis/compare_L64_multiL_HMC_cumulants.py")
    lines.append(r"% raw (physical) field space, Z_2-symmetrised by cat([phi, -phi])")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Local observables on the per-configuration magnetisation "
        r"$M=\bar\phi$ at $L=64$, $\lambda=0.9$, comparing HMC with "
        r"cross-$L$ DM samples from multi-$L$ training sets "
        r"$\{4,8\}$, $\{4,8,16\}$, $\{4,8,16,32\}$ and, where available, the "
        r"in-distribution single-$L$ DM trained directly at $L=64$ "
        rf"(EM/{schedule}, $2000$ steps, multi-$L$ epoch $={ep}$). "
        r"All quantities are evaluated in the physical (raw) field space, "
        r"and each ensemble is $Z_2$-symmetrised by concatenating "
        r"$\{\phi,-\phi\}$ so that $\langle M\rangle=0$ exactly and the odd "
        r"cumulants $\kappa_3,\kappa_5,\kappa_7$ vanish identically. "
        r"For each $\kappa$ the HMC reference is shown both with the full "
        r"$N=2560$ ensemble and with a random $N=512$ subsample matched to "
        r"the DM sample count, isolating the genuine bias of the diffusion "
        r"model from finite-statistics effects. "
        r"Cumulants are computed by the recursion of \cite{Hammoud:2024}; "
        r"errors are bootstrap $1\sigma$ over configurations.}"
    )
    lines.append(r"\label{tab:multiL_HMC_cumulants}")
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$\kappa$ & source "
        r"& $\langle|\bar\phi|\rangle$ & $\chi$ & $U_4$ "
        r"& $\kappa_2$ & $\kappa_4$ & $\kappa_6$ & $\kappa_8$ \\"
    )
    lines.append(r"\midrule")
    for ki, k in enumerate(kappas):
        block = by_k[k]
        for ri, r in enumerate(block):
            kappa_cell = f"${k:g}$" if ri == 0 else ""
            cells = [
                kappa_cell,
                _src_label(r["source"]),
                _val_err(r["abs_M"], r["d_abs_M"]),
                _val_err(r["chi"],   r["d_chi"]),
                _val_err(r["U4"],    r["d_U4"]),
                _val_err(r["k2"],    r["d_k2"]),
                _val_err(r["k4"],    r["d_k4"]),
                _val_err(r["k6"],    r["d_k6"]),
                _val_err(r["k8"],    r["d_k8"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if ki < len(kappas) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    # Optional second table: single-site cumulants (Aarts et al. convention).
    has_site = any("site_k2" in r for r in rows)
    if has_site:
        lines.append(r"")
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Single-site cumulants $\kappa_n^{(\text{site})}$ of the "
            r"underlying field $\phi(x)$ for the same ensembles as "
            r"\Tab{tab:multiL_HMC_cumulants}, computed by pooling every "
            r"$(\text{cfg},x)$ pair into one sample of $\phi$, following "
            r"\cite{Hammoud:2024} eq. (5.11). By translation invariance "
            r"$\mu_n(x)=\langle\phi^n(x)\rangle$ is $x$-independent, so the "
            r"pooled estimator achieves an effective sample size $N_{\rm cfg}V$ "
            r"and the bootstrap is still resampled at the configuration level "
            r"(the independent statistical unit). Compared with the "
            r"$\bar\phi$-based cumulants of \Tab{tab:multiL_HMC_cumulants}, "
            r"these are not suppressed by powers of $V$: at fixed correlation "
            r"length $\xi$, $\kappa_n(\bar\phi)\sim\xi^{D(n-1)}/V^{n-1}$ "
            r"whereas $\kappa_n^{(\text{site})}=O(1)$ in lattice units. "
            r"$U_4^{(\text{site})}=1-\langle\phi^4\rangle/(3\langle\phi^2\rangle^2)$ "
            r"is the on-site Binder analogue.}"
        )
        lines.append(r"\label{tab:multiL_HMC_cumulants_persite}")
        lines.append(r"\begin{tabular}{llcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"$\kappa$ & source & $\langle|\phi|\rangle$ "
            r"& $\kappa_2^{(\text{site})}$ & $\kappa_4^{(\text{site})}$ "
            r"& $\kappa_6^{(\text{site})}$ & $\kappa_8^{(\text{site})}$ "
            r"& $U_4^{(\text{site})}$ \\"
        )
        lines.append(r"\midrule")
        for ki, k in enumerate(kappas):
            block = by_k[k]
            for ri, r in enumerate(block):
                kappa_cell = f"${k:g}$" if ri == 0 else ""
                cells = [
                    kappa_cell,
                    _src_label(r["source"]),
                    _val_err(r.get("site_abs_phi", float('nan')),
                             r.get("d_site_abs_phi", float('nan'))),
                    _val_err(r.get("site_k2", float('nan')),
                             r.get("d_site_k2", float('nan'))),
                    _val_err(r.get("site_k4", float('nan')),
                             r.get("d_site_k4", float('nan'))),
                    _val_err(r.get("site_k6", float('nan')),
                             r.get("d_site_k6", float('nan'))),
                    _val_err(r.get("site_k8", float('nan')),
                             r.get("d_site_k8", float('nan'))),
                    _val_err(r.get("site_U4_site", float('nan')),
                             r.get("d_site_U4_site", float('nan'))),
                ]
                lines.append(" & ".join(cells) + r" \\")
            if ki < len(kappas) - 1:
                lines.append(r"\midrule")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
