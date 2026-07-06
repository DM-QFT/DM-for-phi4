"""
HMC-equivalent effective sample size N_eff vs training epoch.

For each Z2-even observable O and each saved DM checkpoint we report two
quantities (Eqs. (1) and (2) in ess_mse section of DM.tex),

    N_eff      = sigma_p^2 / ( delta_hat^2 + V_DM )
    N_eff_ref  = sigma_p^2 / ( delta_hat^2 + V_DM + V_HMC )

with

    delta_hat^2 = max[ (Obar_DM - Obar_HMC)^2 - V_DM - V_HMC, 0 ]   (shrinkage)
    V_DM        = Var_DM(O) / K                                      (K i.i.d.)
    V_HMC       = 2 tau_int(O) * Var_HMC(O) / N_HMC                  (Sokal window)
    sigma_p^2   = Var_HMC(O)                                         (target variance)

The shrinkage subtracts the noise floor V_DM + V_HMC from the empirical
squared difference so that delta_hat^2 is an (approximately) unbiased
non-negative estimator of the true squared systematic DM bias.  V_HMC uses
the Sokal-windowed integrated autocorrelation time of the HMC chain so
that residual HMC autocorrelation (especially near criticality) is not
mistaken for a small reference variance.

DM samples are expected as
    {samples_dir}/samples_em_steps{N}_epoch={E}.npy
with shape (L, L, K_total) in PHYSICAL units.  Two layouts are accepted
and auto-detected:

  (a) Raw  (current sampling pipeline):
        K_total = K raw DM configurations, used as-is.
  (b) Z2-paired  (legacy sampling, sample_phi4_sweep.py --z2):
        K_total = 2K and the second half is the elementwise negation of
        the first half; the analysis drops the redundant flipped copies
        and uses only the K raw configurations.

In either case the K independent samples define V_DM = sigma_DM^2 / K.
Observables are the Z2-even set {|m|, m^2, m^4, S/V}.

Bootstrap CI: DM rows are resampled i.i.d. (K with replacement) and HMC
rows by a moving block bootstrap with block size ceil(2*tau_int), with
tau_int held fixed at its full-chain value.  The reported centre is the
full-data point estimate; error bars are the [16%, 84%] bootstrap
percentiles.  Bootstrap mean is *not* used as the centre because
N_eff = sigma_p^2 / MSE has a long upper tail.

Usage:
    python analysis/ess_mse_vs_epoch.py \
        --k 0.2705 --sigma 450 \
        --hmc_data trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2 \
        --out_npz analysis/results/ess_mse_vs_epoch_L128_k0.2705_sigma450_ema.npz
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- I/O --------------------

def load_hmc_reference(data_path):
    """HMC cfgs in physical units, shape (N, L, L), upcast to float64."""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    return cfgs


def find_sample_files(samples_dir, sde_steps):
    """Return sorted [(epoch_int, path), ...] for samples_em_steps{N}_epoch=*.npy."""
    pattern = re.compile(rf"^samples_em_steps{sde_steps}_epoch=(\d+)\.npy$")
    out = []
    for p in sorted(Path(samples_dir).iterdir()):
        m = pattern.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def log_select(items, n):
    if len(items) <= n:
        return items
    idx = np.unique(np.round(np.geomspace(1, len(items), n)).astype(int) - 1)
    return [items[i] for i in idx]


# -------------------- observables --------------------

def action_density_2d(phi_phys, k, l):
    """phi^4 action density S/V per sample in physical units. phi_phys: (N, L, L)."""
    nb = np.roll(phi_phys, 1, axis=1) + np.roll(phi_phys, 1, axis=2)
    s_total = np.sum(-2.0 * k * phi_phys * nb
                     + (1.0 - 2.0 * l) * phi_phys ** 2
                     + l * phi_phys ** 4, axis=(1, 2))
    V = phi_phys.shape[1] * phi_phys.shape[2]
    return s_total / V


def diagonal_propagator_modes(L, D):
    """Three diagonal-mode bands (n, n, ...) for k_min, k_mid, k_large.

    k_min:   n = 1                            (single mode)
    k_mid:   n in [L/8, L/4)                  (band-averaged)
    k_large: n in [3L/8, L/2]                 (band-averaged, includes the
             corner if L is even)
    """
    n_min_set   = [1]
    n_mid_set   = list(range(L // 8, L // 4))
    n_large_set = list(range(3 * L // 8, L // 2 + 1))
    return {
        "Gk_min":   [tuple([n] * D) for n in n_min_set],
        "Gk_mid":   [tuple([n] * D) for n in n_mid_set],
        "Gk_large": [tuple([n] * D) for n in n_large_set],
    }


def propagator_per_sample(phi, modes):
    """Per-sample per-mode propagator |phi_hat(k)|^2 / V at chosen modes.

    phi: float64 (N, L, ..., L) with D = phi.ndim - 1 spatial dims.
    modes: dict {name: [k_index_tuple of length D, ...]}.

    Returns {name: (N, M) float64} where M is the number of modes in the
    band.  Per-mode values are NOT averaged here; downstream the band
    N_eff is built by averaging the per-mode shrinkage-corrected N_eff,
    which gives every k in the band equal weight regardless of the very
    different magnitudes of |phi_hat(k)|^2.
    """
    spatial = phi.shape[1:]
    D = len(spatial)
    V = float(np.prod(spatial))
    axes = tuple(range(1, D + 1))
    fk = np.fft.fftn(phi, axes=axes)
    pk = (fk.real ** 2 + fk.imag ** 2) / V
    out = {}
    for name, idxs in modes.items():
        vals = np.stack([pk[(slice(None),) + idx] for idx in idxs], axis=-1)
        out[name] = vals
    return out


def compute_observables(cfgs_phys, k, l):
    """Per-sample Z2-even observables. cfgs_phys: (N, L, L).

    Returns scalar moments {|m|, m^2, m^4, S/V} and three band-averaged
    diagonal propagator values {G(k_min), G(k_mid), G(k_large)}.
    """
    Mbar = cfgs_phys.mean(axis=(1, 2))
    obs = {
        "absMbar": np.abs(Mbar),
        "Mbar2":   Mbar ** 2,
        "Mbar4":   Mbar ** 4,
        "SoverV":  action_density_2d(cfgs_phys, k, l),
    }
    L = cfgs_phys.shape[1]
    obs.update(propagator_per_sample(cfgs_phys,
                                     diagonal_propagator_modes(L, D=2)))
    return obs


# -------------------- HMC autocorrelation --------------------

def autocorr_fft(x, max_lag=None):
    """Biased normalised autocorrelation rho(t) = c(t)/c(0), t = 0..max_lag-1."""
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    n = x.size
    if max_lag is None:
        max_lag = n // 4
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    F = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(F * np.conjugate(F), n=nfft)[:max_lag].real / n
    return acov / acov[0]


def tau_int_sokal(x, c=5.0, max_lag=None):
    """Integrated autocorrelation time via Sokal's automatic window."""
    rho = autocorr_fft(x, max_lag=max_lag)
    cum = 0.5 + np.cumsum(rho[1:])
    W   = np.arange(1, cum.size + 1)
    closed = W >= c * np.abs(cum)
    if np.any(closed):
        idx = int(np.argmax(closed))
        return float(cum[idx]), int(W[idx]), True
    return float(cum[-1]), int(W[-1]), False


def block_bootstrap_indices(N, block_size, rng):
    """Moving block bootstrap (Kuensch 1989) index array of length N."""
    if block_size <= 1 or block_size >= N:
        return rng.integers(0, N, N)
    n_blocks = (N + block_size - 1) // block_size
    starts = rng.integers(0, N - block_size + 1, n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()
    return idx[:N]


# -------------------- N_eff^MSE bootstrap --------------------

def mse_neff_bootstrap(obs_dm, obs_hmc, tau_hmc, n_boot=400, seed=0):
    """Shrinkage + block-bootstrap N_eff and N_eff_ref.

    obs_dm:  (N_DM,)  or (N_DM, M)   per-sample values, one column per mode.
    obs_hmc: (N_HMC,) or (N_HMC, M)  per-sample values, one column per mode.
    tau_hmc: scalar or (M,)          integrated autocorrelation time(s).

    For M > 1 the function computes the shrinkage-corrected N_eff and
    N_eff_ref at each mode independently and reports the uniform mean
    across modes.  Bootstrap indices are shared across modes within
    each replica (DM rows i.i.d., HMC rows via moving block bootstrap
    with block size ceil(2 * max(tau_hmc))).
    """
    obs_dm  = np.asarray(obs_dm,  dtype=np.float64)
    obs_hmc = np.asarray(obs_hmc, dtype=np.float64)
    if obs_dm.ndim == 1:
        obs_dm = obs_dm[:, None]
    if obs_hmc.ndim == 1:
        obs_hmc = obs_hmc[:, None]
    tau_arr = np.atleast_1d(np.asarray(tau_hmc, dtype=np.float64))

    N_DM, M = obs_dm.shape
    N_HMC = obs_hmc.shape[0]
    if obs_hmc.shape[1] != M or tau_arr.size != M:
        raise ValueError(f"mode count mismatch: obs_dm={obs_dm.shape}, "
                         f"obs_hmc={obs_hmc.shape}, tau={tau_arr.shape}")

    rng = np.random.default_rng(seed)
    # One block size for the whole band: the largest 2*tau among modes,
    # which is the most conservative choice (resolves the slowest mode).
    block = max(1, int(np.ceil(2.0 * float(tau_arr.max()))))

    # Per-mode point quantities (vectorised over modes).
    sigma_p2 = obs_hmc.var(axis=0, ddof=1)                     # (M,)
    mean_dm  = obs_dm.mean(axis=0)                              # (M,)
    mean_hmc = obs_hmc.mean(axis=0)                             # (M,)
    var_dm   = obs_dm.var(axis=0, ddof=1) / N_DM                # (M,)
    var_hmc  = 2.0 * tau_arr * sigma_p2 / N_HMC                 # (M,)
    diff_sq    = (mean_dm - mean_hmc) ** 2                      # (M,)
    delta2_hat = np.maximum(diff_sq - var_dm - var_hmc, 0.0)    # (M,)

    dp_modes = delta2_hat + var_dm
    dr_modes = delta2_hat + var_dm + var_hmc
    neff_pure_modes = np.where(dp_modes > 0, sigma_p2 / dp_modes, np.inf)
    neff_ref_modes  = np.where(dr_modes > 0, sigma_p2 / dr_modes, np.inf)

    def safe_mean(a):
        ok = a[np.isfinite(a)]
        return float(ok.mean()) if ok.size else np.inf

    neff_pure = safe_mean(neff_pure_modes)
    neff_ref  = safe_mean(neff_ref_modes)

    boot_pure = np.empty(n_boot)
    boot_ref  = np.empty(n_boot)
    for b in range(n_boot):
        i_dm  = rng.integers(0, N_DM, N_DM)
        i_hmc = block_bootstrap_indices(N_HMC, block, rng)
        x_dm  = obs_dm[i_dm]
        x_hmc = obs_hmc[i_hmc]
        s2  = x_hmc.var(axis=0, ddof=1)
        bd  = x_dm.mean(axis=0) - x_hmc.mean(axis=0)
        vd  = x_dm.var(axis=0, ddof=1) / N_DM
        vh  = 2.0 * tau_arr * s2 / N_HMC
        d2  = np.maximum(bd ** 2 - vd - vh, 0.0)
        dp  = d2 + vd
        dr  = d2 + vd + vh
        np_modes = np.where(dp > 0, s2 / dp, np.inf)
        nr_modes = np.where(dr > 0, s2 / dr, np.inf)
        boot_pure[b] = safe_mean(np_modes)
        boot_ref[b]  = safe_mean(nr_modes)

    def stats(arr):
        ok = arr[np.isfinite(arr)]
        if ok.size == 0:
            return dict(mean=np.nan, median=np.nan, p16=np.nan, p84=np.nan, sem=0.0)
        return dict(
            mean   = float(ok.mean()),
            median = float(np.percentile(ok, 50)),
            p16    = float(np.percentile(ok, 16)),
            p84    = float(np.percentile(ok, 84)),
            sem    = float(ok.std(ddof=1)) if ok.size > 1 else 0.0,
        )
    sp = stats(boot_pure)
    sr = stats(boot_ref)

    # Band-averaged scalar summaries (uniform mean across modes).
    bias_band    = float(mean_dm.mean() - mean_hmc.mean())
    diff_sq_band = float(diff_sq.mean())
    d2_hat_band  = float(delta2_hat.mean())
    var_dm_band  = float(var_dm.mean())
    var_hmc_band = float(var_hmc.mean())
    sigma_p2_band = float(sigma_p2.mean())
    tau_band     = float(tau_arr.mean())

    # Efficiency ratio: N_eff / N_DM in [0, ~1].  DM-size-independent in the
    # variance-dominated regime where it reduces to sigma_p^2 / sigma_DM^2.
    invK = 1.0 / N_DM
    return {
        # point estimates (use as plotting centre)
        "neff":     neff_pure,
        "neff_ref": neff_ref,
        "eff":      neff_pure * invK,
        "eff_ref":  neff_ref  * invK,
        # bootstrap distribution (N_eff)
        "neff_mean":   sp["mean"],   "neff_median":   sp["median"],
        "neff_p16":    sp["p16"],    "neff_p84":      sp["p84"],
        "neff_sem":    sp["sem"],
        "neff_ref_mean":   sr["mean"],   "neff_ref_median":   sr["median"],
        "neff_ref_p16":    sr["p16"],    "neff_ref_p84":      sr["p84"],
        "neff_ref_sem":    sr["sem"],
        # bootstrap distribution (efficiency ratio = N_eff / N_DM)
        "eff_mean":    sp["mean"]   * invK, "eff_median":   sp["median"] * invK,
        "eff_p16":     sp["p16"]    * invK, "eff_p84":      sp["p84"]    * invK,
        "eff_sem":     sp["sem"]    * invK,
        "eff_ref_mean":   sr["mean"]   * invK, "eff_ref_median":   sr["median"] * invK,
        "eff_ref_p16":    sr["p16"]    * invK, "eff_ref_p84":      sr["p84"]    * invK,
        "eff_ref_sem":    sr["sem"]    * invK,
        # bookkeeping
        "N_DM":       N_DM,
        "n_modes":    M,
        # MSE decomposition (band-averaged over modes for M > 1)
        "bias_raw":   bias_band,
        "diff_sq":    diff_sq_band,
        "delta2_hat": d2_hat_band,
        "var_dm":     var_dm_band,
        "var_hmc":    var_hmc_band,
        "sigma_p2":   sigma_p2_band,
        "tau_hmc":    tau_band,
        "block_size": block,
        "regime":     "bias-dom" if diff_sq_band > (var_dm_band + var_hmc_band)
                       else "var-dom",
    }


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--samples_dir", default=None,
                   help="Dir containing samples_em_steps{N}_epoch=*.npy. "
                        "Default: runs/phi4_L{L}_k{k}_l{l}_ncsnpp_sigma{SIGMA}/data")
    p.add_argument("--sigma", type=int, default=None,
                   help="Used to build samples_dir when --samples_dir not given.")
    p.add_argument("--network", default="ncsnpp")
    p.add_argument("--sde_steps", type=int, default=2000)
    p.add_argument("--n_epochs", type=int, default=20,
                   help="Cap on number of ckpts; <=0 uses all available.")
    p.add_argument("--epochs", default=None,
                   help="Override: comma-separated epoch list.")
    p.add_argument("--hmc_data", required=True)
    p.add_argument("--n_boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260423)
    p.add_argument("--out_npz", required=True)
    args = p.parse_args()

    # Resolve samples_dir
    if args.samples_dir is None:
        if args.sigma is None:
            sys.exit("ERROR: provide either --samples_dir or --sigma.")
        args.samples_dir = (f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_"
                            f"{args.network}_sigma{args.sigma}/data")
    samples_dir = Path(args.samples_dir)
    if not samples_dir.is_dir():
        sys.exit(f"ERROR: samples dir not found: {samples_dir}")

    all_files = find_sample_files(samples_dir, args.sde_steps)
    if not all_files:
        sys.exit(f"ERROR: no samples_em_steps{args.sde_steps}_epoch=*.npy "
                 f"in {samples_dir}")
    print(f"[ess] found {len(all_files)} sample files in {samples_dir}")

    if args.epochs:
        wanted = [int(e) for e in args.epochs.split(",") if e.strip()]
        files = [next(((ep, p) for ep, p in all_files if ep == w), None) for w in wanted]
        files = [f for f in files if f is not None]
    elif args.n_epochs > 0:
        files = log_select(all_files, args.n_epochs)
    else:
        files = all_files
    epochs = [ep for ep, _ in files]
    print(f"[ess] sweeping {len(files)} epochs: {epochs}")

    # HMC reference
    print(f"[ess] loading HMC ref: {args.hmc_data}")
    hmc_phys = load_hmc_reference(args.hmc_data)
    print(f"[ess] HMC samples: {hmc_phys.shape}")
    obs_hmc = compute_observables(hmc_phys, args.k, args.l)

    obs_names = ["absMbar", "Mbar2", "Mbar4", "SoverV",
                 "Gk_min", "Gk_mid", "Gk_large"]
    print(f"[ess] HMC observables and Sokal-window tau_int (chain order assumed):")
    tau_hmc = {}
    for name in obs_names:
        a = obs_hmc[name]
        if a.ndim == 1:
            tau, W, closed = tau_int_sokal(a)
            tau_hmc[name] = tau
            sem = a.std(ddof=1) / np.sqrt(len(a))
            eff_indep = len(a) / (2.0 * tau) if tau > 0 else len(a)
            flag = "" if closed else "  [WARNING: window did not close]"
            print(f"        {name:<8s}  <O>={a.mean():.6g} +/- {sem:.2g}   "
                  f"sigma_p={a.std(ddof=1):.4g}   tau_int={tau:.2f}  W={W}  "
                  f"N_HMC/2tau={eff_indep:.0f}{flag}")
        else:
            # Multi-mode propagator band: per-mode tau.
            M = a.shape[1]
            taus = np.empty(M)
            n_open = 0
            for j in range(M):
                t_j, _, c_j = tau_int_sokal(a[:, j])
                taus[j] = t_j
                n_open += int(c_j)
            tau_hmc[name] = taus
            flag = "" if n_open == M else f"  [WARNING: {M-n_open}/{M} windows did not close]"
            print(f"        {name:<8s}  M={M}  tau_int "
                  f"mean={taus.mean():.2f}  min={taus.min():.2f}  "
                  f"max={taus.max():.2f}{flag}")

    epoch_data = {n: [] for n in obs_names}

    for ie, (ep, fpath) in enumerate(files):
        arr = np.load(fpath)                       # (L, L, N_total)
        if arr.ndim != 3:
            sys.exit(f"ERROR: unexpected shape {arr.shape} in {fpath}")
        x_phys = arr.transpose(2, 0, 1).astype(np.float64, copy=False)
        # Auto-detect the layout: legacy --z2 output concatenated [+x, -x]
        # so the second half is exactly the elementwise negation of the
        # first half.  Drop the redundant flipped copies if present so
        # that N_dm = K is the independent count for Var(Obar_DM).
        n_total = x_phys.shape[0]
        if n_total % 2 == 0 and np.array_equal(x_phys[:n_total // 2],
                                               -x_phys[n_total // 2:]):
            x_phys = x_phys[: n_total // 2]
            layout = "Z2-paired (trimmed)"
        else:
            layout = "raw"
        obs_dm = compute_observables(x_phys, args.k, args.l)

        print(f"[ess] ep={ep:>5d}  ({ie+1}/{len(files)})  "
              f"N_dm={len(x_phys)}  layout={layout}")
        for name in obs_names:
            res = mse_neff_bootstrap(obs_dm[name], obs_hmc[name],
                                     tau_hmc=tau_hmc[name],
                                     n_boot=args.n_boot,
                                     seed=args.seed + ie)
            epoch_data[name].append(res)
            r = res
            print(f"        {name:<8s}  d2_hat={r['delta2_hat']:.3g}  "
                  f"V_DM={r['var_dm']:.3g}  V_HMC={r['var_hmc']:.3g}  "
                  f"eff={r['eff']:.3g} [{r['eff_p16']:.3g}, {r['eff_p84']:.3g}]  "
                  f"eff_ref={r['eff_ref']:.3g}  ({r['regime']})")

    # ---- Save npz ----
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    eps_arr = np.array(epochs)
    save_dict = {"epochs": eps_arr, "k": args.k, "l": args.l, "L": args.L,
                 "sde_steps": args.sde_steps, "n_hmc": len(hmc_phys),
                 "samples_dir": str(samples_dir)}
    save_keys = (
        # point estimates
        "neff", "neff_ref", "eff", "eff_ref",
        # bootstrap summaries (N_eff)
        "neff_mean", "neff_median", "neff_sem", "neff_p16", "neff_p84",
        "neff_ref_mean", "neff_ref_median", "neff_ref_sem",
        "neff_ref_p16", "neff_ref_p84",
        # bootstrap summaries (efficiency ratio = N_eff / N_DM)
        "eff_mean", "eff_median", "eff_sem", "eff_p16", "eff_p84",
        "eff_ref_mean", "eff_ref_median", "eff_ref_sem",
        "eff_ref_p16", "eff_ref_p84",
        # bookkeeping
        "N_DM", "n_modes",
        # MSE decomposition (band-averaged when n_modes > 1)
        "bias_raw", "diff_sq", "delta2_hat",
        "var_dm", "var_hmc", "sigma_p2",
        "tau_hmc", "block_size",
    )
    for name in obs_names:
        records = epoch_data[name]
        for key in save_keys:
            save_dict[f"{name}__{key}"] = np.array([r[key] for r in records])
    np.savez(out, **save_dict)
    print(f"[ess] saved npz -> {out}")

    # ---- Quick-look plot: N_eff (pure DM), centre = point estimate ----
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'lines.linewidth': 1.5, 'lines.markersize': 5,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    scalar_names = ["absMbar", "Mbar2", "Mbar4", "SoverV"]
    prop_names   = ["Gk_min", "Gk_mid", "Gk_large"]
    colors = {"absMbar": "C1", "Mbar2": "C2", "Mbar4": "C3", "SoverV": "C4",
              "Gk_min": "C5", "Gk_mid": "C6", "Gk_large": "C0"}
    pretty = {"absMbar": r"$|\bar\phi|$", "Mbar2": r"$\bar\phi^2$",
              "Mbar4": r"$\bar\phi^4$", "SoverV": r"$S/V$",
              "Gk_min": r"$G(k_{\rm min})$",
              "Gk_mid": r"$\langle G(k_{\rm mid})\rangle$",
              "Gk_large": r"$\langle G(k_{\rm large})\rangle$"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5),
                             sharex=True, sharey='row')
    row_titles = ["scalar moments", "diagonal propagator"]
    for row, names in enumerate((scalar_names, prop_names)):
        for col, (suffix, ylab) in enumerate((
            ("eff",     r'$N_{\rm eff}/N_{\rm DM}$  (pure DM, shrinkage)'),
            ("eff_ref", r'$N_{\rm eff,ref}/N_{\rm DM}$  (with HMC ref.\ variance)'),
        )):
            ax = axes[row, col]
            for name in names:
                centre = save_dict[f"{name}__{suffix}"]
                p16    = save_dict[f"{name}__{suffix}_p16"]
                p84    = save_dict[f"{name}__{suffix}_p84"]
                lo = np.clip(centre - p16, 0.0, None)
                hi = np.clip(p84 - centre, 0.0, None)
                ax.errorbar(eps_arr, centre, yerr=[lo, hi],
                            marker='o', capsize=2, color=colors[name],
                            label=pretty[name])
            ax.axhline(1.0, color='k', lw=0.6, ls=':')
            ax.set_xscale('log'); ax.set_yscale('log')
            if row == 1:
                ax.set_xlabel('Training epoch')
            ax.set_ylabel(ylab if col == 0 else
                          ylab.replace('pure DM, shrinkage', 'pure')
                              .replace('with HMC ref.\\ variance', 'with HMC ref.'))
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=9, title=row_titles[row])
    K_dm = int(save_dict[f"{obs_names[0]}__N_DM"][-1])
    fig.suptitle(rf'2D $\phi^4$  $L={args.L},\;\kappa={args.k}$  '
                 rf'($N_{{\rm DM}}={K_dm}$, $N_{{\rm HMC}}={len(hmc_phys)}$)')
    fig.tight_layout()
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf)
    fig.savefig(out.with_suffix(".png"))
    print(f"[ess] saved pdf -> {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
