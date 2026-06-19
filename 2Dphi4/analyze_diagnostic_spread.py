"""
Analyze late-training snapshot spread.

Loads the 6 sample arrays produced by run_diagnostic_finetune.sh, computes
|m| and chi_conn at each snapshot, and reports:
  - O(epoch) values with within-snapshot block-jackknife errors
  - σ_between (std across snapshots)
  - δ_within  (mean within-snapshot statistical error)
  - ratio r = σ_between / δ_within
  - linear drift over the 100-epoch window
  - drift-dominated / noise-dominated / spread classification

Usage:
    python analyze_diagnostic_spread.py \
        --run_dir runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640 \
        --tags ep10000 ep10020 ep10040 ep10060 ep10080 ep10100
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def _samples_to_flat(samples):
    """Return (N, L*L). sample_phi4.py saves shape (L, L, N)."""
    x = np.asarray(samples)
    if x.ndim == 3 and x.shape[-1] != x.shape[0]:
        # (L, L, N) → (N, L, L)
        x = np.transpose(x, (2, 0, 1))
    return x.reshape(x.shape[0], -1)


def magnetization_abs(x_flat):
    m = x_flat.mean(axis=1)
    return float(np.mean(np.abs(m)))


def chi_conn(x_flat, L):
    m = x_flat.mean(axis=1)
    return float(L * L * (np.mean(m * m) - np.mean(np.abs(m)) ** 2))


def block_jackknife(x_flat, fn, n_blocks=32):
    """Block-jackknife mean and error of fn(x_flat)."""
    N = x_flat.shape[0]
    bs = N // n_blocks
    if bs == 0:
        return fn(x_flat), float("nan")
    x_flat = x_flat[: bs * n_blocks]
    full = fn(x_flat)
    pseudo = np.empty(n_blocks)
    for i in range(n_blocks):
        mask = np.ones(n_blocks, dtype=bool)
        mask[i] = False
        idx = np.concatenate([np.arange(j * bs, (j + 1) * bs) for j in np.where(mask)[0]])
        pseudo[i] = fn(x_flat[idx])
    mean = pseudo.mean()
    err = np.sqrt((n_blocks - 1) * np.mean((pseudo - mean) ** 2))
    return full, err


def classify(name, vals, errs, epochs):
    sigma_between = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    delta_within = float(np.mean(errs))
    r = sigma_between / delta_within if delta_within > 0 else float("inf")

    # Linear drift
    slope, intercept = np.polyfit(epochs, vals, 1)
    drift_total = slope * (epochs.max() - epochs.min())
    drift_ratio = abs(drift_total) / (sigma_between + 1e-30)

    print(f"\n=== {name} ===")
    for e, v, dv in zip(epochs, vals, errs):
        print(f"  ep={e:>6d}:  {v:.6e}  +/- {dv:.2e}")
    print(f"  sigma_between (std over snapshots)    = {sigma_between:.4e}")
    print(f"  delta_within  (mean within-ckpt err)  = {delta_within:.4e}")
    print(f"  ratio r = sigma_between / delta_within = {r:.3f}")
    print(f"  linear drift over {int(epochs.max() - epochs.min())} epochs = {drift_total:+.4e}")
    print(f"  |drift| / sigma_between               = {drift_ratio:.3f}")

    if drift_ratio > 1.5:
        verdict = "DRIFT-DOMINATED — training has NOT converged at this scale"
    elif r < 1.0:
        verdict = "NOISE-DOMINATED — snapshot spread below within-ckpt statistical error; cannot resolve"
    elif r < 3.0:
        verdict = "MILD SPREAD — snapshot fluctuations comparable to statistical error"
    else:
        verdict = "LARGE SPREAD — snapshot fluctuations exceed statistical error; init-variance proxy likely meaningful"
    print(f"  -> {verdict}")
    return dict(sigma_between=sigma_between, delta_within=delta_within,
                ratio=r, drift_total=drift_total, drift_ratio=drift_ratio,
                verdict=verdict)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tags", nargs="+", required=True,
                   help="Tags like ep10000 ep10020 ... matching the --ep arg used in sampling")
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--n_blocks", type=int, default=32)
    args = p.parse_args()

    data_dir = os.path.join(args.run_dir, "data")

    epochs = []
    for tag in args.tags:
        m = re.match(r"ep(\d+)", tag)
        if not m:
            raise ValueError(f"Tag '{tag}' does not match epNNNNN pattern")
        epochs.append(int(m.group(1)))
    epochs = np.array(epochs)

    m_vals, m_errs, chi_vals, chi_errs = [], [], [], []
    for tag in args.tags:
        fname = os.path.join(data_dir, f"samples_em_steps{args.num_steps}_{tag}.npy")
        x = np.load(fname)
        x_flat = _samples_to_flat(x)
        print(f"{tag}: loaded {fname}  flat shape={x_flat.shape}")

        mv, me = block_jackknife(x_flat, magnetization_abs, n_blocks=args.n_blocks)
        cv, ce = block_jackknife(x_flat, lambda y: chi_conn(y, args.L), n_blocks=args.n_blocks)
        m_vals.append(mv); m_errs.append(me)
        chi_vals.append(cv); chi_errs.append(ce)

    m_vals = np.array(m_vals); m_errs = np.array(m_errs)
    chi_vals = np.array(chi_vals); chi_errs = np.array(chi_errs)

    diag_m = classify("|m|", m_vals, m_errs, epochs)
    diag_c = classify("chi_conn", chi_vals, chi_errs, epochs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(epochs, m_vals, yerr=m_errs, marker="o", capsize=3, lw=1)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(r"$\langle |m| \rangle$")
    axes[0].set_title(f"|m| vs epoch (r={diag_m['ratio']:.2f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(epochs, chi_vals, yerr=chi_errs, marker="o", capsize=3, lw=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(r"$\chi_{\rm conn}$")
    axes[1].set_title(f"chi_conn vs epoch (r={diag_c['ratio']:.2f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.run_dir, "diagnostic_spread.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved plot: {out_png}")

    summary = os.path.join(args.run_dir, "diagnostic_spread.txt")
    with open(summary, "w") as f:
        for name, diag, vals, errs in [("|m|", diag_m, m_vals, m_errs),
                                       ("chi_conn", diag_c, chi_vals, chi_errs)]:
            f.write(f"=== {name} ===\n")
            for e, v, dv in zip(epochs, vals, errs):
                f.write(f"  ep={e}:  {v:.6e} +/- {dv:.2e}\n")
            for k, v in diag.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"Saved summary: {summary}")


if __name__ == "__main__":
    main()
