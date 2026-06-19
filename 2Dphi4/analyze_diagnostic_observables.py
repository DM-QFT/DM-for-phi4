"""
Z2-symmetrized observables on the diagnostic snapshot ensemble.

Observables (magnetization m = (1/V) Σ_x φ_x):
    <|m|>      (= <|φ̄|>)
    χ          = V (<m²> - <|m|>²)
    U_4        = 1 - <m⁴> / (3 <m²>²)

Site-based observables (single-site marginal of φ, flattened over sites+configs):
    <|φ|>
    κ_2^(s)    = <φ²>
    κ_4^(s)    = <φ⁴> - 3 <φ²>²
    κ_6^(s)    = <φ⁶> - 15 <φ⁴><φ²> + 30 <φ²>³
    κ_8^(s)    = <φ⁸> - 28 <φ⁶><φ²> - 35 <φ⁴>² + 420 <φ⁴><φ²>² - 630 <φ²>⁴
    U_4^(s)    = 1 - <φ⁴>_site / (3 <φ²>²_site)

Z2 symmetrization: append -φ to φ along the config axis.
Errors: block jackknife with n_blocks=32 over (post-Z2) configs.
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def load_configs(path):
    """sample_phi4 saves shape (L, L, N). Return (N, V) flat configs and L."""
    x = np.load(path)
    if x.ndim == 3 and x.shape[0] == x.shape[1]:
        x = np.transpose(x, (2, 0, 1))
    N, L, _ = x.shape
    return x.reshape(N, L * L).astype(np.float64), L


def z2_augment(x_flat):
    return np.concatenate([x_flat, -x_flat], axis=0)


def block_sums(x_block):
    """Compute raw sums on a block of configurations.

    Returns sums (not means) so leave-one-out jackknife can reuse them.
    Uses incremental multiplications (p4 = p2*p2 etc.) to avoid the slow
    np.power log/exp path for high integer exponents.
    """
    Nc = x_block.shape[0]
    V = x_block.shape[1]
    m = x_block.mean(axis=1)
    abs_m = np.abs(m)
    m2 = m * m
    m4 = m2 * m2
    phi = x_block.ravel()
    p2 = phi * phi
    p4 = p2 * p2
    p6 = p4 * p2
    p8 = p4 * p4
    return {
        "Nc": Nc,
        "Ns": Nc * V,
        "sum_abs_m": float(abs_m.sum()),
        "sum_m2": float(m2.sum()),
        "sum_m4": float(m4.sum()),
        "sum_abs_phi": float(np.abs(phi).sum()),
        "sum_p2": float(p2.sum()),
        "sum_p4": float(p4.sum()),
        "sum_p6": float(p6.sum()),
        "sum_p8": float(p8.sum()),
    }


def obs_from_sums(s, V):
    Nc = s["Nc"]; Ns = s["Ns"]
    abs_m_mean = s["sum_abs_m"] / Nc
    m2_mean = s["sum_m2"] / Nc
    m4_mean = s["sum_m4"] / Nc
    chi = V * (m2_mean - abs_m_mean * abs_m_mean)
    U4 = 1.0 - m4_mean / (3.0 * m2_mean * m2_mean)

    s2 = s["sum_p2"] / Ns
    s4 = s["sum_p4"] / Ns
    s6 = s["sum_p6"] / Ns
    s8 = s["sum_p8"] / Ns
    abs_phi_mean = s["sum_abs_phi"] / Ns
    k2 = s2
    k4 = s4 - 3.0 * s2 * s2
    k6 = s6 - 15.0 * s4 * s2 + 30.0 * s2 ** 3
    k8 = (s8
          - 28.0 * s6 * s2
          - 35.0 * s4 * s4
          + 420.0 * s4 * s2 * s2
          - 630.0 * s2 ** 4)
    U4_s = 1.0 - s4 / (3.0 * s2 * s2)
    return {
        "abs_phi_bar": float(abs_m_mean),
        "chi": float(chi),
        "U4": float(U4),
        "abs_phi": float(abs_phi_mean),
        "kappa2_s": float(k2),
        "kappa4_s": float(k4),
        "kappa6_s": float(k6),
        "kappa8_s": float(k8),
        "U4_s": float(U4_s),
    }


def block_jackknife(x_flat, L, n_blocks=32):
    """Fast block jackknife via precomputed per-block sums and leave-one-out subtraction."""
    Nc = x_flat.shape[0]
    V = x_flat.shape[1]
    bs = Nc // n_blocks
    if bs == 0:
        full = obs_from_sums(block_sums(x_flat), V)
        return full, {k: float("nan") for k in full}
    x_flat = x_flat[: bs * n_blocks]
    per_block = [block_sums(x_flat[i * bs : (i + 1) * bs]) for i in range(n_blocks)]
    total = {k: sum(b[k] for b in per_block) for k in per_block[0]}
    full = obs_from_sums(total, V)
    pseudo = []
    for i in range(n_blocks):
        leave = {k: total[k] - per_block[i][k] for k in total}
        pseudo.append(obs_from_sums(leave, V))
    errs = {}
    for k in full:
        vals = np.array([p[k] for p in pseudo])
        errs[k] = float(np.sqrt((n_blocks - 1) * np.mean((vals - vals.mean()) ** 2)))
    return full, errs


def fmt_err(value, error, sig_digits=1):
    """Format 'val(err)' with error written as last digits.

    Examples (sig_digits=1):
        fmt_err(1.45307, 0.00035) → '1.4531(4)'
        fmt_err(5.529, 0.196)     → '5.5(2)'
        fmt_err(169.73, 0.57)     → '169.7(6)'
        fmt_err(-6329, 28.3)      → '-6329(28)' (sig_digits=2 here gives this)

    Falls back to plain f'{value}' if error is 0 or non-finite.
    """
    if not np.isfinite(error) or error == 0:
        return f"{value:g}"
    import math
    exp_err = int(math.floor(math.log10(abs(error))))
    # Round error to sig_digits significant figures
    factor = 10 ** (exp_err - sig_digits + 1)
    err_rounded = round(error / factor) * factor
    # Re-evaluate exponent in case rounding changed magnitude (e.g., 0.95 → 1.0)
    exp_err = int(math.floor(math.log10(abs(err_rounded)))) if err_rounded != 0 else exp_err
    decimals = max(0, -(exp_err - sig_digits + 1))
    if decimals == 0:
        val_str = f"{round(value):d}"
        err_int = int(round(err_rounded))
    else:
        val_str = f"{value:.{decimals}f}"
        err_int = int(round(err_rounded * 10 ** decimals))
    return f"{val_str}({err_int})"


# LaTeX display names for the 9 observables
TEX_NAMES = {
    "abs_phi_bar": r"$\langle|\bar\phi|\rangle$",
    "chi": r"$\chi$",
    "U4": r"$U_4$",
    "abs_phi": r"$\langle|\phi|\rangle$",
    "kappa2_s": r"$\kappa_2^{(s)}$",
    "kappa4_s": r"$\kappa_4^{(s)}$",
    "kappa6_s": r"$\kappa_6^{(s)}$",
    "kappa8_s": r"$\kappa_8^{(s)}$",
    "U4_s": r"$U_4^{(s)}$",
}


def emit_tex_tables(args, epochs, all_results, summary, filter_info, out_tex):
    obs_keys = list(all_results[epochs[0]]["vals"].keys())

    lines = []
    lines.append(r"% Auto-generated by analyze_diagnostic_observables.py")
    lines.append(r"% Late-training snapshot ensemble — outlier-filtered, Z2-symmetrized.")
    lines.append("")
    # --- Table A: summary (publication-ready) ---
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Late-training snapshot ensemble for the 2D $\phi^4$ diffusion model "
                 r"at $\kappa=0.28$, $\lambda=0.022$, $L=128$. "
                 r"Values are averages over six fine-tune snapshots spanning epochs "
                 fr"{min(epochs)}--{max(epochs)}. "
                 r"$\sigma_{\rm snap}$ is the standard deviation across snapshots; "
                 r"$\delta_{\rm stat}$ is the mean within-snapshot block-jackknife error "
                 r"on Z2-symmetrized samples ($N=1024$ each, "
                 fr"filtered to $\phi_{{\rm site}} \in [{args.norm_min:.3f},\,{args.norm_max:.3f}]$). "
                 r"$r = \sigma_{\rm snap}/\delta_{\rm stat}$ quantifies the resolvable "
                 r"snapshot variability relative to statistical noise.}")
    lines.append(r"\label{tab:snapshot_summary}")
    lines.append(r"\begin{ruledtabular}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"observable & mean & $\sigma_{\rm snap}$ & $\delta_{\rm stat}$ & $r$ \\")
    lines.append(r"\hline")
    for s in summary:
        k = s["obs"]
        lines.append(f"{TEX_NAMES.get(k, k)} & "
                     f"{fmt_err(s['mean'], s['delta_within'], sig_digits=1)} & "
                     f"{s['sigma_between']:.2e} & {s['delta_within']:.2e} & "
                     f"{s['ratio_r']:.2f} \\\\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{ruledtabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table B: per-snapshot detail (split into mag + site parts to fit page) ---
    mag_keys = ["abs_phi_bar", "chi", "U4"]
    site_keys = ["abs_phi", "kappa2_s", "kappa4_s", "kappa6_s", "kappa8_s", "U4_s"]

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\caption{Per-snapshot observables (outlier-filtered, Z2-symmetrized). "
                 r"$N_{\rm kept}$ is the number of configurations retained after discarding any "
                 r"sample with at least one site value outside the training-data range "
                 fr"$[{args.norm_min:.3f},\,{args.norm_max:.3f}]$ "
                 r"(equivalently, outside $[-1,1]$ in the normalized field). "
                 r"Errors are block-jackknife standard deviations (32 blocks).}")
    lines.append(r"\label{tab:snapshot_detail}")
    lines.append(r"\begin{ruledtabular}")
    n_cols = 1 + 1 + len(mag_keys) + len(site_keys)
    col_spec = "l" + "c" * (n_cols - 1)
    lines.append(fr"\begin{{tabular}}{{{col_spec}}}")
    hdr = "epoch & $N_{\\rm kept}$ & " + " & ".join(TEX_NAMES[k] for k in mag_keys + site_keys) + r" \\"
    lines.append(hdr)
    lines.append(r"\hline")
    for e in epochs:
        row = [str(e), str(filter_info[e]["n_kept"])]
        for k in mag_keys + site_keys:
            row.append(fmt_err(all_results[e]["vals"][k], all_results[e]["errs"][k], sig_digits=1))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{ruledtabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX tables: {out_tex}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tags", nargs="+", required=True,
                   help="Tags matching --ep used in sampling, e.g. ep10000_n1024 ...")
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--n_blocks", type=int, default=32)
    p.add_argument("--out_tag", default="n1024_z2")
    p.add_argument("--norm_min", type=float, default=-4.645815254332451,
                   help="Physical lower bound. Configs with any site < norm_min are dropped.")
    p.add_argument("--norm_max", type=float, default=4.7210209675109,
                   help="Physical upper bound. Configs with any site > norm_max are dropped.")
    p.add_argument("--no_filter", action="store_true",
                   help="Disable outlier filtering")
    args = p.parse_args()

    data_dir = os.path.join(args.run_dir, "data")
    epochs = []
    for t in args.tags:
        m = re.match(r"ep(\d+)", t)
        if not m:
            raise ValueError(f"Tag '{t}' does not match epNNNNN pattern")
        epochs.append(int(m.group(1)))

    all_results = {}
    filter_info = {}
    for tag, ep in zip(args.tags, epochs):
        fname = os.path.join(data_dir, f"samples_em_steps{args.num_steps}_{tag}.npy")
        x, L = load_configs(fname)
        n_total = x.shape[0]
        if args.no_filter:
            n_kept = n_total
            n_dropped = 0
        else:
            keep = (x.min(axis=1) >= args.norm_min) & (x.max(axis=1) <= args.norm_max)
            x = x[keep]
            n_kept = int(keep.sum())
            n_dropped = n_total - n_kept
        filter_info[ep] = {"n_total": n_total, "n_kept": n_kept, "n_dropped": n_dropped}
        x_z2 = z2_augment(x)
        print(f"\n[{tag}] N_total={n_total}  N_kept={n_kept} (dropped {n_dropped})  post-Z2={x_z2.shape[0]}")
        full, errs = block_jackknife(x_z2, L, n_blocks=args.n_blocks)
        all_results[ep] = {"vals": full, "errs": errs}
        for k in full:
            print(f"   {k:>12s} = {full[k]:.6e}  +/- {errs[k]:.2e}")

    obs_keys = list(all_results[epochs[0]]["vals"].keys())

    # Per-observable snapshot spread
    print("\n========== Snapshot spread summary ==========")
    header = f"{'observable':>14s}  {'mean':>14s}  {'sigma_between':>14s}  {'delta_within':>14s}  {'r':>8s}  {'drift_100ep':>14s}"
    print(header)
    summary = []
    for k in obs_keys:
        vals = np.array([all_results[e]["vals"][k] for e in epochs])
        errs = np.array([all_results[e]["errs"][k] for e in epochs])
        sigma_b = float(np.std(vals, ddof=1))
        delta_w = float(np.mean(errs))
        r = sigma_b / delta_w if delta_w > 0 else float("inf")
        slope, _ = np.polyfit(epochs, vals, 1)
        drift = slope * (max(epochs) - min(epochs))
        summary.append(dict(obs=k, vals=vals.tolist(), errs=errs.tolist(),
                            mean=float(vals.mean()), sigma_between=sigma_b,
                            delta_within=delta_w, ratio_r=r, drift=float(drift)))
        print(f"{k:>14s}  {vals.mean():>14.6e}  {sigma_b:>14.4e}  {delta_w:>14.4e}  {r:>8.3f}  {drift:>+14.4e}")

    out_txt = os.path.join(args.run_dir, f"diagnostic_observables_{args.out_tag}.txt")
    with open(out_txt, "w") as f:
        f.write("=== Per-snapshot values (post-Z2, block-jackknife errors) ===\n")
        f.write(f"{'ep':>6s}  " + "  ".join(f"{k:>26s}" for k in obs_keys) + "\n")
        for e in epochs:
            row = f"{e:>6d}  "
            for k in obs_keys:
                v = all_results[e]['vals'][k]
                dv = all_results[e]['errs'][k]
                row += f"{v:>12.4e}+-{dv:<11.2e}  "
            f.write(row.rstrip() + "\n")
        f.write("\n=== Snapshot spread summary ===\n")
        f.write(header + "\n")
        for s in summary:
            f.write(f"{s['obs']:>14s}  {s['mean']:>14.6e}  {s['sigma_between']:>14.4e}  {s['delta_within']:>14.4e}  {s['ratio_r']:>8.3f}  {s['drift']:>+14.4e}\n")
    print(f"\nSaved {out_txt}")

    out_json = os.path.join(args.run_dir, f"diagnostic_observables_{args.out_tag}.json")
    with open(out_json, "w") as f:
        json.dump({"epochs": epochs, "snapshots": all_results, "summary": summary}, f, indent=2)
    print(f"Saved {out_json}")

    # 9-panel plot
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, k in zip(axes.flatten(), obs_keys):
        vals = np.array([all_results[e]["vals"][k] for e in epochs])
        errs = np.array([all_results[e]["errs"][k] for e in epochs])
        sigma_b = float(np.std(vals, ddof=1))
        delta_w = float(np.mean(errs))
        r = sigma_b / delta_w if delta_w > 0 else float("inf")
        ax.errorbar(epochs, vals, yerr=errs, marker="o", capsize=3, lw=1)
        ax.set_title(f"{k}  (r={r:.2f})")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.run_dir, f"diagnostic_observables_{args.out_tag}.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")

    out_tex = os.path.join(args.run_dir, f"diagnostic_observables_{args.out_tag}.tex")
    emit_tex_tables(args, epochs, all_results, summary, filter_info, out_tex)


if __name__ == "__main__":
    main()
