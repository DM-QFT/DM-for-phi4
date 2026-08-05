"""
HMC-only physical-observable table for 2D phi^4, lattice sizes L=16,32,64
across kappa = 0.26 (symmetric), 0.2705 (near-critical), 0.28 (broken).

Reproduces the column set and conventions of \\Tab{tab:multiL_HMC_cumulants}
in paper/DM/DM.tex, restricted to the HMC (Wolff+FAHMC) reference. To guarantee
the estimator definitions are identical to the ones that produced the paper
table, the dimension-agnostic estimators are imported directly from the
authoritative generator

    3Dphi4/analysis/physical observable/compare_L64_multiL_HMC_cumulants.py

and only the 2D action and the 2D data loader are defined locally.

Pipeline (matches the 3D reference):
  1. Load raw HMC cfgs (physical field space), shape (N, L, L).
  2. Z_2-symmetrise in raw space:  cat([s, -s])  ->  <M> = 0 exactly.
  3. Global/action observables on the per-cfg magnetisation M = <phi>_V:
       <|phi-bar|>, chi = V(<M^2> - <|M|>^2), <S>/V, U_4 = 1 - <M^4>/(3<M^2>^2)
  4. Zero-mode connected cumulants kappa^(0)_2,4,6,8 of M (recursion; odd
     cumulants vanish by Z_2).
  5. Single-site cumulants (persite_avg, DM.tex Eq. eq:site_cumulants_avg):
       <|phi|>, kappa^(site)_2,4,6,8, U_4^(site).
     The per-site-then-average estimator (default) is the definition stated in
     the DM.tex caption. The published 2D block of tab:multiL_HMC_cumulants was
     actually produced with the *pool* estimator (pool every (cfg,x), Aarts
     et al. 2410.21212v2 eq. (5.11)); the two agree on <|phi|>, kappa_2,
     kappa_4 and differ only in kappa_6, kappa_8 by an O(Var_x m_k / N_cfg)
     finite-N term. Use --site_method pool to reproduce the published numbers.
  Errors are bootstrap 1-sigma over configurations (paired over the Z_2 halves
  by default, so every draw stays Z_2-symmetric).

Outputs results/hmc_observables_L16_64/{hmc_observables_L16_64.csv,
summary.txt, hmc_observables_L16_64.tex}.

Usage (from 2Dphi4/):
    python analysis/hmc_observables_table_2d.py
    python analysis/hmc_observables_table_2d.py --Ls 16 32 64 128   # +validation
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]                       # repo root
sys.path.insert(0, str(REPO))                # phi4_action
sys.path.insert(0, str(REPO / "3Dphi4"))     # _layout (imported by the reference)

from phi4_action import phi4_action as _phi4_action_unified


# --- import the verified estimators from the 3D generator (path has a space) ---
_REF_PATH = REPO / "3Dphi4" / "analysis" / "physical observable" \
    / "compare_L64_multiL_HMC_cumulants.py"
_spec = importlib.util.spec_from_file_location("ref_cumulants_3d", _REF_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

magnetisation             = ref.magnetisation
bootstrap_stats           = ref.bootstrap_stats
bootstrap_stats_per_site  = ref.bootstrap_stats_per_site
bootstrap_S_per_V         = ref.bootstrap_S_per_V
_val_err                  = ref._val_err


# ---------------------------------------------------------------------------
# 2D-specific I/O and action
# ---------------------------------------------------------------------------

def load_hmc_2d(path, max_n=None):
    """HMC cfgs in physical phi space, shape (N, L, L)."""
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    return cfgs if max_n is None else cfgs[:max_n]


def action_per_cfg(cfgs, k, l):
    """Per-cfg phi^4 action S(phi), 2D. cfgs in physical space (N, L, L);
    pass phi_min=-1, phi_max=1 so the unified-action denormalisation is the
    identity."""
    phi = torch.as_tensor(cfgs[:, None], dtype=torch.float64)   # (N,1,L,L)
    S = _phi4_action_unified(phi, k=k, l=l, phi_min=-1.0, phi_max=1.0,
                             spatial_dims=2)
    return S.numpy()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fmt(v, e, p=5):
    if not np.isfinite(v):
        return f"{'nan':>14}"
    return f"{v:.{p}g}({e:.2g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--kappas", type=str, nargs="+",
                    default=["0.26", "0.2705", "0.28"])
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hmc_max", type=int, default=None,
                    help="cap on HMC cfgs (default: use all)")
    ap.add_argument("--independent_bootstrap", action="store_true",
                    help="use independent-index bootstrap on the full 2N array "
                         "instead of the default paired (per Z_2-half) resampler")
    ap.add_argument("--site_method", choices=["pool", "persite_avg"],
                    default="persite_avg",
                    help="single-site cumulant estimator. 'persite_avg' is the "
                         "DM.tex Eq. (eq:site_cumulants_avg) definition (default); "
                         "'pool' reproduces the published 2D table numbers.")
    ap.add_argument("--out_dir", type=str,
                    default=str(HERE.parents[2] / "results"
                                / "hmc_observables_L16_64"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for L in args.Ls:
        V = L * L
        for k_str in args.kappas:
            k = float(k_str)
            path = (REPO / "2Dphi4" / "trainingdata"
                    / f"cfgs_wolff_fahmc_k={k_str}_l={args.lam:g}_{L}^2.jld2")
            if not path.exists():
                print(f"[skip] missing {path}")
                continue

            hmc = load_hmc_2d(str(path), max_n=args.hmc_max)
            # Z_2 symmetrise in raw space (physical phi -> -phi).
            hmc_z2 = np.concatenate([hmc, -hmc], axis=0)
            n_raw = None if args.independent_bootstrap else hmc.shape[0]
            print(f"\n========== L={L}  κ={k_str}  "
                  f"(N_raw={hmc.shape[0]}, N_z2={hmc_z2.shape[0]}) ==========")

            # ---- global / action + zero-mode cumulants ----
            M = magnetisation(hmc_z2)
            g_mean, g_err = bootstrap_stats(M, V, n_boot=args.n_boot,
                                            seed=args.seed, n_raw=n_raw)
            S = action_per_cfg(hmc_z2, k=k, l=args.lam)
            sv_m, sv_e = bootstrap_S_per_V(S, V, n_boot=args.n_boot,
                                           seed=args.seed, n_raw=n_raw)
            g_mean["S_V"], g_err["S_V"] = sv_m, sv_e

            # ---- single-site cumulants (persite_avg = paper definition) ----
            s_mean, s_err = bootstrap_stats_per_site(
                hmc_z2, n_boot=args.n_boot, seed=args.seed, n_raw=n_raw,
                method=args.site_method)

            row = dict(L=L, kappa=k_str, N_raw=hmc.shape[0], V=V)
            for kk in ["abs_M", "chi", "S_V", "U4", "k2", "k4", "k6", "k8"]:
                row[kk] = g_mean[kk]
                row["d_" + kk] = g_err[kk]
            for kk in ["abs_phi", "k2", "k4", "k6", "k8", "U4_site"]:
                row["site_" + kk] = s_mean[kk]
                row["d_site_" + kk] = s_err[kk]
            rows.append(row)

            print(f"  <|phi_bar|> = {fmt(g_mean['abs_M'], g_err['abs_M'])}   "
                  f"chi = {fmt(g_mean['chi'], g_err['chi'], 4)}   "
                  f"<S>/V = {fmt(g_mean['S_V'], g_err['S_V'])}   "
                  f"U4 = {fmt(g_mean['U4'], g_err['U4'])}")
            print(f"  k0_2={fmt(g_mean['k2'], g_err['k2'])}  "
                  f"k0_4={fmt(g_mean['k4'], g_err['k4'])}  "
                  f"k0_6={fmt(g_mean['k6'], g_err['k6'])}  "
                  f"k0_8={fmt(g_mean['k8'], g_err['k8'])}")
            print(f"  <|phi|>={fmt(s_mean['abs_phi'], s_err['abs_phi'])}  "
                  f"k2s={fmt(s_mean['k2'], s_err['k2'])}  "
                  f"k4s={fmt(s_mean['k4'], s_err['k4'])}  "
                  f"k6s={fmt(s_mean['k6'], s_err['k6'])}  "
                  f"k8s={fmt(s_mean['k8'], s_err['k8'])}  "
                  f"U4s={fmt(s_mean['U4_site'], s_err['U4_site'])}")

    if not rows:
        print("No data found — nothing written.")
        return

    # -------- CSV --------
    csv_path = out_dir / "hmc_observables_L16_64.csv"
    cols = ["L", "kappa", "N_raw", "V",
            "abs_M", "d_abs_M", "chi", "d_chi", "S_V", "d_S_V", "U4", "d_U4",
            "k2", "d_k2", "k4", "d_k4", "k6", "d_k6", "k8", "d_k8",
            "site_abs_phi", "d_site_abs_phi",
            "site_k2", "d_site_k2", "site_k4", "d_site_k4",
            "site_k6", "d_site_k6", "site_k8", "d_site_k8",
            "site_U4_site", "d_site_U4_site"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nsaved {csv_path}")

    # -------- text table --------
    txt_path = out_dir / "summary.txt"
    write_txt_table(rows, txt_path, args)
    print(f"saved {txt_path}")

    # -------- LaTeX: paper fragment (\input into DM.tex) + standalone preview --
    tex_path = out_dir / "hmc_observables_L16_64.tex"
    write_latex_table(rows, tex_path, args)
    print(f"saved {tex_path}")
    prev_path = out_dir / "hmc_observables_L16_64_preview.tex"
    write_latex_preview(rows, prev_path, args)
    print(f"saved {prev_path}")


def write_txt_table(rows, out, args):
    def c(v, e, p=4):
        return fmt(v, e, p)

    hdr = (f"{'L':>4} {'κ':>7} "
           f"{'<|φ̄|>':>13} {'χ':>13} {'<S>/V':>13} {'U4':>13} "
           f"{'κ0_2':>13} {'κ0_4':>13} {'κ0_6':>13} {'κ0_8':>13} "
           f"{'<|φ|>':>13} {'κ2_s':>13} {'κ4_s':>13} {'κ6_s':>13} "
           f"{'κ8_s':>13} {'U4_s':>13}")
    lines = [
        f"HMC (Wolff+FAHMC) physical observables — 2D phi^4, λ={args.lam:g}",
        "columns follow paper/DM/DM.tex Tab. tab:multiL_HMC_cumulants (HMC only)",
        "Z_2-symmetrised raw field space; cumulants by moment recursion;",
        f"bootstrap 1σ over cfgs (n_boot={args.n_boot}, "
        f"{'independent' if args.independent_bootstrap else 'paired'}).",
        "chi = V(<M^2>-<|M|>^2),  U4 = 1 - <M^4>/(3<M^2>^2),",
        f"κ0_n = zero-mode cumulants of M=φ̄,  κn_s = single-site "
        f"(method={args.site_method}; 'pool' reproduces the published L=128 table).",
        "",
        hdr, "-" * len(hdr),
    ]
    for r in rows:
        lines.append(
            f"{r['L']:>4} {r['kappa']:>7} "
            f"{c(r['abs_M'], r['d_abs_M']):>13} {c(r['chi'], r['d_chi']):>13} "
            f"{c(r['S_V'], r['d_S_V']):>13} {c(r['U4'], r['d_U4']):>13} "
            f"{c(r['k2'], r['d_k2']):>13} {c(r['k4'], r['d_k4']):>13} "
            f"{c(r['k6'], r['d_k6']):>13} {c(r['k8'], r['d_k8']):>13} "
            f"{c(r['site_abs_phi'], r['d_site_abs_phi']):>13} "
            f"{c(r['site_k2'], r['d_site_k2']):>13} "
            f"{c(r['site_k4'], r['d_site_k4']):>13} "
            f"{c(r['site_k6'], r['d_site_k6']):>13} "
            f"{c(r['site_k8'], r['d_site_k8']):>13} "
            f"{c(r['site_U4_site'], r['d_site_U4_site']):>13}"
        )
    out.write_text("\n".join(lines) + "\n")


def _tabular_lines(rows):
    """The booktabs `tabular` body shared by the paper fragment and the
    standalone preview."""
    Ls = sorted({r["L"] for r in rows})
    by_L = {L: [r for r in rows if r["L"] == L] for L in Ls}
    lines = [
        r"\begin{tabular}{llcccccccccccccc}",
        r"\toprule",
        (r" & & \multicolumn{4}{c}{Global / action} "
         r"& \multicolumn{4}{c}{Zero-mode cumulants $\kappa^{(0)}_n$} "
         r"& \multicolumn{6}{c}{Single-site cumulants} \\"),
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-16}",
        (r"$L$ & $\kappa$ & $\langle|\bar\phi|\rangle$ & $\chi$ "
         r"& $\langle S\rangle/V$ & $U_4$ "
         r"& $\kappa^{(0)}_2$ & $\kappa^{(0)}_4$ & $\kappa^{(0)}_6$ & $\kappa^{(0)}_8$ "
         r"& $\langle|\phi|\rangle$ & $\kappa_2^{(\mathrm{site})}$ "
         r"& $\kappa_4^{(\mathrm{site})}$ & $\kappa_6^{(\mathrm{site})}$ "
         r"& $\kappa_8^{(\mathrm{site})}$ & $U_4^{(\mathrm{site})}$ \\"),
        r"\midrule",
    ]
    for li, L in enumerate(Ls):
        for ri, r in enumerate(by_L[L]):
            L_cell = f"${L}$" if ri == 0 else ""
            cells = [
                L_cell, f"${r['kappa']}$",
                _val_err(r["abs_M"], r["d_abs_M"]),
                _val_err(r["chi"], r["d_chi"]),
                _val_err(r["S_V"], r["d_S_V"]),
                _val_err(r["U4"], r["d_U4"]),
                _val_err(r["k2"], r["d_k2"]),
                _val_err(r["k4"], r["d_k4"]),
                _val_err(r["k6"], r["d_k6"]),
                _val_err(r["k8"], r["d_k8"]),
                _val_err(r["site_abs_phi"], r["d_site_abs_phi"]),
                _val_err(r["site_k2"], r["d_site_k2"]),
                _val_err(r["site_k4"], r["d_site_k4"]),
                _val_err(r["site_k6"], r["d_site_k6"]),
                _val_err(r["site_k8"], r["d_site_k8"]),
                _val_err(r["site_U4_site"], r["d_site_U4_site"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if li < len(Ls) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def write_latex_table(rows, out, args):
    """Paper fragment: a `table*` to be \\input into DM.tex (uses the paper's
    \\Tab/\\Eq macros and booktabs/graphicx loaded by DM.tex)."""
    Ls = sorted({r["L"] for r in rows})
    head = [
        r"% auto-generated by 2Dphi4/analysis/hmc_observables_table_2d.py",
        r"% paper fragment: \input into DM.tex (needs booktabs, graphicx, \Tab, \Eq)",
        r"% HMC (Wolff+FAHMC) reference, raw field space, Z_2-symmetrised cat([phi,-phi])",
        r"\begin{table*}[!tbp]",
        r"\centering",
        (r"\caption{HMC (FA-HMC--Wolff) physical observables for the 2D $\phi^4$ "
         rf"theory at $\lambda={args.lam:g}$, for lattice sizes "
         + ", ".join(f"$L={L}$" for L in Ls) + " and "
         r"$\kappa=0.26$ (symmetric), $0.2705\simeq\kappa_c$ (near-critical), "
         r"$0.28$ (broken). Conventions follow "
         r"\Tab{tab:multiL_HMC_cumulants}. All quantities are in the physical "
         r"field space and each ensemble is $Z_2$-symmetrised by concatenating "
         r"$\{\phi,-\phi\}$ so that $\langle M\rangle=0$ exactly and the odd "
         r"cumulants vanish. The single-site cumulants are evaluated per site "
         r"and then averaged over sites (\Eq{eq:site_cumulants_avg}). Errors are "
         r"bootstrap $1\sigma$ over configurations.}"),
        r"\label{tab:hmc_obs_L16_64_2d}",
        r"\resizebox{\textwidth}{!}{%",
    ]
    tail = [r"%", r"}", r"\end{table*}"]
    lines = head + _tabular_lines(rows) + tail
    out.write_text("\n".join(lines) + "\n")


def write_latex_preview(rows, out, args):
    """Standalone document that compiles on its own (pdflatex) to preview the
    table. Uses the `standalone` class with a bare `tabular` (no float, no
    \\resizebox), which crops tightly to the table."""
    lines = [
        r"% auto-generated by 2Dphi4/analysis/hmc_observables_table_2d.py",
        r"% standalone PREVIEW: compile directly with `pdflatex` to view the table.",
        r"% The paper version is hmc_observables_L16_64.tex (\input into DM.tex).",
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
    ] + _tabular_lines(rows) + [r"\end{document}"]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
