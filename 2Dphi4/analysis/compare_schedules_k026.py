"""Compare DM samples generated with log vs linear schedule (num_steps=4000)
against HMC, on κ=0.26 (sym phase, where G(1) showed −3.3% bias).

Inputs (in same run/data dir):
  samples_em_steps4000_epoch=10000_log.npy
  samples_em_steps4000_epoch=10000_linear.npy
Plus the existing num_steps=2000 log run for reference.

Output: results/dm_vs_hmc_L128/schedule_compare_k026.txt + .pdf
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))
sys.path.append(str(HERE.parents[2]))

import h5py, numpy as np, matplotlib.pyplot as plt
from phi4_action import phi4_action as _phi4_action_unified
import torch


RUN = HERE.parents[1] / "runs" / "phi4_L128_k0.26_l0.022_ncsnpp_sigma300"
HMC = HERE.parents[1] / "trainingdata" / "cfgs_wolff_fahmc_k=0.26_l=0.022_128^2.jld2"
OUT = HERE.parents[1] / "results" / "dm_vs_hmc_L128"
OUT.mkdir(parents=True, exist_ok=True)

K, LAM = 0.26, 0.022
N_USE = 1024            # after Z2 augment of 512 DM samples
JK_BINS = 32


def load_dm(path, z2=True):
    arr = np.load(path)
    cfgs = np.transpose(arr, (2, 0, 1)).astype(np.float64)
    if z2:
        cfgs = np.stack([cfgs, -cfgs], axis=1).reshape(-1, *cfgs.shape[1:])
    return cfgs


def load_hmc(n_match):
    with h5py.File(HMC, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    return cfgs[-n_match:]


def action(cfgs):
    nb = np.roll(cfgs, 1, axis=1) + np.roll(cfgs, 1, axis=2)
    return (-2 * K * cfgs * nb + (1 - 2 * LAM) * cfgs ** 2
            + LAM * cfgs ** 4).sum(axis=(1, 2))


def G1(cfgs):
    nb = np.roll(cfgs, 1, axis=1) + np.roll(cfgs, 1, axis=2)
    return (cfgs * nb).mean() / 2.0


def jk(values, func=np.mean, n_bins=JK_BINS):
    x = np.asarray(values, dtype=np.float64)
    N = x.shape[0]
    n_bins = int(min(n_bins, N)); bs = N // n_bins
    x = x[: bs * n_bins]
    binned = x.reshape(n_bins, bs, *x.shape[1:]).mean(axis=1)
    full = np.asarray(func(binned, axis=0))
    jk = np.empty((n_bins,) + full.shape, dtype=np.float64)
    for i in range(n_bins):
        jk[i] = np.asarray(func(np.delete(binned, i, axis=0), axis=0))
    err = np.sqrt((n_bins - 1) * np.mean((jk - full) ** 2, axis=0))
    return (float(full), float(err)) if full.shape == () else (full, err)


def report_summary(name, cfgs):
    V = cfgs.shape[-1] ** 2
    m = cfgs.mean(axis=(1, 2))
    am = np.abs(m); m2 = m ** 2; m4 = m ** 4
    S = action(cfgs)
    g1 = G1(cfgs)
    p2 = (cfgs ** 2).mean()
    out = dict(N=cfgs.shape[0])
    out["abs_m"] = jk(am)
    out["chi"] = jk(np.stack([am, m2], axis=-1),
                    func=lambda a, axis=0: V * (a.mean(axis=axis)[..., 1]
                                                 - a.mean(axis=axis)[..., 0] ** 2))
    out["U4"] = jk(np.stack([m2, m4], axis=-1),
                   func=lambda a, axis=0: 1.0 - a.mean(axis=axis)[..., 1]
                                                / (3.0 * a.mean(axis=axis)[..., 0] ** 2))
    out["S_mean"] = jk(S)
    out["S_per_V"] = jk(S / V)
    out["G1"] = g1                                # not jackknifed (cheap single scalar)
    out["phi2"] = p2
    out["G1_over_phi2"] = g1 / p2
    return out, dict(m=m, S=S)


def fmt(v, e):
    return f"{v:.4g} ± {e:.2g}"


def main():
    hmc = load_hmc(N_USE)
    print(f"HMC: {hmc.shape}")
    sets = {"HMC": hmc}
    raw = {}
    raw["HMC"] = (hmc.mean(axis=(1, 2)), action(hmc))

    for tag, path_glob in [
        ("DM log   steps=2000", "samples_em_steps2000_epoch=10000.npy"),
        ("DM log   steps=4000", "samples_em_steps4000_epoch=10000_log.npy"),
        ("DM linear steps=4000", "samples_em_steps4000_epoch=10000_linear.npy"),
    ]:
        p = RUN / "data" / path_glob
        if not p.exists():
            print(f"  [skip] {p.name} not present"); continue
        c = load_dm(p, z2=True)
        # match to N_USE
        c = c[:N_USE] if c.shape[0] > N_USE else c
        sets[tag] = c
        print(f"  {tag}: {c.shape} (z2 augmented)")
        m = c.mean(axis=(1, 2)); S = action(c)
        raw[tag] = (m, S)

    print()
    rows = ["name", "N", "⟨|m|⟩", "χ", "U₄", "⟨S⟩", "⟨S/V⟩", "G(1)", "⟨φ²⟩", "G(1)/⟨φ²⟩"]
    widths = [22, 6, 14, 12, 14, 14, 14, 9, 9, 12]
    print("  ".join(f"{r:>{w}s}" for r, w in zip(rows, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    results = {}
    for name, c in sets.items():
        out, _ = report_summary(name, c)
        results[name] = out
        print("  ".join([
            f"{name:>{widths[0]}s}",
            f"{out['N']:>{widths[1]}d}",
            f"{fmt(*out['abs_m']):>{widths[2]}s}",
            f"{fmt(*out['chi']):>{widths[3]}s}",
            f"{fmt(*out['U4']):>{widths[4]}s}",
            f"{fmt(*out['S_mean']):>{widths[5]}s}",
            f"{fmt(*out['S_per_V']):>{widths[6]}s}",
            f"{out['G1']:>{widths[7]}.5f}",
            f"{out['phi2']:>{widths[8]}.5f}",
            f"{out['G1_over_phi2']:>{widths[9]}.5f}",
        ]))

    print()
    print("Δ vs HMC (DM / HMC − 1):")
    h = results["HMC"]
    for name in results:
        if name == "HMC": continue
        d = results[name]
        rel_S = (d["S_per_V"][0] - h["S_per_V"][0]) / h["S_per_V"][0]
        rel_G1 = (d["G1"] - h["G1"]) / h["G1"]
        rel_p2 = (d["phi2"] - h["phi2"]) / h["phi2"]
        rel_R = (d["G1_over_phi2"] - h["G1_over_phi2"]) / h["G1_over_phi2"]
        print(f"  {name:>22s}   ΔS/V={rel_S:+7.2%}   ΔG(1)={rel_G1:+7.2%}   "
              f"Δ⟨φ²⟩={rel_p2:+7.2%}   ΔG(1)/⟨φ²⟩={rel_R:+7.2%}")

    # ── plot S/V histograms and (m, S/V) joint scatter ──
    V = 128 * 128
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"HMC": "k", "DM log   steps=2000": "C0",
              "DM log   steps=4000": "C2", "DM linear steps=4000": "C3"}
    bins_S = None
    for name, (m, S) in raw.items():
        if name not in colors: continue
        sv = S / V
        if bins_S is None:
            allsv = np.concatenate([raw[k][1] / V for k in raw if k in colors])
            bins_S = np.linspace(np.percentile(allsv, 1), np.percentile(allsv, 99), 50)
        axes[0].hist(sv, bins=bins_S, density=True, alpha=0.35,
                     color=colors[name], label=name, histtype="stepfilled")
        axes[1].scatter(m, sv, s=4, alpha=0.4, color=colors[name], label=name)
    axes[0].set_xlabel("S/V"); axes[0].set_ylabel("density")
    axes[0].set_title("S/V distribution at κ=0.26 (sym)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("m"); axes[1].set_ylabel("S/V")
    axes[1].set_title("(m, S/V) joint scatter")
    axes[1].legend(fontsize=9, markerscale=2.5); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "schedule_compare_k026.pdf", bbox_inches="tight")
    print(f"\nsaved {OUT / 'schedule_compare_k026.pdf'}")


if __name__ == "__main__":
    main()
