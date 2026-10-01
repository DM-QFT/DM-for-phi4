"""ML4PS Fig. 4(c): wall time versus accuracy of the L=64 nonequilibrium benchmark."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "plots"))
from modelb_palette import ALT, DM, REFERENCE  # noqa: E402

plt.style.use(ROOT.parent / "plots" / "cjp_prl_nature.mplstyle")

DATA = ROOT / "results/gpu_solver_nonequilibrium_calibrated_metrics.npz"
OUT = ROOT / "draft/figures/prl/cost_accuracy_L64.pdf"
OBS = ("action", "phi2", "phi4")
NBOOT = 1000

# key, label, color, marker. This work: filled markers with a direct label. Libraries: open markers, legend.
METHODS = (
    ("dpm", "DPM-Solver-v3\n(this work)", DM, "o"),
    ("sosra_fused", "SOSRA\n(this work)", REFERENCE, "s"),
    ("sosra_sciml", "SOSRA (StochasticDiffEq.jl)", ALT, "s"),
    ("shark", "ShARK (Diffrax)", ALT, "^"),
    ("em_diffrax", "EM (Diffrax)", ALT, "D"),
    ("em_sciml", "EM (StochasticDiffEq.jl)", ALT, "v"),
    ("em_torchsde", "EM (torchsde)", ALT, "P"),
)
LABEL_OFFSET = {"dpm": ((7, 4), "left", "bottom"), "sosra_fused": ((7, -4), "left", "top")}

d = np.load(DATA)
ref = np.stack([d[f"baseline_{o}"] for o in OBS])


def eps(obs, idx):
    # PRL Eq. (S21): RMS over S/V, phi2, phi4 of the relative shift of the ensemble mean (PRL 里乘 100 作百分比)
    m = obs[:, idx].mean(1)
    r = ref[:, idx].mean(1)
    return np.sqrt(np.mean(((m - r) / np.abs(r)) ** 2))


n = ref.shape[1]
full = np.arange(n)
# paired resampling: the same trajectory indices for the reference and for every method
boot = np.random.default_rng(0).integers(0, n, (NBOOT, n))

fig, ax = plt.subplots(figsize=(3.375, 2.55))
for key, label, color, marker in METHODS:
    obs = np.stack([d[f"{key}_{o}"] for o in OBS])
    e = eps(obs, full)
    err = np.std([eps(obs, b) for b in boot])
    t = float(d[f"{key}_seconds"][0])
    print(f"{key:12s} t={t:8.3f} s  eps={e:.5f} +- {err:.5f}")
    filled = key in LABEL_OFFSET
    ax.errorbar(t, e, yerr=err, color=color, marker=marker, ls="none",
                markerfacecolor=color if filled else "none", ms=5.0,
                elinewidth=0.8, capsize=1.5, zorder=3,
                label=None if filled else label)
    if filled:
        (dx, dy), ha, va = LABEL_OFFSET[key]
        ax.annotate(label, (t, e), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9, color=color, linespacing=1.0)

# Printed at 0.48 of the NeurIPS text width (scale 0.78), so 8.5 pt here is about 6.6 pt on paper
ax.legend(loc="upper left", bbox_to_anchor=(0.09, 0.99), fontsize=8.5,
          handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0)
ax.set_xscale("log")
ax.set_xlim(1.5, 900)
ax.set_ylim(0, 0.0036)
ax.set_yticks((0, 0.001, 0.002, 0.003))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))  # 不要自动的 ×10⁻³ 偏移标注
ax.set_xlabel("wall time (s)")
ax.set_ylabel("RMS relative error")
ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9)
fig.savefig(OUT)
print("saved", OUT)
