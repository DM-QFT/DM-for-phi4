"""3D betaconf_sensitivity figure for the draft, cjp style, from saved data."""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(HERE.parent / "cjp.mplstyles")
# node's texlive lacks type1cm.sty -> emulate the look with STIX (bundled)
mpl.rcParams.update({"text.usetex": False, "font.family": "STIXGeneral",
                     "mathtext.fontset": "stix"})

OUT = HERE.parents[1] / "paper" / "betaconfig" / "assets" / "figs" / "3dphi4"
OUT.mkdir(parents=True, exist_ok=True)
RES = HERE.parent / "results" / "verification"

hmc = json.load(open(RES / "c2_hmc_3d_L64.json"))
data = {L: sum((json.load(open(RES / f"c2_results_3d_L{L}_{s}.json"))
                for s in ("log", "linear")), []) for L in (32, 64)}

fig, ax = plt.subplots()
cols4 = {(32, "log"): "navy", (32, "linear"): "crimson",
         (64, "log"): "green", (64, "linear"): "chocolate"}
for Lv, ls, mkk in [(32, "-", ("o", "s")), (64, "--", ("^", "v"))]:
    for sched, mk in [("log", mkk[0]), ("linear", mkk[1])]:
        r = [q for q in data[Lv] if q["sched"] == sched and q["b"] > 0.5]
        ax.errorbar([q["steps"] for q in r], [q["b"] for q in r],
                    yerr=[q["be"] for q in r], marker=mk, ms=5, ls=ls, lw=0.8,
                    color=cols4[(Lv, sched)], capsize=2,
                    label=f"{sched.capitalize()}, $L={Lv}$")
ax.axhspan(hmc["b"] - hmc["be"], hmc["b"] + hmc["be"], color="gray", alpha=0.4,
           label="HMC")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"EM steps $N_t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}$")
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "betaconf_sensitivity.pdf")
fig.savefig(RES / "figs" / "draft_betaconf_sensitivity_3d.png", dpi=150)
print(f"saved {OUT}/betaconf_sensitivity.pdf")
