"""Preview of cjp.mplstyles: two-panel APS-width figure + palette swatch."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
plt.style.use(HERE / "cjp.mplstyles")

epoch = np.array([1, 2, 3, 5, 8, 14, 22, 40, 70, 120, 300, 600, 1000, 1600, 4000, 8000, 2e4])
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(7.06, 2.47))  # size of the reference figure

ax = axes[0]
for i, (decay, lab) in enumerate([(0.35, r"$|\bar\phi|$"), (0.9, r"$\bar\phi^2$"),
                                  (1.6, r"$\bar\phi^4$"), (2.4, r"$S/V$")]):
    y = 10 ** (-decay * np.log10(epoch) - 2 * decay) * (1 + 0.15 * rng.standard_normal(epoch.size))
    ax.plot(epoch, y, label=lab)
ax.axhline(1.0, color="k", ls="--", lw=0.8, marker="")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel(r"$\mathrm{ESS_{eff,ref}}/N_\mathrm{DM}$")
ax.text(0.06, 0.30, "3D, L64\n" + r"$\kappa=0.1923$, (crit)", transform=ax.transAxes)
ax.legend(ncol=2, loc="lower right")

ax = axes[1]
for amp, lab in [(1.0, r"$G(k_\mathrm{min})$"), (0.3, r"$\langle G(k_\mathrm{mid})\rangle$"),
                 (0.08, r"$\langle G(k_\mathrm{large})\rangle$")]:
    y = amp * epoch ** 0.55 / (1 + epoch ** 0.55) * (1 + 0.1 * rng.standard_normal(epoch.size))
    ax.errorbar(epoch, y, yerr=0.12 * y, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel(r"$\mathrm{ESS_{eff,ref}}/N_\mathrm{DM}$")
ax.legend(loc="lower right")

fig.savefig(HERE / "style_demo.pdf")
fig.savefig(HERE / "style_demo.png", dpi=300)

# single-column reference: default figsize, one panel
fig2, ax = plt.subplots()
x = np.linspace(0, 1, 60)
for i in range(5):
    ax.plot(x[::6], np.exp(-(i + 1) * x[::6]) + 0.02 * i, label=f"$c_{i}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\langle\phi(t)\phi(0)\rangle$")
ax.legend(ncol=2)
fig2.savefig(HERE / "style_demo_1col.pdf")
fig2.savefig(HERE / "style_demo_1col.png", dpi=300)
print("wrote", HERE / "style_demo.pdf")
