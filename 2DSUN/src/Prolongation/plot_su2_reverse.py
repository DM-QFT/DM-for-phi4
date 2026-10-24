"""Final Wilson loops, action density, effective string tension and second moments
after reverse diffusion from Haar, fine and coarse starts, L=32, beta=2."""
import sys
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/su2_prolongation_reverse/reverse_tau1_1p5_3_train1024_epoch0500_3472772_n51200"
sys.path.insert(0, str(ROOT.parent / "plots"))
import modelb_palette  # noqa: E402,F401  registers the modelb_field colormap the style sheet names
plt.style.use(ROOT.parent / "plots/cjp_prl_nature.mplstyle")
# Four categories take the first four colors of the cycle.
vermillion, blue, purple, green = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]
RT = [(1, 1), (1, 2), (2, 2), (2, 3)]
# Haar runs all 2000 reverse steps from t=1, Sigma^2=36.9, so it carries no Sigma^2 label.
cases = [("haar_steps2000", "Haar", vermillion, "o", "-"),
         ("fine_steps1070", "Fine start", green, "o", "-"),
         ("fine_steps829", "Fine start", green, "s", "--"),
         ("relaxed_steps1070", "Coarse start", purple, "o", "-"),
         ("relaxed_steps829", "Coarse start", purple, "s", "--")]

with h5py.File(RUN.with_suffix(".h5")) as f:
    beta = f.attrs["beta"]
    w = f["wilson23"]
    sigma_exact = w.attrs["string_tension_exact"]
    ticks = ["Haar"]+[rf"$\Sigma^2={f[name].attrs['tau']:.2g}$" for name, *_ in cases[1:]]
    stat, sigma = {}, {}
    for name in ["reference", *(c[0] for c in cases)]:
        g = w[name]
        stat[name] = dict(zip(g["statistic_names"].asstr()[:], zip(g["statistic_mean"][:], g["statistic_se"][:])))
        sigma[name] = g["string_tension_mean"][:], g["string_tension_se"][:]
        P, dP = stat[name]["W11"]
        # Wilson action per plaquette, S/V = beta(1 - Tr U_p/2).
        stat[name]["S"] = beta*(1-P), beta*dP


def points(ax, key, ylabel):
    m, e = stat["reference"][key]
    ax.axhspan(m-e, m+e, color=blue, alpha=0.12, linewidth=0)
    ax.axhline(m, color=blue, linestyle="--", label="heatbath")
    for i, (name, _, color, marker, _) in enumerate(cases):
        ax.errorbar(i, *stat[name][key], color=color, marker=marker, linestyle="none")
    ax.set_xticks(range(len(cases)), ticks)
    ax.set_xlim(-0.5, len(cases)-0.5)
    ax.xaxis.set_minor_locator(NullLocator())
    # Steps of 2.5 would give loops near <W^2> = 1/4 a sixth decimal.
    ax.yaxis.set_major_locator(MaxNLocator("auto", steps=[1, 2, 5, 10]))
    # Plain decimals keep the offset text from sitting above the frame.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylabel(ylabel)


def groups(ax):
    # Two-line labels such as "Coarse start, Sigma^2=3" overlap, so the start type labels each pair.
    g = ax.secondary_xaxis("bottom")
    g.set_xticks([1.5, 3.5], ["Fine start", "Coarse start"])
    g.tick_params(length=0, pad=15)
    g.xaxis.set_minor_locator(NullLocator())
    g.spines["bottom"].set_visible(False)


for suffix, key, power in (("wilson_loops", "", ""), ("second_moments", "_second", "^2")):
    # Full width, with each panel's axes box 2.11 in tall as the style sheet asks.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.68), sharex=True)
    for ax, (r, t) in zip(axes.flat, RT):
        points(ax, f"W{r}{t}{key}", rf"$\langle W_{{{r}\times{t}}}{power}\rangle$")
    for ax in axes[1]:
        groups(ax)
    axes[0, 0].legend()
    fig.savefig(f"{RUN}_{suffix}.pdf")

fig, ax = plt.subplots()
points(ax, "S", r"$\langle S/V\rangle=\beta(1-\langle W_{1\times1}\rangle)$")
groups(ax)
ax.legend()
fig.savefig(f"{RUN}_action_density.pdf")

fig, ax = plt.subplots()
A = [r*t for r, t in RT]
ax.axhline(sigma_exact, color="0.3", linestyle=":")
ax.errorbar(A, *sigma["reference"], color=blue, marker="D")
for (name, label, color, marker, style), tick in zip(cases, ticks):
    ax.errorbar(A, *sigma[name], color=color, marker=marker, linestyle=style,
                label=label if tick == label else label+" "+tick)
# Seven legend rows cover the curves in one column, so the two references are labelled on their lines.
ax.annotate(rf"Exact $\sigma a^2={sigma_exact:.5f}$", (A[-1], sigma_exact), xytext=(-6, 2),
            textcoords="offset points", ha="right", va="bottom", color="0.3")
ax.annotate("heatbath", (A[1], sigma["reference"][0][1]), xytext=(3, -5),
            textcoords="offset points", ha="left", va="top", color=blue)
ax.set_xticks(A, [rf"${r}\times{t}$" for r, t in RT])
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel(r"$R\times T$")
ax.set_ylabel(r"$-\ln\langle W_{R\times T}\rangle/(RT)$")
# The legend reaches past the 2x2 points at single-column width, so it gets room above them.
ax.set_ylim(top=0.868)
ax.legend(loc="upper left")
fig.savefig(f"{RUN}_string_tension.pdf")
