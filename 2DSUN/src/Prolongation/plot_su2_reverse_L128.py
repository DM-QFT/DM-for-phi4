"""Final Wilson loops, their area dependence, action density, effective string tension and
second moments after reverse diffusion from Haar and coarse starts, L=128, beta=200."""
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results/su2_prolongation_reverse"
OUT = DIR / "reverse_L64_to_L128_beta200_Sigma2_1p5_to_0p02_n512_betanoise"
# Squares up to 32x32 and all second moments are not in the run files. Measuring them reads every
# configuration, about 1 GB per ensemble, so the per-configuration site averages are stored here once.
DATA = Path(f"{OUT}_squares.h5")
sys.path.insert(0, str(ROOT.parent / "plots"))
sys.path.insert(0, str(Path(__file__).parent))
import modelb_palette  # noqa: E402,F401  registers the modelb_field colormap the style sheet names
from refine_su2 import qmul, conj, roll  # noqa: E402
plt.style.use(ROOT.parent / "plots/cjp_prl_nature.mplstyle")
# Haar vermillion as in the L=32 figures and heatbath black as the reference. The coarse starts are
# ordered by Sigma^2 and take viridis colors from the largest to the smallest noise.
vermillion = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
black = "k"
R = np.array([1, 2, 4, 8, 16, 32])
# Every square up to L/4 samples the area axis between the stored squares.
RR = np.arange(1, 33)
# Fine starts were not run at L=128.
coarse = [("Sigma2_1p5_n512_steps1281", "s"), ("Sigma2_0p8_n512_steps1141", "^"), ("Sigma2_0p2_n512_steps0834", "v"),
          ("Sigma2_0p15_n512_steps0771", "<"), ("Sigma2_0p1_n512_steps0684", ">"),
          ("Sigma2_0p05_n512_steps0538", "p"), ("Sigma2_0p02_n512_steps0361", "h")]
viridis = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(coarse)))
cases = [("haar", vermillion, "o")] + [(name, tuple(c), m) for (name, m), c in zip(coarse, viridis)]


def links(m):
    # from_matrix without the renormalisation. Generated links are float32 accurate, |q|-1 ~ 1.5e-7,
    # and the stored loops were measured on them as saved.
    return np.stack((m[..., 0, 0].real, m[..., 0, 1].imag, m[..., 0, 1].real, m[..., 0, 0].imag), -1).transpose(0, 3, 2, 1, 4)


def squares(path, index, w):
    # Site averages of W and W^2 per configuration for every square R x R. The lines grow by one link
    # per R, and the site average of W must reproduce the stored loops.
    out = np.empty((2, len(index), len(RR)))
    with h5py.File(path) as f:
        for b in range(0, len(index), 128):
            u = links(f["cfgs"][index[b:b+128]])
            a, c = u[:, 0], u[:, 1]
            for k, r in enumerate(RR):
                if r > 1:
                    a = qmul(a, roll(u[:, 0], 0, r-1))
                    c = qmul(c, roll(u[:, 1], 1, r-1))
                W = qmul(qmul(qmul(a, roll(c, 0, r)), conj(roll(a, 1, r))), conj(c))[..., 0]
                out[:, b:b+128, k] = W.mean(axis=(1, 2)), (W**2).mean(axis=(1, 2))
    np.testing.assert_allclose(out[0][:, R-1], w, rtol=0, atol=1e-12)
    return out


def mean_se(x, block):
    b = x.reshape(-1, block, x.shape[1]).mean(axis=1)
    return x.mean(axis=0), b.std(axis=0, ddof=1)/np.sqrt(len(b))


# full keeps the stored R x T loops, src the configurations behind each ensemble.
full, src = [], []
with h5py.File(DIR / f"reverse_L64_to_L128_beta200_{coarse[0][0]}_betanoise.h5") as f:
    beta, P_exact = f.attrs["beta"], f.attrs["plaq_exact"]
    rt = f["wilson_RT"][:].tolist()
    j = [rt.index([r, r]) for r in R]
    full.append((f["heatbath_wilson"][:], f.attrs["heatbath_error_block"]))
    src.append((f.attrs["heatbath_source"], f["heatbath_parent_id"][:]))
    full.append((f["haar_wilson"][:], f.attrs["haar_error_block"]))
    src.append((f.attrs["baseline"], np.arange(len(full[1][0]))))
ticks = ["Haar"]
for name, _ in coarse:
    path = DIR / f"reverse_L64_to_L128_beta200_{name}_betanoise.h5"
    with h5py.File(path) as f:
        full.append((f["wilson"][:], f.attrs["error_block"]))
        ticks.append(f"{f.attrs['Sigma2']:.2g}")
    src.append((path, np.arange(len(full[-1][0]))))
names = ["heatbath"] + [c[0] for c in cases]
# Ensembles already in DATA are read back, new ones are measured once and added.
with h5py.File(DATA, "a") as g:
    g.attrs["R"] = RR
    for n, (path, index), (w, _) in zip(names, src, full):
        if n not in g:
            g[n] = squares(path, index, w[:, j])
    sq = [g[n][:] for n in names]
data = [(w[:, j], s[1][:, R-1], block) for (w, block), s in zip(full, sq)]

# Rows are heatbath followed by the cases, columns are the loop sizes R x R.
W = np.stack([mean_se(w, block) for w, _, block in data], axis=1)
W2 = np.stack([mean_se(w2, block) for _, w2, block in data], axis=1)
# Wilson action per plaquette, S/V = beta(1 - Tr U_p/2).
S = beta*(1-W[0, :, 0]), beta*W[1, :, 0]
sigma_exact = -np.log(P_exact)
print("<W>", W[0], "\n+/-", W[1], "\n<W^2>", W2[0], "\n+/-", W2[1], "\nS/V", S, sep="\n")


def points(ax, m, e, ylabel):
    ax.axhspan(m[0]-e[0], m[0]+e[0], color=black, alpha=0.12, linewidth=0)
    ax.axhline(m[0], color=black, linestyle="--", label="heatbath")
    for i, (_, color, marker) in enumerate(cases):
        ax.errorbar(i, m[i+1], e[i+1], color=color, marker=marker, linestyle="none")
    ax.set_xticks(range(len(cases)), ticks)
    ax.set_xlim(-0.5, len(cases)-0.5)
    ax.xaxis.set_minor_locator(NullLocator())
    # Steps of 2.5 would give loops near <W^2> = 1/4 a sixth decimal.
    ax.yaxis.set_major_locator(MaxNLocator("auto", steps=[1, 2, 5, 10]))
    # Plain decimals keep the offset text from sitting above the frame.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylabel(ylabel)


def groups(ax):
    # The coarse starts carry only their Sigma^2 as tick labels and share one group label below.
    g = ax.secondary_xaxis("bottom")
    g.set_xticks([len(cases)/2], [r"Coarse start $\Sigma^2$"])
    g.tick_params(length=0, pad=15)
    g.xaxis.set_minor_locator(NullLocator())
    g.spines["bottom"].set_visible(False)


for suffix, M, power in (("wilson_loops", W, ""), ("second_moments", W2, "^2")):
    # Full width, with each panel's axes box 2.11 in tall as the style sheet asks.
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.84), sharex=True)
    for k, (ax, r) in enumerate(zip(axes.flat, R)):
        points(ax, M[0, :, k], M[1, :, k], rf"$\langle W_{{{r}\times{r}}}{power}\rangle$")
    for ax in axes[-1]:
        groups(ax)
    axes[0, 0].legend(loc="center left")
    fig.savefig(f"{OUT}_{suffix}.pdf")

fig, ax = plt.subplots()
points(ax, *S, r"$\langle S/V\rangle=\beta(1-\langle W_{1\times1}\rangle)$")
groups(ax)
ax.legend(loc="center left")
fig.savefig(f"{OUT}_action_density.pdf")

# In two dimensions <W> = P^{RT} depends on the area only. The log area axis spreads the small
# loops, and the linear <W> axis keeps the large loops that are not resolved above zero.
# The stored rectangles add areas between the squares at small RT.
rect = [k for k, (r, t) in enumerate(rt) if r != t]
area = np.concatenate([np.prod(rt, axis=1)[rect], RR**2])
Wa = np.stack([mean_se(np.concatenate([w[:, rect], s[0]], axis=1), block) for (w, block), s in zip(full, sq)], axis=1)
a = np.geomspace(0.8, 1.25*area.max(), 200)
labels = ["heatbath", "Haar"] + [rf"$\Sigma^2={t}$" for t in ticks[1:]]
colors, markers = [black] + [c[1] for c in cases], ["D"] + [c[2] for c in cases]


def legends(ax, exact, handles, reference_loc, coarse_loc):
    # Exact, heatbath and Haar in one legend, the coarse starts ordered by Sigma^2 in a second one.
    ax.add_artist(ax.legend(handles=[exact, *handles[:2]], loc=reference_loc))
    ax.legend(handles=handles[2:], loc=coarse_loc, title="Coarse start")


fig, ax = plt.subplots()
# The style cycles markers too, so the exact curve needs marker="none".
exact, = ax.plot(a, P_exact**a, color="0.3", linestyle=":", marker="none", label="Exact")
handles = [ax.errorbar(area, Wa[0, i], Wa[1, i], color=color, marker=marker, markersize=3, linestyle="none", label=label)
           for i, (color, marker, label) in enumerate(zip(colors, markers, labels))]
ax.set_xscale("log")
ax.set_xlim(a[0], a[-1])
ax.set_xlabel(r"$RT$")
ax.set_ylabel(r"$\langle W_{R\times T}\rangle$")
# Above the large-area tail and below the small loops the panel is empty.
legends(ax, exact, handles, "upper right", "lower left")
fig.savefig(f"{OUT}_area_law.pdf")

# -ln<W>/(RT) on the same loops and area axis. Each curve stops at the first loop that is not three
# standard errors above zero. Larger loops that still pass are upward fluctuations and bias -ln<W> low.
order = np.argsort(area, kind="stable")
fig, ax = plt.subplots()
exact, = ax.plot(a, np.full_like(a, sigma_exact), color="0.3", linestyle=":", marker="none", label="Exact")
handles = []
for i, (color, marker, label) in enumerate(zip(colors, markers, labels)):
    m, e = Wa[0, i][order], Wa[1, i][order]
    ok = np.cumprod(m > 3*e).astype(bool)
    A = area[order][ok]
    handles.append(ax.errorbar(A, -np.log(m[ok])/A, e[ok]/(m[ok]*A), color=color, marker=marker, markersize=3,
                               linewidth=0.8, label=label))
ax.annotate(rf"$\sigma a^2={sigma_exact:.5g}$", (a[-1], sigma_exact), xytext=(-6, -4),
            textcoords="offset points", ha="right", va="top", color="0.3")
ax.set_xscale("log")
ax.set_xlim(a[0], a[-1])
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$RT$")
ax.set_ylabel(r"$-\ln\langle W_{R\times T}\rangle/(RT)$")
# The steep Haar, 1.5 and 0.8 curves end near RT = 100, so the right half above them is empty.
legends(ax, exact, handles, "center right", "upper left")
fig.savefig(f"{OUT}_string_tension.pdf")
