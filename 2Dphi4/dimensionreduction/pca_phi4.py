# PCA of phi^4 configs across the phase transition (kappa_c ~ 0.2708), per kappa.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use(Path(__file__).parents[2] / "plots" / "cjp_prl_nature.mplstyle")

L = 32
kappas = [0.26, 0.2705, 0.28]
datadir = Path("/work/gw97/w97000/project/DM/2Dphi4/trainingdata/npz")
outdir = Path(__file__).parent

evr, proj, modes, M, Gsort = {}, {}, {}, {}, {}
for k in kappas:
    X = np.load(datadir / f"cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.npz")["cfgs"].reshape(-1, L * L)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    # fix SVD sign so PC1 points along +M
    flip = np.where(Vt.sum(1) < 0, -1.0, 1.0)
    evr[k] = S**2 / np.sum(S**2)
    proj[k] = U * S * flip
    modes[k] = Vt * flip[:, None]
    M[k] = X.mean(1)
    # momentum-space propagator <|phi(k)|^2>, FFT normalized so sum over k = total variance
    g = (np.abs(np.fft.fft2((X - X.mean(0)).reshape(-1, L, L)) / L) ** 2).mean(0).ravel()
    Gsort[k] = np.sort(g)[::-1] / g.sum()
    print(f"kappa={k}: EVR PC1 = {evr[k][0]:.4f}, corr(PC1, M) = {np.corrcoef(proj[k][:, 0], M[k])[0, 1]:.6f}")

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_box_aspect(0.75)

ax = axes[0, 0]
for i, k in enumerate(kappas):
    y = evr[k][evr[k] > 1e-12]  # drop numerical zeros beyond the rank when n < L^2
    ax.plot(np.arange(1, len(y) + 1), y, ls="none", label=f"$\\kappa={k}$")
    ax.plot(np.arange(1, len(Gsort[k]) + 1), Gsort[k], ls="-", lw=0.8, marker="none", color=f"C{i}")
ax.plot([], [], ls="-", lw=0.8, marker="none", color="k", label="$G(k)=\\langle|\\phi(k)|^2\\rangle$ sorted")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("PC index")
ax.set_ylabel("normalized eigenvalue")
ax.legend()
ax.text(0.05, 0.06, f"$L={L}$, $\\lambda=0.022$", transform=ax.transAxes)

ax = axes[0, 1]
for k in kappas:
    ax.plot(proj[k][:, 0], proj[k][:, 1], ls="none", alpha=0.15, label=f"$\\kappa={k}$", rasterized=True)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
leg = ax.legend(loc="lower left", borderaxespad=0.05)
for h in leg.legend_handles:
    h.set_alpha(1)

ax = axes[1, 0]
for k in kappas:
    ax.plot(M[k], proj[k][:, 0], ls="none", alpha=0.15, label=f"$\\kappa={k}$", rasterized=True)
Mall = np.concatenate([M[k] for k in kappas])
p1all = np.concatenate([proj[k][:, 0] for k in kappas])
a, b = np.polyfit(Mall, p1all, 1)
xf = np.array([Mall.min(), Mall.max()])
ax.plot(xf, a * xf + b, "k--", lw=1, marker="none")
ax.text(0.97, 0.05, f"fit: PC1 $= {a:.2f}\\,M {b:+.3f}$", transform=ax.transAxes, ha="right")
ax.set_xlabel("$M = L^{-2}{\\sum}_x\\, \\phi_x$")
ax.set_ylabel("PC1")
leg = ax.legend(loc="upper left")
for h in leg.legend_handles:
    h.set_alpha(1)

# the PC2..PC5 basis vectors themselves (unit coefficient), kappa=0.2705
recon = modes[0.2705][1:5].reshape(4, L, L)
vmax = np.abs(recon).max()
sub = axes[1, 1].get_subplotspec().subgridspec(2, 2)
fig.delaxes(axes[1, 1])
saxs = []
for j in range(4):
    sax = fig.add_subplot(sub[j // 2, j % 2])
    im = sax.imshow(recon[j], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    sax.set_xticks([])
    sax.set_yticks([])
    sax.text(0.05, 0.95, f"PC{j + 2}", transform=sax.transAxes, va="top")
    saxs.append(sax)
saxs[0].text(0.05, 0.76, "$\\kappa=0.2705$", transform=saxs[0].transAxes, va="top")
fig.colorbar(im, ax=saxs)

fig.savefig(outdir / f"pca_phi4_L{L}.png", dpi=300)
print(f"saved {outdir / f'pca_phi4_L{L}.png'}")

np.savez(outdir / f"pca_phi4_L{L}.npz",
         **{f"evr_k{k}": evr[k] for k in kappas},
         **{f"proj_k{k}": proj[k][:, :10] for k in kappas},
         **{f"modes_k{k}": modes[k][:10].reshape(10, L, L) for k in kappas},
         **{f"M_k{k}": M[k] for k in kappas})
