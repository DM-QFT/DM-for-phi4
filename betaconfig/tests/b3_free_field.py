"""B3: free-field generalisation of the Gaussian example.

For S = phi^T M phi/2 under VE noising (Cov = M^-1 + tau*1):
    beta_conf(tau) = TrM/(TrM + tau TrM^2) + O(1/V),
reducing to the draft's sigma0^2/(sigma0^2+tau) for M = 1/sigma0^2.
Lattice: omega_k = 2 - 4 kappa (cos k1 + cos k2), free below kappa=1/4.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import FIGS, beta_terms_phys

K = 0.20
DEV = "cuda"
NS = 4096
torch.manual_seed(2)

taus = np.logspace(-3, 3, 25)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
results = {}
for L in [16, 64]:
    k1 = 2 * np.pi * np.fft.fftfreq(L)
    om = 2 - 4 * K * (np.cos(k1)[:, None] + np.cos(k1)[None, :])
    assert om.min() > 0
    trM, trM2 = om.sum(), (om ** 2).sum()
    pred = trM / (trM + taus * trM2)

    # exact free-field sampling: filter white noise so Cov = M^-1
    om_t = torch.from_numpy(np.sqrt(om)).to(DEV)
    eta = torch.randn(NS, L, L, dtype=torch.float64, device=DEV)
    phi0 = torch.fft.ifft2(torch.fft.fft2(eta) / om_t).real

    # correctness gates: sampled action variance and beta_conf at tau=0
    S = 0.5 * ((torch.fft.fft2(phi0).abs() ** 2) * om_t ** 2).sum(dim=(1, 2)).cpu().numpy() / L ** 2
    assert abs(S.mean() / (L * L / 2) - 1) < 0.05          # <S> = V/2 equipartition
    t1, t2 = beta_terms_phys(phi0, K, 0.0)
    b0 = (t1 + t2).mean().item()
    sem0 = (t1 + t2).std().item() / np.sqrt(NS)
    assert abs(b0 - 1) < 5 * sem0, (b0, sem0)
    print(f"L={L}: tau=0 beta_conf = {b0:.5f} ± {sem0:.5f}  (exact 1)  ok")

    mb, sb, mt2 = [], [], []
    for tau in taus:
        p = phi0 + np.sqrt(tau) * torch.randn_like(phi0)
        t1, t2 = beta_terms_phys(p, K, 0.0)
        b = (t1 + t2).cpu().numpy()
        mb.append(b.mean()); sb.append(b.std(ddof=1) / np.sqrt(NS))
        mt2.append(t2.mean().item())
    results[L] = (np.array(mb), np.array(sb), np.array(mt2), pred)

ax = axes[0]
for L, c in zip([16, 64], ["C0", "C1"]):
    mb, sb, mt2, pred = results[L]
    ax.errorbar(taus, mb, yerr=sb, fmt="o", ms=3, color=c, label=f"measured L={L}")
    ax.plot(taus, pred, "-", color=c, lw=1)
k1 = 2 * np.pi * np.fft.fftfreq(64)
om64 = 2 - 4 * K * (np.cos(k1)[:, None] + np.cos(k1)[None, :])
G0_free = (1 / om64).mean()
ax.plot(taus, G0_free / (G0_free + taus), "k:", lw=1,
        label=r"iid $\sigma_0^2/(\sigma_0^2+\tau)$, $\sigma_0^2=\langle\phi^2\rangle$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"noise variance $\tau$"); ax.set_ylabel(r"$\beta_{\rm conf}$")
ax.set_title(r"free field: $\mathrm{Tr}M/(\mathrm{Tr}M+\tau\,\mathrm{Tr}M^2)$")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for L, c in zip([16, 64], ["C0", "C1"]):
    mb, sb, mt2, pred = results[L]
    ax.plot(taus, -np.array(mt2) * L * L, "o-", ms=3, lw=0.8, color=c,
            label=f"L={L}")
ax.set_xscale("log")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$-V\,\langle t_2\rangle$")
ax.set_title("directional term is O(1/V) along the whole flow")
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)

fig.suptitle(rf"B3: free lattice field, $\kappa={K}$, $\lambda=0$")
fig.tight_layout()
fig.savefig(FIGS / "b3_free_field.pdf")
fig.savefig(FIGS / "b3_free_field.png", dpi=160)

np.savez(FIGS.parent / "b3_data.npz", taus=taus,
         **{f"{k}_L{L}": v for L in [16, 64]
            for k, v in zip(("mb", "sb", "mt2", "pred"), results[L])})

mb, sb, _, pred = results[64]
dev = np.abs(mb / pred - 1)
print(f"L=64 max |measured/pred - 1| over tau grid: {dev.max():.4f}")
print(f"saved -> {FIGS}/b3_free_field.pdf")
