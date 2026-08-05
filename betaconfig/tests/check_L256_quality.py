"""Quality audit of the freshly generated L=256 ensembles.

Checks: MC-time drift & periodic oscillations (power spectrum), stored-config
independence, spatial G(k) smoothness + finite-size consistency vs L=128,
sector balance and basic sanity.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import FIGS, load_hmc, beta_conf_phys, blocked_mean_sem

KAPPAS = ["0.26", "0.2705", "0.28"]
L, L2 = 256, 0.022
DEV = "cuda"


def acf(x, maxlag=60):
    x = x - x.mean()
    v = (x * x).mean()
    return np.array([1.0] + [(x[:-t] * x[t:]).mean() / v for t in range(1, maxlag)])


fig, axes = plt.subplots(len(KAPPAS), 4, figsize=(19, 3.6 * len(KAPPAS)))
for row, k in enumerate(KAPPAS):
    p = torch.from_numpy(load_hmc(k, L, n=10240)).to(DEV)
    assert torch.isfinite(p).all()
    m = p.mean(dim=(1, 2)).cpu().numpy()
    phi2 = (p * p).mean(dim=(1, 2)).cpu().numpy()

    # (1) MC history: drift of phi^2 (first tenth vs rest), sign balance
    n10 = len(phi2) // 10
    d1, e1 = phi2[:n10].mean(), phi2[:n10].std() / np.sqrt(n10)
    d2, e2 = phi2[n10:].mean(), phi2[n10:].std() / np.sqrt(len(phi2) - n10)
    drift_z = (d1 - d2) / np.hypot(e1, e2)
    pos = (m > 0).mean()
    flips = (np.sign(m[1:]) != np.sign(m[:-1])).mean()

    # (2) periodicity: power spectrum of demeaned |m| and phi^2 series
    peaks = {}
    for name, x in [("m", m), ("phi2", phi2)]:
        P = np.abs(np.fft.rfft(x - x.mean())) ** 2
        P = P[1:]                                    # drop zero mode
        thr = P.mean() * (np.log(len(P)) + 5)        # white-noise extreme-value level
        idx = np.where(P > thr)[0]
        peaks[name] = [(int(i + 1), float(P[i] / P.mean())) for i in idx]

    # (3) consecutive-config overlap: r ~ 1 would mean frozen/duplicated configs
    dphi2 = ((p[1:] - p[:-1]) ** 2).mean(dim=(1, 2)).cpu().numpy()
    var_site = (p - p.mean()).pow(2).mean().item()
    r_consec = 1 - dphi2.mean() / (2 * var_site)

    # (4) beta_conf identity + tau_int of stored configs
    b = beta_conf_phys(p, float(k), L2)
    mb, eb = blocked_mean_sem(b.cpu().numpy())
    C = acf(np.abs(m))
    tau = 0.5 + C[1:][C[1:] > 0].sum()

    # (5) spatial: ensemble G(k) on axis, vs L=128 (even modes coincide)
    fk = torch.fft.rfft2(p)
    Gax = (fk[:, 0, :].abs() ** 2).mean(dim=0).cpu().numpy() / (L * L)
    p128 = torch.from_numpy(load_hmc(k, 128)).to(DEV)
    fk128 = torch.fft.rfft2(p128)
    Gax128 = (fk128[:, 0, :].abs() ** 2).mean(dim=0).cpu().numpy() / (128 * 128)
    del p128
    # spurious-mode scan: G(k) log-smoothness on the axis (2nd difference)
    lg = np.log(Gax[1:])
    rough = np.abs(np.diff(lg, 2)).max()

    print(f"k={k}: drift(phi2 first10% vs rest)={drift_z:+.1f}σ  sign(m>0)={pos:.3f} "
          f"flip-rate={flips:.3f}  r_consec={r_consec:+.4f}  "
          f"beta_conf={mb:.5f}({eb:.5f})  tau_int(|m|)={tau:.2f} cfgs  "
          f"maxG-roughness={rough:.3f}")
    for name, pk in peaks.items():
        print(f"   spectrum[{name}]: " + ("no peaks above noise level" if not pk
              else f"PEAKS at bins {pk} (freq units 1/5120 cfgs)"))

    ax = axes[row, 0]
    ax.plot(phi2, lw=0.4)
    ax.set_title(f"κ={k}: φ² history"); ax.set_xlabel("cfg index")
    ax = axes[row, 1]
    ax.plot(m, lw=0.4)
    ax.set_title("m history"); ax.set_xlabel("cfg index")
    ax = axes[row, 2]
    for name, x in [("m", m), ("phi2", phi2)]:
        P = np.abs(np.fft.rfft(x - x.mean())) ** 2
        ax.plot(np.arange(1, len(P)), P[1:] / P[1:].mean(), lw=0.4, label=name)
    ax.axhline(np.log(2560) + 5, color="r", ls=":", lw=1, label="peak threshold")
    ax.set_yscale("log"); ax.set_title("power spectrum (norm.)")
    ax.set_xlabel("frequency bin"); ax.legend(fontsize=7)
    ax = axes[row, 3]
    kax = np.arange(len(Gax))
    ax.plot(kax[1:], Gax[1:], lw=0.8, label="L=256")
    ax.plot(2 * np.arange(1, len(Gax128)), Gax128[1:], "--", lw=0.8, label="L=128 (k doubled)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("G(k) on axis"); ax.set_xlabel("mode n (L=256 units)")
    ax.legend(fontsize=7)

    del p
    torch.cuda.empty_cache()

fig.tight_layout()
fig.savefig(FIGS / "check_L256_quality.png", dpi=130)
print(f"saved -> {FIGS}/check_L256_quality.png")
