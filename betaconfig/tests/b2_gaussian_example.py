"""B2: Gaussian example numerics (draft's to-do, main.tex l.403).

VE: beta_conf(t)=sigma0^2/(sigma0^2+t) is exact at ANY N>2 (lambda_t=0 kills
all 1/N terms). VP with mu0!=0: verify the -2 lambda_t^2/N correction.
MC of <1/|x|^2> has infinite variance for N<=4, so N=3,4 use quadrature only.
"""
import numpy as np
import matplotlib.pyplot as plt

from common import FIGS

rng = np.random.default_rng(0)
MU0, S0 = 1.5, 0.8


def beta_hat_mean_mc(N, mu_t, s_t, nsamp=2_000_000):
    x = mu_t + s_t * rng.standard_normal((nsamp, N))
    b = (N - 2) * S0 ** 2 / ((x - MU0) ** 2).sum(axis=1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nsamp)


def beta_exact(N, dmu2, s_t2):
    """(N-2)(s0^2/s_t^2) I, I = 1/2 sum_n Pois(n; z)/(N/2-1+n) with z = N dmu2/(2 s_t2).

    Series form of the draft's alpha-integral; stable for all N>2 and all z.
    """
    z = N * dmu2 / s_t2 / 2
    c = N / 2 - 1
    nmax = int(z + 12 * np.sqrt(z + 1) + 20)
    n = np.arange(nmax)
    logp = n * np.log(z + 1e-300) - z - np.cumsum(np.log(np.maximum(n, 1)))
    I = 0.5 * np.sum(np.exp(logp) / (c + n))
    return (N - 2) * S0 ** 2 / s_t2 * I


# ---------- VE: exact at any N ----------
print("VE scheme: prediction s0^2/(s0^2+t), exact for all N>2")
ts = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
for N in [3, 4, 8, 64]:
    for t in ts:
        s_t2 = S0 ** 2 + t
        pred = S0 ** 2 / s_t2
        ex = beta_exact(N, 0.0, s_t2)
        line = f"  N={N:3d} t={t:6.1f}: quad={ex:.6f} pred={pred:.6f}"
        if N > 4:
            mc, err = beta_hat_mean_mc(N, MU0 * np.ones(N), np.sqrt(s_t2))
            z = (mc - pred) / err
            line += f"  MC={mc:.6f}({err:.6f}) z={z:+.1f}"
        assert abs(ex / pred - 1) < 1e-4, (N, t, ex, pred)
        print(line)

# ---------- VP: 1/N corrections at fixed t ----------
T = 1.0
mu_t = MU0 * np.exp(-T / 2)
s_t2 = S0 ** 2 * np.exp(-T) + (1 - np.exp(-T))
dmu2 = (mu_t - MU0) ** 2
lam = dmu2 / (s_t2 + dmu2)
beta_inf = S0 ** 2 / (s_t2 + dmu2)
print(f"\nVP scheme at t={T}: lambda_t={lam:.4f}, predicted N-coefficient "
      f"-2 lambda^2 = {-2 * lam ** 2:.4f}")

# quadrature validated against MC where the MC variance is finite
for N in [8, 64]:
    mc, err = beta_hat_mean_mc(N, mu_t * np.ones(N), np.sqrt(s_t2))
    ex = beta_exact(N, dmu2, s_t2)
    print(f"  N={N}: quad={ex:.6f}  MC={mc:.6f}({err:.6f})  z={(mc - ex) / err:+.1f}")

Ns = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096])
R = np.array([Ns[i] * (beta_exact(Ns[i], dmu2, s_t2) / beta_inf - 1)
              for i in range(len(Ns))])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
tt = np.linspace(0, 4, 200)
for N, c in zip([4, 16, 1024], ["C0", "C1", "C2"]):
    bb = [beta_exact(N, (MU0 * np.exp(-t / 2) - MU0) ** 2,
                     S0 ** 2 * np.exp(-t) + 1 - np.exp(-t)) for t in tt]
    ax.plot(tt, bb, color=c, label=f"exact N={N}")
binf = [S0 ** 2 / (S0 ** 2 * np.exp(-t) + 1 - np.exp(-t)
                   + (MU0 * np.exp(-t / 2) - MU0) ** 2) for t in tt]
ax.plot(tt, binf, "k--", lw=1, label=r"$N\to\infty$ leading")
ax.set_xlabel("t"); ax.set_ylabel(r"$\beta_{\rm conf}(t)$")
ax.set_title(f"VP flow, $\\mu_0={MU0}$, $\\sigma_0={S0}$")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(1 / Ns, R, "o-", label=r"$N(\beta_N/\beta_\infty-1)$")
ax.axhline(-2 * lam ** 2, ls=":", color="C3",
           label=rf"$-2\lambda_t^2 = {-2 * lam ** 2:.4f}$")
ax.set_xscale("log")
ax.set_xlabel("1/N"); ax.set_title(f"1/N coefficient at t={T}")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("B2: analytical Gaussian example — finite-N checks")
fig.tight_layout()
fig.savefig(FIGS / "b2_gaussian_example.pdf")
fig.savefig(FIGS / "b2_gaussian_example.png", dpi=160)
print(f"large-N limit of N(beta_N/beta_inf - 1): {R[-1]:.4f} "
      f"(predicted {-2 * lam ** 2:.4f})")
print(f"saved -> {FIGS}/b2_gaussian_example.pdf")
