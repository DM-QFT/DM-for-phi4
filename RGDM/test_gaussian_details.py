"""Null-baseline test: Haar-decompose kappa=0.28 L=128 configs, replace the
3 detail channels by iid Gaussians (matched per-channel std), reconstruct,
and compare observables against the original HMC ensemble.

Usage: python test_gaussian_details.py
"""
import numpy as np
import h5py

ROOT = "/work/gw97/w97000/project/DM/2Dphi4/trainingdata"
KAPPA = "0.28"
NBIN = 32
RS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

f = h5py.File(f"{ROOT}/cfgs_wolff_fahmc_k={KAPPA}_l=0.022_128^2.jld2", "r")
x = f["cfgs"][:].astype(np.float32)
f.close()
N = x.shape[0]

# ---- Haar transform (a = block average) ----
p, q = x[:, 0::2, 0::2], x[:, 1::2, 0::2]
r_, s = x[:, 0::2, 1::2], x[:, 1::2, 1::2]
a = (p + q + r_ + s) / 4
h = (p - q + r_ - s) / 4
v = (p + q - r_ - s) / 4
d = (p - q - r_ + s) / 4


def reconstruct(a, h, v, d):
    y = np.empty((a.shape[0], 128, 128), dtype=np.float32)
    y[:, 0::2, 0::2] = a + h + v + d
    y[:, 1::2, 0::2] = a - h + v - d
    y[:, 0::2, 1::2] = a + h - v - d
    y[:, 1::2, 1::2] = a - h - v + d
    return y


assert np.abs(reconstruct(a, h, v, d) - x).max() < 1e-5, "Haar roundtrip failed"

# ---- detail-channel diagnostics: what iid Gaussian ignores ----
print("=== detail channel stats (kappa=%s) ===" % KAPPA)
for name, c in [("h", h), ("v", v), ("d", d)]:
    ac1 = np.mean(c * np.roll(c, 1, axis=1)) / np.mean(c**2)
    print(f"  {name}: mean={c.mean():+.4f}  std={c.std():.4f}  "
          f"spatial autocorr r=1: {ac1:+.4f}")
print(f"  cross-channel corr(h,v)={np.mean(h*v)/(h.std()*v.std()):+.4f}  "
      f"(h,d)={np.mean(h*d)/(h.std()*d.std()):+.4f}  "
      f"(v,d)={np.mean(v*d)/(v.std()*d.std()):+.4f}")
print(f"  corr(|h|, |grad a|): ", end="")
ga = np.abs(a - np.roll(a, 1, axis=1))
hh = np.abs(h)
print(f"{np.mean((hh-hh.mean())*(ga-ga.mean()))/(hh.std()*ga.std()):+.4f}")

# ---- Gaussian-detail reconstruction ----
rng = np.random.default_rng(0)
xg = reconstruct(
    a,
    rng.normal(0, h.std(), a.shape).astype(np.float32),
    rng.normal(0, v.std(), a.shape).astype(np.float32),
    rng.normal(0, d.std(), a.shape).astype(np.float32),
)

# sanity: coarse content exactly preserved
assert np.abs(xg.mean(axis=(1, 2)) - x.mean(axis=(1, 2))).max() < 1e-4


# ---- observables with jackknife over NBIN config bins ----
def per_bin_obs(y):
    """Return per-bin sums of everything we need, shape (NBIN, nobs)."""
    obs = []
    for b in range(NBIN):
        z = y[b * (N // NBIN):(b + 1) * (N // NBIN)]
        m = [np.mean(z**k) for k in (2, 4, 6, 8)]
        grad = 0.5 * (np.mean((z - np.roll(z, 1, axis=1))**2)
                      + np.mean((z - np.roll(z, 1, axis=2))**2))
        G = [0.5 * (np.mean(z * np.roll(z, rr, axis=1))
                    + np.mean(z * np.roll(z, rr, axis=2))) for rr in RS]
        obs.append(m + [grad] + G)
    return np.array(obs)


def derived(row):
    m2, m4, m6, m8 = row[:4]
    k4 = m4 - 3 * m2**2
    k6 = m6 - 15 * m4 * m2 + 30 * m2**3
    k8 = m8 - 28 * m6 * m2 - 35 * m4**2 + 420 * m4 * m2**2 - 630 * m2**4
    return np.concatenate([[m2, k4 / m2**2, k6 / m2**3, k8 / m2**4],
                           row[4:]])


def jackknife(binned):
    full = derived(binned.mean(axis=0))
    loo = np.array([derived(np.delete(binned, i, axis=0).mean(axis=0))
                    for i in range(NBIN)])
    err = np.sqrt((NBIN - 1) / NBIN * ((loo - full)**2).sum(axis=0))
    return full, err


names = (["<phi^2>", "k4s", "k6s", "k8s", "<(grad phi)^2>"]
         + [f"G(r={rr})" for rr in RS])
ref, ref_e = jackknife(per_bin_obs(x))
gau, gau_e = jackknife(per_bin_obs(xg))

print("\n=== observables: HMC vs Gaussian-detail reconstruction ===")
print(f"{'obs':>16} {'HMC':>12} {'gauss-detail':>13} "
      f"{'rel.dev':>9} {'sig.':>7}")
lines = []
for i, nm in enumerate(names):
    dev = gau[i] - ref[i]
    sig = dev / np.sqrt(ref_e[i]**2 + gau_e[i]**2)
    rel = dev / ref[i]
    print(f"{nm:>16} {ref[i]:12.5f} {gau[i]:13.5f} {rel:+8.2%} {sig:+7.1f}")
    lines.append((nm, ref[i], ref_e[i], gau[i], gau_e[i], rel, sig))

with open("/work/gw97/w97000/project/DM/RGDM/gaussian_details_test_k0.28.dat",
          "w") as fo:
    fo.write("# obs  HMC  err  gauss  err  rel_dev  n_sigma\n")
    for nm, a_, b_, c_, d_, e_, f_ in lines:
        fo.write(f"{nm.replace(' ', ''):20s} {a_: .6e} {b_:.2e} "
                 f"{c_: .6e} {d_:.2e} {e_:+.4e} {f_:+.2f}\n")
print("\nsaved -> RGDM/gaussian_details_test_k0.28.dat")
