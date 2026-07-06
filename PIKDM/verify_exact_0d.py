"""
Step 1: validate the analytic ground truth + validation harness for the 0D
phi^4 PIK benchmark, BEFORE any neural network is involved.

Schedule: smoothstep alpha(t) = 3t^2 - 2t^3, alpha_dot = 6t(1-t)
  -> alpha_dot(0) = alpha_dot(1) = 0, so v = 0 at both endpoints
     (this is what the tex claims; the linear alpha=t contradicts it).

Checks:
  (a) flux normalization  : integral of p*(dS_dt - <dS_dt>) over phi == 0
  (b) formula consistency : numeric d_t p  vs  analytic -p*(dS_dt - <dS_dt>)
  (c) reverse transport   : ODE from base p_B -> moments match target p_tar
  (d) forward transport   : ODE from target p_tar -> moments match base N(0,1)
If all pass, v_exact and the ODE-integration harness are trustworthy.
"""

import numpy as np

L, N = 8.0, 2049
phi = np.linspace(-L, L, N)
dphi = phi[1] - phi[0]
w = np.full(N, dphi); w[0] *= 0.5; w[-1] *= 0.5


def alpha(t):  return 3.0 * t**2 - 2.0 * t**3
def adot(t):   return 6.0 * t * (1.0 - t)

def S(t):      a = alpha(t); return (-1.0 + 1.5 * a) * phi**2 + (1.0 - a) / 8.0 * phi**4
def dS_dt(t):  return adot(t) * (1.5 * phi**2 - 0.125 * phi**4)

def density(t):
    lp = -S(t); lp -= lp.max()
    p = np.exp(lp)
    return p / np.sum(p * w)

def cumtrapz0(y):
    """Cumulative trapezoid from the left edge, aligned to the grid (length N)."""
    out = np.empty(N)
    out[0] = 0.0
    np.cumsum(0.5 * (y[1:] + y[:-1]) * dphi, out=out[1:])
    return out

def v_exact(t):
    p = density(t)
    src = p * (dS_dt(t) - np.sum(p * w * dS_dt(t)))
    cum = cumtrapz0(src)                       # = p * v  (the probability flux)
    return cum / np.maximum(p, 1e-300), cum[-1]


def cdf(t):
    p = density(t)
    return cumtrapz0(p)

def sample_density(t, n, rng):
    """Inverse-CDF sampling from p_t on the grid."""
    C = cdf(t); C /= C[-1]
    u = rng.random(n)
    return np.interp(u, C, phi)


def integrate(x, t_start, t_end, M=2000):
    """RK4 transport of samples x along dx/dt = v_t, from t_start to t_end.
    Forward: integrate(x, 0.0, 1.0); reverse: integrate(x, 1.0, 0.0) (h<0)."""
    def vfun(t, q):
        ve, _ = v_exact(t)
        return np.clip(np.interp(q, phi, ve), -50.0, 50.0)
    h = (t_end - t_start) / M
    t = t_start
    for _ in range(M):
        k1 = vfun(t, x)
        k2 = vfun(t + 0.5 * h, x + 0.5 * h * k1)
        k3 = vfun(t + 0.5 * h, x + 0.5 * h * k2)
        k4 = vfun(t + h, x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return x


def moments(x):       return np.mean(x**2), np.mean(x**4)
def grid_moments(t):
    p = density(t); return np.sum(p * w * phi**2), np.sum(p * w * phi**4)


if __name__ == "__main__":
    print("=== (a) flux normalization  | int p*(dS_dt-<dS_dt>) dphi should be ~0 ===")
    amax = 0.0
    for t in [0.2, 0.5, 0.8]:
        _, tail = v_exact(t)
        amax = max(amax, abs(tail))
        print(f"  t={t}: cum[-1] = {tail:+.3e}")
    print(f"  max|cum[-1]| = {amax:.2e}  {'PASS' if amax < 1e-8 else 'FAIL'} (threshold 1e-8)\n")

    print("=== (b) formula consistency | numeric d_t p vs -p*(dS_dt-<dS_dt>) ===")
    dt = 1e-4; emax = 0.0
    for t in [0.2, 0.5, 0.8]:
        dtp_num = (density(t + dt) - density(t - dt)) / (2 * dt)
        p = density(t)
        dtp_ana = -p * (dS_dt(t) - np.sum(p * w * dS_dt(t)))
        m = p > 1e-3
        rel = np.max(np.abs(dtp_num - dtp_ana)[m]) / np.max(np.abs(dtp_ana)[m])
        emax = max(emax, rel)
        print(f"  t={t}: max rel err (bulk) = {rel:.2e}")
    print(f"  {'PASS' if emax < 1e-3 else 'FAIL'} (threshold 1e-3)\n")

    rng = np.random.default_rng(0)
    n = 200_000

    print("=== (c) reverse transport   | base -> target ===")
    x = rng.standard_normal(n)                 # p_B = N(0,1)
    x = integrate(x, 1.0, 0.0)
    m2, m4 = moments(x); g2, g4 = grid_moments(0.0)
    print(f"  reverse <phi^2>={m2:.4f} <phi^4>={m4:.4f}")
    print(f"  target  <phi^2>={g2:.4f} <phi^4>={g4:.4f}")
    ok_c = abs(m2 - g2) < 0.05 and abs(m4 - g4) < 0.5
    print(f"  {'PASS' if ok_c else 'FAIL'}\n")

    print("=== (d) forward transport   | target -> base N(0,1) ===")
    x = sample_density(0.0, n, rng)            # p_tar
    x = integrate(x, 0.0, 1.0)
    m2, m4 = moments(x)
    print(f"  forward <phi^2>={m2:.4f} (->1) <phi^4>={m4:.4f} (->3)")
    ok_d = abs(m2 - 1.0) < 0.05 and abs(m4 - 3.0) < 0.3
    print(f"  {'PASS' if ok_d else 'FAIL'}")
