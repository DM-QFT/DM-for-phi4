"""Publication figures for the draft's scalar section, cjp style, from saved data."""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.style.use(HERE.parent / "cjp.mplstyles")
# node's texlive lacks type1cm.sty -> emulate the look with STIX (bundled)
mpl.rcParams.update({"text.usetex": False, "font.family": "STIXGeneral",
                     "mathtext.fontset": "stix"})

OUT = HERE.parents[1] / "paper" / "betaconfig" / "assets" / "figs" / "2dphi4"
OUT.mkdir(parents=True, exist_ok=True)
PREV = HERE.parent / "results" / "verification" / "figs"

COLS = ["navy", "crimson", "green", "chocolate", "indigo", "darkgoldenrod"]
MARKS = ["o", "^", "s", "p", "v", "x"]
LS = [8, 16, 32, 64, 128, 256]

npz = {L: np.load(HERE.parent / "results" / "beta_conf_traj_multiL" /
                  f"beta_conf_fwd_rev_L{L}_k0.28_l0.022_ep10000_steps2000_log.npz")
       for L in LS}


def save(fig, name):
    # fixed 6x4.5 canvas so paired panels scale identically in the draft
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(PREV / f"draft_{name}.png", dpi=150)
    plt.close(fig)
    print(f"saved {name}")


# --- (1a) beta_conf along forward/backward, all L ---
fig, ax = plt.subplots()
for L, c in zip(LS, COLS):
    d = npz[L]
    ax.plot(d["t"], d["hmc_mean"], "-", color=c, lw=1.2, label=rf"$L={L}$")
    ax.plot(d["t"], d["dm_mean"], "--", color=c, lw=1.2)
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}(t)$")
ax.set_ylim(-0.04, 1.09)
ax.plot([], [], "-", color="k", lw=1.2, label="Forward")
ax.plot([], [], "--", color="k", lw=1.2, label="Backward")
ax.legend(fontsize=13, ncol=2, loc="center left")
save(fig, "betaconf_traj_multiL")

# --- (1b) backward - forward difference ---
fig, ax = plt.subplots()
for L, c, mk in zip(LS, COLS, MARKS):
    d = npz[L]
    diff = d["dm_mean"] - d["hmc_mean"]
    err = np.hypot(d["dm_sem"], d["hmc_sem"])
    sl = slice(None, None, 12)
    ax.errorbar(d["sigma_t"][sl], diff[sl], yerr=err[sl], marker=mk, ms=3.5,
                ls="-", lw=0.7, color=c, elinewidth=0.6, capsize=1.5,
                label=rf"$L={L}$")
ax.axhline(0.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma_t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}^{\,\mathrm{bwd}}-\beta_{\mathrm{conf}}^{\,\mathrm{fwd}}$")
ax.set_ylim(-0.011, 0.011)
ax.legend(fontsize=13, ncol=2)
save(fig, "betaconf_traj_diff")

# --- (2a) volume scaling of the directional term ---
rows = np.loadtxt(HERE.parent / "results" / "verification" / "a2_terms.csv", delimiter=",")
fig, ax = plt.subplots()
for k, c, mk in zip(["0.26", "0.2705", "0.28"], COLS, MARKS):
    r = rows[(rows[:, 0] == float(k)) & (rows[:, 2] == 0.0)]
    ax.errorbar(r[:, 1], -r[:, 5], yerr=r[:, 6], marker=mk, ms=6, ls="",
                color=c, capsize=2, label=rf"$\kappa={k}$")
Lg = np.array([7, 310])
r28 = rows[(rows[:, 0] == 0.28) & (rows[:, 2] == 0.0)]
ax.plot(Lg, -r28[0, 5] * (8 / Lg) ** 2, "k:", lw=1, label=r"$\propto 1/L^2$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$-\langle t_2\rangle$")
ax.legend(fontsize=13)
save(fig, "betaconf_term_scaling")

# --- (2b) sensitivity to the sampler, L=64 and 256 ---
c2 = json.load(open(HERE.parent / "results" / "verification" / "c2_results.json"))
c2b = json.load(open(HERE.parent / "results" / "verification" / "c2_results_L256.json"))
b_hmc, e_hmc = 0.99986, 0.00024      # HMC reference, c2_run.log
fig, ax = plt.subplots()
cols4 = {(64, "log"): "navy", (64, "linear"): "crimson",
         (256, "log"): "green", (256, "linear"): "chocolate"}
for data, Lv, ls, mkk in [(c2, 64, "-", ("o", "s")), (c2b, 256, "--", ("^", "v"))]:
    for sched, mk in [("log", mkk[0]), ("linear", mkk[1])]:
        c = cols4[(Lv, sched)]
        r = [q for q in data if q["sched"] == sched and q["b"] > 0.5]
        ax.errorbar([q["steps"] for q in r], [q["b"] for q in r],
                    yerr=[q["be"] for q in r], marker=mk, ms=5, ls=ls, lw=0.8,
                    color=c, capsize=2, label=f"{sched.capitalize()}, $L={Lv}$")
ax.axhspan(b_hmc - e_hmc, b_hmc + e_hmc, color="gray", alpha=0.4, label="HMC")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"EM steps $N_t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}$")
ax.set_ylim(0.92, 1.19)
ax.legend(fontsize=13)
save(fig, "betaconf_sensitivity")

# --- (2b') free-field closed form (appendix) ---
b3 = np.load(HERE.parent / "results" / "verification" / "b3_data.npz")
fig, ax = plt.subplots()
for L, c, mk in zip([16, 64], ["navy", "crimson"], ["o", "s"]):
    ax.errorbar(np.sqrt(b3["taus"]), b3[f"mb_L{L}"], yerr=b3[f"sb_L{L}"],
                marker=mk, ms=5, ls="", color=c, capsize=2, label=rf"$L={L}$")
    ax.plot(np.sqrt(b3["taus"]), b3[f"pred_L{L}"], "-", color=c, lw=1)
ax.plot([], [], "-", color="k", lw=1,
        label=r"$\mathrm{Tr}M/(\mathrm{Tr}M+\sigma_t^2\,\mathrm{Tr}M^2)$")
sg0 = np.logspace(0.15, 1.45, 10)
anchor = b3["pred_L64"][-1] * b3["taus"][-1]
ax.plot(sg0, 4 * anchor / sg0 ** 2, "-.", color="green", lw=1,
        label=r"$\propto \sigma_t^{-2}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\sigma_t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}$")
ax.legend(fontsize=13)
save(fig, "betaconf_freefield")

# --- (2b'') interacting theory: quartic sigma_t^-4 tail (appendix) ---
b1 = np.load(HERE.parent / "results" / "verification" / "b1_data.npz")
Cn = float((npz[64]["norm_max"] - npz[64]["norm_min"]) / 2)
fig, ax = plt.subplots()
sig, m = b1["s_n_L64"] / Cn, b1["meas_L64"]
ax.plot(sig[::6], m[::6], marker="o", ms=4, ls="", color="navy",
        label=r"Forward, $L=64$")
G0 = 2.6892                       # <phi^2> of the kappa=0.28 target
sg = np.logspace(np.log10(0.4), np.log10(sig.max()), 60)
ax.plot(sg, 1 / (20 * 0.022 * (G0 + (Cn * sg) ** 2) ** 2), ":", color="k",
        lw=1.4, label=r"$1/(20\lambda\langle\phi^2\rangle^2)$")
sg2 = np.logspace(-0.7, 0.7, 10) / Cn
ax.plot(sg2, 0.25 / (Cn * sg2) ** 2, "-.", color="green", lw=1,
        label=r"$\propto \sigma_t^{-2}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\sigma_t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}$")
ax.legend(fontsize=12, loc="lower left")
save(fig, "betaconf_tail")

# --- (2c) Gaussian example, VP scheme: finite-N curves and 1/N coefficient ---
MU0, S0 = 1.5, 0.8


def beta_exact(N, dmu2, s_t2):
    """Series form of the draft's alpha-integral (see b2_gaussian_example.py)."""
    z = N * dmu2 / s_t2 / 2
    c = N / 2 - 1
    n = np.arange(int(z + 12 * np.sqrt(z + 1) + 20))
    logp = n * np.log(z + 1e-300) - z - np.cumsum(np.log(np.maximum(n, 1)))
    return (N - 2) * S0 ** 2 / s_t2 * 0.5 * np.sum(np.exp(logp) / (c + n))


def mus(t):
    return (MU0 * np.exp(-t / 2) - MU0) ** 2, S0 ** 2 * np.exp(-t) + 1 - np.exp(-t)


tt = np.linspace(0, 4, 200)
fig, ax = plt.subplots()
for N, c in zip([4, 16, 1024], COLS):
    ax.plot(tt, [beta_exact(N, *mus(t)) for t in tt], "-", color=c, lw=1.2,
            label=rf"$N={N}$")
ax.plot(tt, [S0 ** 2 / sum(mus(t)) for t in tt], "k--", lw=1,
        label=r"$N\to\infty$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}(t)$")
ax.legend(fontsize=13)
save(fig, "betaconf_gauss_vp")

tt2 = np.linspace(0.001, 4, 160)
fig, ax = plt.subplots()
for N, c in zip([4, 16, 64], COLS):
    y = []
    for t in tt2:
        d, s = mus(t)
        lam2 = (d / (s + d)) ** 2
        beta_LN = S0 ** 2 / (s + d) * (1 - 2 * lam2 / N)
        y.append(beta_exact(N, d, s) - beta_LN)
    ax.plot(tt2, y, "-", color=c, lw=1.2, label=rf"$N={N}$")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}-\beta_{\mathrm{large}\,N}$")
ax.legend(fontsize=13)
save(fig, "betaconf_gauss_1overN")

# --- (3) quasi-equilibrium family vs true marginals ---
# Boltzmann side from Wolff+FAHMC (cluster updates: reliable IR at criticality)
c3 = json.load(open(HERE.parent / "results" / "verification" / "c3_wolff.json"))
fig, ax = plt.subplots()
labels = {"phi2": r"$\langle\phi^2\rangle$",
          "binder": r"$\langle m^4\rangle/\langle m^2\rangle^2$",
          "chi_c": r"$\chi_c$", "Fk1": r"$G(k_{\min})$"}
for key, c, mk in zip(["phi2", "binder", "chi_c", "Fk1"], COLS, MARKS):
    st = [r["st"] for r in c3["A"]]
    rel, err = [], []
    for r in c3["A"]:
        a, ea = r["o_eq"][key]; b, eb = r["o_fwd"][key]
        rel.append(a / b - 1)
        err.append(np.hypot(ea / b, a * eb / b ** 2))
    ax.errorbar(st, rel, yerr=err, marker=mk, ms=6, ls="-", lw=0.8, color=c,
                capsize=2, label=labels[key])
ax.axhline(0.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma_t$")
ax.set_ylabel(r"$O_{e^{-\beta S}}/O_{p_t}-1$")
ax.legend(fontsize=13)
save(fig, "betaconf_quasieq")

# --- (3b) annealed evolution with the quasi-equilibrium drift ---
d64 = np.load(HERE.parent / "results" / "beta_conf_traj_multiL" /
              "beta_conf_fwd_rev_L64_k0.28_l0.022_ep10000_steps2000_log.npz")
an = np.load(HERE.parent / "results" / "verification" / "c3b_anneal.npz")
fig, ax = plt.subplots()
ax.plot(d64["sigma_t"], d64["hmc_mean"], "-", color="navy", lw=1.4,
        label="Forward (exact)")
ax.plot(d64["sigma_t"], d64["dm_mean"], "--", color="crimson", lw=1.4,
        label="Trained model")
ax.plot(an["sigma"], an["beta"], ":", color="green", lw=2,
        label=r"$-\beta_{\mathrm{conf}}(t)\,\nabla S$ drift")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlim(2.2e-3, 4)
ax.set_xlabel(r"$\sigma_t$")
ax.set_ylabel(r"$\beta_{\mathrm{conf}}$")
ax.legend(fontsize=13, loc="upper right")
save(fig, "betaconf_anneal")
