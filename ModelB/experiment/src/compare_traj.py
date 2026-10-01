"""DM rollout and the paper's Langevin model vs the measured remainder of every track, from the origin t = 10 min.
Usage: python src/compare_traj.py runs/traj_eq_dt10s"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
RUN = EXP_DIR / sys.argv[1]
ROLLOUT = max((RUN / "samples").glob("rollout_ep*_R*.h5"), key=lambda p: int(p.stem.split("_ep")[1].split("_")[0]))
OUT = EXP_DIR / "results" / RUN.name
TAU_HIST_S = (10, 60, 300, 1200, 3000)      # panels of pdf_dr, cut to the length of the run
# Milestones of the paper's Fig. 9, x = 0.5 and 3.0 um from the barrier with lambda = 4.135 um, mirrored because
# our reflected frame drives toward -x: A -> B along the drive passes the well bottom u = n.
U_A, U_B = 0.5 - 0.5 / 4.135, 0.5 - 3.0 / 4.135
TPT_EDGES = np.arange(0, 121, 10.0)  # s, the paper's bins
SOURCES = ("true", "gen", "lang")
STYLE = {"true": dict(color="k", marker="o", ls="none", ms=3, label="data"),
         "gen": dict(color="C3", label="DM kernel"),
         "lang": dict(color="C0", ls="--", label="Langevin (paper)")}
plt.style.use(EXP_DIR.parents[1] / "plots/cjp.mplstyles")


def fringe(r, x0, tilt, lam):
    return (r[..., 0] - x0) / lam - tilt * r[..., 1]


def kurt(v):
    v = v - v.mean()
    return (v ** 4).mean() / (v ** 2).mean() ** 2 - 3


def sem(a):
    return a.std(0, ddof=1) / np.sqrt(a.shape[0])


def first_hit(hit):
    """Index of the first True per row, -1 if none."""
    return np.where(hit.any(1), hit.argmax(1), -1)


def transition_paths(u):
    """Paper Fig. 7b and 9: time in steps from the last crossing of one milestone to the first crossing of the other,
    crossings interpolated linearly between samples. Returns (time, is A->B, passes the well bottom)."""
    lo, hi = sorted((U_A, U_B))
    times, ab, well = [], [], []
    for row in u:
        c = np.floor(row - lo) + np.floor(row - hi) + 2              # milestone levels below each sample
        js, ts = [], []
        for i in np.flatnonzero(c[1:] != c[:-1]):
            a, b = int(c[i]), int(c[i + 1])
            j = np.arange(a + 1, b + 1) if b > a else np.arange(a, b, -1)   # levels crossed, in path order
            lev = (j - 1) // 2 + np.where(j % 2 == 1, lo, hi)            # level j = c just above it; odd = lower level of cell (j-1)//2
            js.append(j)
            ts.append(i + (lev - row[i]) / (row[i + 1] - row[i]))
        if not js:
            continue
        j, t = np.concatenate(js), np.concatenate(ts)
        is_a = (j % 2 == 1) == (U_A == lo)
        m = np.flatnonzero(is_a[1:] != is_a[:-1]) + 1
        times.append(t[m] - t[m - 1])
        ab.append(is_a[m - 1])
        well.append(np.minimum(j[m], j[m - 1]) % 2 == 1)             # both levels in one cell: the path crossed u = n
    return np.concatenate(times), np.concatenate(ab), np.concatenate(well)


def save_tsv(name, header, cols):
    np.savetxt(OUT / name, np.column_stack(cols), delimiter="\t", header="\t".join(header), comments="")


def save_fig(fig, name):
    """The figure as drawn, plus a copy with every axis linear."""
    fig.savefig(OUT / f"{name}.pdf")
    for a in fig.axes:
        a.set_xscale("linear")
        a.set_yscale("linear")
    fig.savefig(OUT / f"{name}_lin.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(ROLLOUT, "r") as h:
        r_gen, r0 = h["r_gen"][:], h["r0"][:]
        origin, dt, group = int(h.attrs["origin_index"]), int(h.attrs["dt_frames"]), h.attrs["group"]
    K, R, N, _ = r_gen.shape
    with h5py.File(EXP_DIR / "runs" / f"langevin_{group}/samples/rollout.h5", "r") as h:
        stride = dt // int(h.attrs["dt_frames"])
        r_lang = h["r_gen"][stride - 1::stride][:K, :R]        # the reference has 32 replicas, keep the first R
        assert int(h.attrs["origin_index"]) == origin and r_lang.shape == r_gen.shape, r_lang.shape
    with h5py.File(EXP_DIR / "data/trajectories.h5", "r") as h:
        g = h[group]
        dt_s, window = float(h.attrs["dt_s"]), int(h.attrs["window_frames"])
        pos_true = g["pos"][:, origin:origin + dt * K + 1:dt]
        pos_win = g["pos"][:, :window]
        fi = g["file_index"][:]
        x0, tilt, lam = (g[k][:][fi] for k in ("x0", "tilt", "lam"))
    assert pos_true.shape == (N, K + 1, 2) and np.array_equal(pos_true[:, 0], r0)
    step_s = dt * dt_s
    tau = step_s * np.arange(1, K + 1)
    D1 = np.diff(pos_win, axis=1).var(axis=(0, 1), dtype=np.float64).mean() / (2 * dt_s)
    v_win = (pos_win[:, dt:, 0] - pos_win[:, :-dt, 0]).mean(dtype=np.float64) / step_s

    d_true = pos_true[:, 1:] - pos_true[:, :1]                                    # (N, K, 2)
    disp = {"true": d_true, "gen": r_gen - r0, "lang": r_lang - r0}               # others (K, R, N, 2)
    per_track = lambda a: a if a.ndim == 3 else a.mean(1).transpose(1, 0, 2)      # -> (N, K, 2)

    # 1. MSD and mean displacement from the common origin
    cols, header = [tau], ["tau_s"]
    for comp, cname in ((0, "x"), (1, "y")):
        for s in SOURCES:
            sq = per_track(disp[s] ** 2)[..., comp]
            cols += [sq.mean(0), sem(sq)]
            header += [f"msd_{cname}_{s}", f"err_{cname}_{s}"]
    for s in SOURCES:
        m = per_track(disp[s])[..., 0]
        cols += [m.mean(0), sem(m)]
        header += [f"mean_dx_{s}", f"err_dx_{s}"]
    save_tsv("msd.tsv", header, cols)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for comp, a in ((0, ax[0]), (1, ax[1])):
        for s in SOURCES:
            sq = per_track(disp[s] ** 2)[..., comp]
            a.errorbar(tau, sq.mean(0), sem(sq), **STYLE[s])
        a.plot(tau, 2 * D1 * tau, color="0.6", lw=0.8, label=r"$2D\tau$")
        a.set(xscale="log", yscale="log", xlabel=r"$\tau$ [s]", ylabel=rf"MSD$_{'xy'[comp]}$ [$\mu$m$^2$]")
    for s in SOURCES:
        m = per_track(disp[s])[..., 0]
        ax[2].errorbar(tau, m.mean(0), sem(m), **STYLE[s])
    ax[2].set(xlabel=r"$\tau$ [s]", ylabel=r"$\langle\Delta x\rangle$ [$\mu$m]")
    ax[0].legend()
    save_fig(fig, "msd")

    # 1b. mean absolute position over the whole record (data from t = 0, models from the origin)
    with h5py.File(EXP_DIR / "data/trajectories.h5", "r") as h:
        pos_all = h[group]["pos"][:, ::25].astype(np.float64)                 # 1 s resolution
    t_all = 25 * dt_s * np.arange(pos_all.shape[1]) / 60
    t_mod = (origin * dt_s + tau) / 60
    mean_all, err_all = pos_all.mean(0), sem(pos_all - pos_all[:, :1])   # band = uncertainty of the drift, not of the absolute mean
    mean_mod = {s: disp[s].mean((1, 2), dtype=np.float64) + r0.mean(0, dtype=np.float64) for s in ("gen", "lang")}
    save_tsv("mean_xy.tsv", ["t_min", "x_true", "err_x", "y_true", "err_y"],
             [t_all, mean_all[:, 0], err_all[:, 0], mean_all[:, 1], err_all[:, 1]])
    save_tsv("mean_xy_models.tsv", ["t_min", "x_gen", "y_gen", "x_lang", "y_lang"],
             [t_mod, mean_mod["gen"][:, 0], mean_mod["gen"][:, 1], mean_mod["lang"][:, 0], mean_mod["lang"][:, 1]])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for comp, a in enumerate(ax):
        a.fill_between(t_all, mean_all[:, comp] - err_all[:, comp], mean_all[:, comp] + err_all[:, comp], color="0.8", lw=0)
        a.plot(t_all, mean_all[:, comp], color="k", lw=0.8, label="data")
        for s in ("gen", "lang"):
            a.plot(t_mod, mean_mod[s][:, comp], **STYLE[s])
        a.axvline(origin * dt_s / 60, color="0.5", ls=":", lw=0.8)
        a.set(xlabel=r"$t$ [min]", ylabel=rf"$\langle {'xy'[comp]}\rangle$ [$\mu$m]")
    ax[0].legend()
    save_fig(fig, "mean_xy")

    # 2. displacement distributions at fixed tau
    taus = [round(t / step_s) for t in TAU_HIST_S if t <= K * step_s]
    fig, ax = plt.subplots(2, len(taus), figsize=(3.0 * len(taus), 5.2))
    moments, pdfs = [], {}
    for j, k in enumerate(taus):
        for comp in (0, 1):
            v = {s: (disp[s][:, k - 1, comp] if s == "true" else disp[s][k - 1, ..., comp]).ravel() for s in SOURCES}
            lo, hi = np.quantile(np.concatenate([v["gen"], v["lang"]]), [5e-4, 1 - 5e-4])
            edges = np.arange(np.floor(lo), np.ceil(hi) + 0.25, 0.25)
            for s in SOURCES:
                p = np.histogram(v[s], bins=edges, density=True)[0]
                pdfs[f"tau{k}_{'xy'[comp]}_{s}"] = p
                ax[comp, j].plot(0.5 * (edges[1:] + edges[:-1]), p, **STYLE[s])
                moments.append((k * step_s, "xy"[comp], s, v[s].mean(), v[s].std(), kurt(v[s])))
            pdfs[f"tau{k}_{'xy'[comp]}_edges"] = edges
            ax[comp, j].set(yscale="log", xlabel=rf"$\Delta {'xy'[comp]}$ [$\mu$m]",
                            title=rf"$\tau$ = {k * step_s:g} s" if comp == 0 else None)
    ax[0, 0].set(ylabel=r"$p(\Delta x)$")
    ax[1, 0].set(ylabel=r"$p(\Delta y)$")
    ax[0, 0].legend()
    save_fig(fig, "pdf_dr")
    np.savez(OUT / "pdf_dr.npz", **pdfs)
    with open(OUT / "moments.tsv", "w") as f:
        f.write("tau_s\tcomp\tsource\tmean\tstd\texcess_kurtosis\n")
        for row in moments:
            f.write("%g\t%s\t%s\t%.5f\t%.5f\t%.4f\n" % row)

    # 3. folded density p(u) and its depth along the rollout
    u = {"true": fringe(pos_true, x0[:, None], tilt[:, None], lam[:, None]),       # (N, K+1)
         "gen": fringe(r_gen, x0, tilt, lam), "lang": fringe(r_lang, x0, tilt, lam)}   # (K, R, N)
    u_win = fringe(pos_win[:, ::25], x0[:, None], tilt[:, None], lam[:, None])
    ub = np.linspace(0, 1, 41)
    pu = lambda q: np.histogram(q % 1, bins=ub, density=True)[0]
    depth = lambda p: np.log(p.max() / p.min())
    p_src = {"true": pu(u["true"][:, 1:]), "gen": pu(u["gen"]), "lang": pu(u["lang"])}
    p_win = pu(u_win)
    save_tsv("folded_u.tsv", ["u", "p_true", "p_gen", "p_lang", "p_window"],
             [0.5 * (ub[1:] + ub[:-1]), p_src["true"], p_src["gen"], p_src["lang"], p_win])
    block = round(100 / step_s)
    depth_k = {s: np.array([depth(pu(u[s][k])) for k in range(K)]) for s in ("gen", "lang")}
    t_block = step_s * block * (np.arange(K // block) + 0.5)
    depth_true = np.array([depth(pu(u["true"][:, 1 + b * block:1 + (b + 1) * block])) for b in range(K // block)])
    save_tsv("depth_vs_time.tsv", ["tau_s", "depth_gen", "depth_lang"], [tau, depth_k["gen"], depth_k["lang"]])
    save_tsv("depth_true_blocks.tsv", ["tau_s", "depth_true"], [t_block, depth_true])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    uc = 0.5 * (ub[1:] + ub[:-1])
    ax[0].plot(uc, p_win, color="0.6", label="data, training window")
    for s in SOURCES:
        ax[0].plot(uc, p_src[s], **STYLE[s])
    ax[0].set(xlabel=r"$u$ mod 1", ylabel=r"$p(u)$", yscale="log")
    ax[0].legend()
    ax[1].plot(t_block, depth_true, **STYLE["true"])
    for s in ("gen", "lang"):
        ax[1].plot(tau, depth_k[s], **STYLE[s])
    ax[1].set(xlabel=r"$\tau$ [s]", ylabel=r"$\max(-\ln p) - \min(-\ln p)$ [$k_BT$]")
    save_fig(fig, "folded_u")

    # 4. well-index hops per step and the survival in the initial well
    n0 = np.floor(u["true"][:, 0] + 0.5)
    n = {"true": np.floor(u["true"] + 0.5)}                                              # (N, K+1)
    for s in ("gen", "lang"):
        n[s] = np.concatenate([np.broadcast_to(n0, (1, R, N)), np.floor(u[s] + 0.5)])    # (K+1, R, N)
    dn = {"true": np.diff(n["true"], axis=1), "gen": np.diff(n["gen"], axis=0), "lang": np.diff(n["lang"], axis=0)}
    ms = np.arange(-3, 4)
    p_hop = {s: np.array([(dn[s] == m).mean() for m in ms]) for s in SOURCES}
    save_tsv("hops.tsv", ["dn"] + [f"P_{s}" for s in SOURCES], [ms] + [p_hop[s] for s in SOURCES])
    p_stay = {"true": (n["true"][:, 1:] == n0[:, None]).mean(0),
              "gen": (n["gen"][1:] == n0).mean((1, 2)), "lang": (n["lang"][1:] == n0).mean((1, 2))}
    save_tsv("pstay.tsv", ["tau_s"] + [f"P_stay_{s}" for s in SOURCES], [tau] + [p_stay[s] for s in SOURCES])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for i, s in enumerate(SOURCES):
        ax[0].bar(ms + 0.25 * (i - 1), p_hop[s], width=0.25, color=STYLE[s]["color"], label=STYLE[s]["label"])
        ax[1].plot(tau, p_stay[s], **STYLE[s])
    ax[0].set(yscale="log", xlabel=rf"$\Delta n$ per {step_s:g} s", ylabel=r"$P(\Delta n)$")
    ax[0].legend()
    ax[1].set(xlabel=r"$\tau$ [s]", ylabel=r"$P(n_\tau = n_0)$", xscale="log")
    save_fig(fig, "hops")

    # 5. one-step kernel conditioned on the origin: well bottom vs barrier, and the drift vs phase (paper Fig. 6)
    o = np.arange(0, window - dt, 25)
    # every rollout step is a (start phase, one-step displacement) sample, like the data remainder
    steps = lambda r: np.diff(np.concatenate([np.broadcast_to(r0, (1, R, N, 2)), r]), axis=0).reshape(-1, 2)
    starts = lambda q: np.concatenate([np.broadcast_to(u["true"][:, 0], (1, R, N)), q[:-1]]).ravel()
    one_step = {"gen": (steps(r_gen), starts(u["gen"])),
                "lang": (steps(r_lang), starts(u["lang"])),
                "remainder": ((pos_true[:, 1:] - pos_true[:, :-1]).reshape(-1, 2), u["true"][:, :-1].ravel()),
                "window": ((pos_win[:, o + dt] - pos_win[:, o]).reshape(-1, 2),
                           fringe(pos_win[:, o], x0[:, None], tilt[:, None], lam[:, None]).ravel())}
    origins = {"bottom": lambda f: np.abs(f) < 0.05, "barrier": lambda f: np.abs(f) > 0.45}
    line = {"gen": STYLE["gen"], "lang": STYLE["lang"], "remainder": dict(color="k", label="data, remainder"),
            "window": dict(color="0.6", label="data, training window")}
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    edges = np.arange(-8, 8.05, 0.1)
    with open(OUT / "kernel_cond.tsv", "w") as f:
        f.write("origin\tsource\tcount\tmean_dx\tstd_dx\tkurt_dx\tstd_dy\tkurt_dy\n")
        for a, (oname, sel) in zip(ax, origins.items()):
            for s, (d1, uo) in one_step.items():
                m = sel((uo + 0.5) % 1 - 0.5)
                v = d1[m]
                f.write("%s\t%s\t%d\t%.5f\t%.5f\t%.4f\t%.5f\t%.4f\n" % (
                    oname, s, m.sum(), v[:, 0].mean(), v[:, 0].std(), kurt(v[:, 0]), v[:, 1].std(), kurt(v[:, 1])))
                a.plot(0.5 * (edges[1:] + edges[:-1]), np.histogram(v[:, 0], bins=edges, density=True)[0], **line[s])
            a.set(yscale="log", xlabel=rf"$\Delta x$ per {step_s:g} s [$\mu$m]", title=f"origin at {oname}")
    ax[0].set(ylabel=r"$p(\Delta x\,|\,u_0)$")
    ax[0].legend()
    save_fig(fig, "kernel_cond")
    fb = np.linspace(-0.5, 0.5, 21)
    fc = 0.5 * (fb[1:] + fb[:-1])
    cols, header = [fc], ["u0"]
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for s, (d1, uo) in one_step.items():
        f = (uo + 0.5) % 1 - 0.5
        idx = np.clip(np.digitize(f, fb) - 1, 0, 19)
        cnt = np.bincount(idx, minlength=20)
        mean_dx = np.bincount(idx, d1[:, 0], minlength=20) / cnt
        p_hop_u = np.bincount(idx, np.abs(d1[:, 0]) > lam.mean() / 2, minlength=20) / cnt
        cols += [mean_dx, p_hop_u]
        header += [f"mean_dx_{s}", f"P_hop_{s}"]
        ax[0].plot(fc, mean_dx, **line[s])
        ax[1].plot(fc, p_hop_u, **line[s])
    save_tsv("drift_u.tsv", header, cols)
    ax[0].set(xlabel=r"$u_0$ (0 = well bottom)", ylabel=rf"$\langle\Delta x\,|\,u_0\rangle$ per {step_s:g} s [$\mu$m]")
    ax[1].set(xlabel=r"$u_0$ (0 = well bottom)", ylabel=rf"$P(|\Delta x| > \lambda/2\,|\,u_0)$", yscale="log")
    ax[0].legend()
    save_fig(fig, "drift_u")

    # 6. path-wise observables at the kernel resolution (paper Secs. 5 and 6)
    seq = lambda a, first: np.concatenate([np.broadcast_to(first, (1,) + a.shape[1:]), a]).reshape(K + 1, -1).T
    x_seq = {"true": pos_true[..., 0], "gen": seq(r_gen[..., 0], r0[:, 0]), "lang": seq(r_lang[..., 0], r0[:, 0])}
    u_seq = {"true": u["true"], "gen": seq(u["gen"], u["true"][:, 0]), "lang": seq(u["lang"], u["true"][:, 0])}
    lam_seq = {"true": lam[:, None], "gen": np.tile(lam, R)[:, None], "lang": np.tile(lam, R)[:, None]}
    summary, fpt, tpt = {}, {}, {}
    fe = step_s * np.arange(0.5, K + 1)
    cond_cols, cond_header = [tau], ["t_s"]
    for s in SOURCES:
        x = x_seq[s]
        hit = x <= x[:, :1] - lam_seq[s] if group == "driven" else np.abs(x - x[:, :1]) >= lam_seq[s]
        k = first_hit(hit)
        fpt[s] = k[k > 0] * step_s
        summary[f"fpt_reached_{s}"] = (k > 0).mean()
        summary[f"fpt_mean_s_{s}"] = fpt[s].mean()
        summary[f"fpt_sem_s_{s}"] = fpt[s].std(ddof=1) / np.sqrt(len(fpt[s]))
        # paper Fig. 8: conditional mean <tau | tau < t> and the fraction of realised crossings vs the cut time t
        ks = np.sort(k[k > 0])
        cum = np.cumsum(ks)
        nj = np.searchsorted(ks, np.arange(1, K + 1), "right")
        cond_cols += [np.where(nj > 0, cum[np.maximum(nj - 1, 0)] / np.maximum(nj, 1), np.nan) * step_s, nj / len(k)]
        cond_header += [f"cond_mean_s_{s}", f"fraction_{s}"]
        t_tpt, is_ab, via_well = transition_paths(u_seq[s])
        t_tpt *= step_s
        tpt[s] = {}
        for name, sel in (("AB", is_ab), ("BA", ~is_ab)):
            t = t_tpt[sel]
            summary[f"tpt_{name}_count_{s}"] = len(t)
            summary[f"tpt_{name}_mean_s_{s}"] = t.mean()
            summary[f"tpt_{name}_sem_s_{s}"] = t.std(ddof=1) / np.sqrt(len(t))
            summary[f"tpt_{name}_well_frac_{s}"] = via_well[sel].mean()
            for route, rsel in (("all", sel), ("well", sel & via_well), ("barrier", sel & ~via_well)):
                tpt[s][f"{name}_{route}"] = np.histogram(t_tpt[rsel], bins=TPT_EDGES, density=True)[0]
    save_tsv("fpt_vs_t.tsv", cond_header, cond_cols)
    save_tsv("fpt.tsv", ["t_s"] + [f"p_{s}" for s in SOURCES],
             [step_s * np.arange(1, K + 1)] + [np.histogram(fpt[s], bins=fe, density=True)[0] for s in SOURCES])
    save_tsv("tpt.tsv", ["t_s"] + [f"p_{key}_{s}" for s in SOURCES for key in tpt[s]],
             [0.5 * (TPT_EDGES[1:] + TPT_EDGES[:-1])] + [tpt[s][key] for s in SOURCES for key in tpt[s]])
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for s in SOURCES:
        ax[0].plot(step_s * np.arange(1, K + 1), np.histogram(fpt[s], bins=fe, density=True)[0], **STYLE[s])
        for i, name in enumerate(("AB", "BA")):
            ax[1 + i].plot(0.5 * (TPT_EDGES[1:] + TPT_EDGES[:-1]), tpt[s][f"{name}_all"], **STYLE[s])
    ax[0].set(xlabel=r"$\tau(L)$ [s]" if group == "driven" else r"two-sided exit time [s]", ylabel="density", yscale="log")
    ax[1].set(xlabel=r"$\mathcal{T}(A\to B)$ [s]", ylabel="density", yscale="log")
    ax[2].set(xlabel=r"$\mathcal{T}(B\to A)$ [s]", ylabel="density", yscale="log")
    ax[0].legend()
    save_fig(fig, "fpt")
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for i, s in enumerate(SOURCES):
        ax[0].plot(tau, cond_cols[1 + 2 * i], **STYLE[s])
        ax[1].plot(tau, cond_cols[2 + 2 * i], **STYLE[s])
    ax[0].axhline(lam.mean() / abs(v_win), color="0.6", lw=0.8, label=r"$L/\bar v$") if group == "driven" else None
    ax[0].set(xlabel=r"cut time $t$ [s]", ylabel=r"$\langle\tau\,|\,\tau<t\rangle$ [s]", xscale="log")
    ax[1].set(xlabel=r"cut time $t$ [s]", ylabel="fraction realised", xscale="log")
    ax[0].legend()
    save_fig(fig, "fpt_vs_t")

    # 7. long-time transport and the time-reversal asymmetry of the one-step kernel
    n10 = round(600 / step_s)
    for s in SOURCES:
        last = (d_true[:, -1] if s == "true" else disp[s][-1].reshape(-1, 2)).astype(np.float64)
        summary[f"v_mean_um_s_{s}"] = last[:, 0].mean() / tau[-1]
        summary[f"D_eff_x_{s}"] = last[:, 0].var() / (2 * tau[-1])
        summary[f"D_eff_y_{s}"] = last[:, 1].var() / (2 * tau[-1])
        summary[f"depth_u_{s}"] = depth(p_src[s])
        summary[f"net_hops_var_10min_{s}"] = (dn[s][:, :n10].sum(1) if s == "true" else dn[s][:n10].sum(0)).var()
    summary["D1_window"] = D1
    summary["v_window"] = v_win
    summary["depth_u_window"] = depth(p_win)
    with open(OUT / "summary.tsv", "w") as f:
        for key, val in summary.items():
            f.write(f"{key}\t{val:.6g}\n")
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    edges = np.arange(-6, 6.01, 0.5)
    centres = 0.5 * (edges[1:] + edges[:-1])
    for s, key in (("true", "remainder"), ("gen", "gen"), ("lang", "lang")):
        c = np.histogram(one_step[key][0][:, 0], bins=edges)[0].astype(float)
        cr = c[::-1]
        ok = (c > 20) & (cr > 20)
        ax.plot(centres[ok], np.log(c[ok] / cr[ok]), **STYLE[s])
    ax.plot(centres, v_win / D1 * centres, color="0.6", lw=0.8, label=r"$\bar v\,\Delta x/D$")
    ax.set(xlabel=rf"$\Delta x$ per {step_s:g} s [$\mu$m]", ylabel=r"$\ln[p(\Delta x)/p(-\Delta x)]$")
    ax.legend()
    save_fig(fig, "asym")
    print("\n".join(f"{k}\t{v:.6g}" for k, v in summary.items()))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
