"""DM vs SOSRA 全对比：python run_all.py [32] [64]。缓存缺失时自动补提取。

只做分析并写 summary.md；图统一由 plots_prl.py 出。
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
from core import OUT, ROOT, T05, T10
from diagonal_modes import mode_layout

DS_LABEL = {"dm": "DM", "sosra": "SOSRA"}
TAIL_XMIN = 2.0          # Γ_eff 尾部拟合窗口（λ0_tail）
FIT_WINDOWS = {"all": (0.0, np.inf), "x>0.25": (0.25, np.inf), "x<6": (0.0, 6.0)}
SLOPE_KS = (3, 4, 5, 6)  # 短时斜率用前 K 个细网格点
BIN_TAIL_XMIN = 1.0      # 分箱后曲线更短，尾部窗口放宽到 x>=1


def fmt(v, e=None):
    if v is None or not np.isfinite(v):
        return "--"
    if e is None or not (np.isfinite(e) and e > 0):
        return f"{v:.3f}"
    p = int(np.floor(np.log10(e)))
    dec = max(0, 1 - p)
    scaled = int(round(e * 10 ** dec))
    if scaled >= 100:
        scaled = int(round(scaled / 10))
        dec -= 1
    return f"{v:.{max(dec, 0)}f}({scaled})"


def rho_stats(triplet, boot):
    rho = core.rho_of(triplet)
    rho_boot = np.stack([core.rho_of(triplet, idx) for idx in boot])
    return rho, rho_boot, core.boot_sem(rho_boot)


def merged_mode(stats05, stats10, mode):
    """两网格可信段拼接：返回 t、rho、boot rho、sem。"""
    rho05, rb05, sem05 = stats05
    rho10, rb10, sem10 = stats10
    m05 = core.leading_mask(T05, rho05[:, mode], sem05[:, mode])
    m10 = core.leading_mask(T10, rho10[:, mode], sem10[:, mode])
    if not (m05.any() or m10.any()):
        return None
    t, r = core.merge_masked(T05, rho05[:, mode], m05, T10, rho10[:, mode], m10)
    _, rb = core.merge_masked(T05, rb05[:, :, mode], m05, T10, rb10[:, :, mode], m10)
    return t, r, rb, core.boot_sem(rb)


def dataset_curves(stats05, stats10, khat):
    """每模合并曲线 + tau + z（含 boot 与模窗口稳定性）。"""
    nmode = len(khat)
    curves = {}
    taus = np.full(nmode, np.nan)
    tau_boot = np.full((stats05[1].shape[0], nmode), np.nan)
    for mode in range(nmode):
        merged = merged_mode(stats05, stats10, mode)
        if merged is None:
            continue
        t, r, rb, sem = merged
        curves[mode] = (t, r, rb, sem)
        taus[mode] = core.tau_1e_linear(t, r)
        for b in range(rb.shape[0]):
            tau_boot[b, mode] = core.tau_1e_linear(t, rb[b])
    z = core.fit_z(khat, taus)
    z_boot = np.array([core.fit_z(khat, tau_boot[b]) for b in range(tau_boot.shape[0])])
    subsets = {}
    for tag, sel in (("n1-3", [0, 1, 2]), ("n2-4", [1, 2, 3])):
        mask = np.full(nmode, np.nan)
        mask[sel] = taus[sel]
        subsets[tag] = core.fit_z(khat, mask)
    return curves, taus, tau_boot, (z, core.boot_sem(z_boot[np.isfinite(z_boot)])), subsets


def collapse_points(curves, taus, mode):
    t, r, rb, sem = curves[mode]
    x = t / taus[mode]
    return x, r, rb, sem


def fit_all_forms(x, rho, sem, boot_rho, r_sum, r_sum_boot):
    """自由/固定 b=1/2/双指数/和值规则约束四种形式 + 窗口稳定性 + boot 误差。

    bsum 的 b=(r_sum-λ0)/a；boot 时 r_sum 随重采样同步变动（Γ0、τ 同批算出）。
    """
    out = {"r_sum": r_sum}
    for form in ("bfree", "bhalf", "2exp"):
        params, chi2dof, aic = core.fit_collapse(x, rho, sem, form)
        out[form] = {"params": params, "chi2dof": chi2dof, "aic": aic}
    params, chi2dof, aic = core.fit_collapse_sum(x, rho, sem, r_sum)
    b_implied = (r_sum - params[0]) / params[1]
    out["bsum"] = {"params": np.array([params[0], params[1], b_implied]),
                   "chi2dof": chi2dof, "aic": aic}
    for wname, (lo, hi) in FIT_WINDOWS.items():
        if wname == "all":
            continue
        w = (x >= lo) & (x <= hi)
        for form in ("bfree", "bsum"):
            if form == "bfree":
                params, chi2dof, _ = core.fit_collapse(x[w], rho[w], sem[w], form)
            else:
                (lam0, a), chi2dof, _ = core.fit_collapse_sum(
                    x[w], rho[w], sem[w], r_sum)
                params = np.array([lam0, a, (r_sum - lam0) / a])
            out.setdefault("windows", {})[wname, form] = (params, chi2dof)
    for form in ("bfree", "bhalf", "bsum"):
        samples = []
        for b, rb in enumerate(boot_rho):
            if form == "bsum":
                (lam0, a), _, _ = core.fit_collapse_sum(x, rb, sem, r_sum_boot[b])
                params = np.array([lam0, a, (r_sum_boot[b] - lam0) / a])
            else:
                params, _, _ = core.fit_collapse(x, rb, sem, form)
            samples.append(params)
        out[form]["boot"] = core.boot_sem(np.array(samples))
    return out


def gn_equal_time(caches, ds, volume, boot):
    """等时 G_n：对该数据集全部时刻与轨道平均（逐轨道值供 bootstrap）。"""
    names = ("dm_t10", "dm_t0p5") if ds == "dm" else ("sosra",)
    per_traj = np.concatenate([
        (caches[n][1] * caches[n][1].conj()).real.astype(np.float64).mean(axis=3)
        / volume
        for n in names
    ]).mean(axis=0)
    g = per_traj.mean(axis=0)
    g_boot = np.stack([per_traj[idx].mean(axis=0) for idx in boot])
    return g, g_boot, per_traj


def short_time_slope(stats05, mode, k):
    """SOSRA 细网格 -ln rho 的 Δt->0 斜率：y=Γ·Δt+c·Δt² 加权拟合前 k 点。"""
    rho, _, sem = stats05
    t = T05[1:k + 1]
    y = -np.log(rho[1:k + 1, mode])
    w = (rho[1:k + 1, mode] / sem[1:k + 1, mode]) ** 2
    design = np.stack([t, t * t], axis=1)
    coeff = np.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * y))
    return coeff[0]


def coarse_segment(t):
    """合并曲线里粗网格点的掩码。"""
    # 细段止于最后一个保留的 0.5 网格点（可能正好是 10 的倍数，故再放宽 0.5），之后才是粗链
    return t > np.max(t[np.mod(t, 10) != 0], initial=-1.0) + 0.5


def gamma_from_curves(curves, taus):
    """各模 Γ_eff 点（x 归一用全样本 tau），返回拼接的 (u, g, err, mode_id)。"""
    us, gs, es, ids = [], [], [], []
    for mode in range(len(taus)):
        if mode not in curves or not np.isfinite(taus[mode]):
            continue
        t, r, rb, sem = curves[mode]
        xm, g, err = core.gamma_points(t / taus[mode], r, sem, seg=coarse_segment(t))
        us.append(1.0 / xm)
        gs.append(g)
        es.append(err)
        ids.append(np.full(len(g), mode))
    return [np.concatenate(v) for v in (us, gs, es, ids)]


def tail_fit(u, g, err, xmin=TAIL_XMIN, fixed_slope=0.5):
    m = (1.0 / u) >= xmin
    c, s = core.wline(u[m], g[m], err[m] ** -2.0, fixed_slope=fixed_slope)
    return c, s, int(m.sum())


# ---------------------------------------------------------------- analysis


def window_drift(modes, volume, boot):
    """时间原点窗口分半，rho 前后半差的最大显著度（lag<=总长 1/4）。"""
    half = modes.shape[0] // 2
    tri_a = core.lag_arrays(modes[:half + 1], volume)
    tri_b = core.lag_arrays(modes[half:], volume)
    nlag = half // 2
    rho_a, rho_b = core.rho_of(tri_a)[1:nlag], core.rho_of(tri_b)[1:nlag]
    diff_boot = np.stack([
        core.rho_of(tri_a, idx)[1:nlag] - core.rho_of(tri_b, idx)[1:nlag]
        for idx in boot
    ])
    sig = np.abs(rho_a - rho_b) / core.boot_sem(diff_boot)
    return np.nanmax(sig, axis=0)


def analyze(L):
    OUT.mkdir(parents=True, exist_ok=True)
    volume = float(L * L)
    nmode = core.NMODE[L]
    mode_n, _, _, _, khat, khat2 = mode_layout(L)
    khat, khat2, mode_n = khat[:nmode], khat2[:nmode], mode_n[:nmode]

    caches = {}
    for name in ("sosra", "dm_t10", "dm_t0p5"):
        if not (core.CACHE / f"L{L}_{name}.npz").exists():
            core.extract(L, name)
        times, modes, m, s = core.load_cache(L, name)
        caches[name] = (times, modes[:, :, :nmode], m, s)

    # DM 可以比 SOSRA 大（L32 扩充到 2048）；两者共享前 n_shared 个初态，
    # 成对比较只在共享前缀上做
    nens = {name: caches[name][1].shape[1] for name in caches}
    assert nens["dm_t10"] == nens["dm_t0p5"]
    n_shared = min(nens["sosra"], nens["dm_t10"])
    m0 = {"dm": caches["dm_t10"][2][0], "sosra": caches["sosra"][2][0]}
    assert np.allclose(m0["dm"], caches["dm_t0p5"][2][0], atol=1e-6)
    assert np.allclose(m0["dm"][:n_shared], m0["sosra"][:n_shared], atol=1e-6)
    boot = {"dm": core.boot_indices(nens["dm_t10"]),
            "sosra": core.boot_indices(nens["sosra"])}
    boot_shared = core.boot_indices(n_shared, seed=core.BOOT_SEED + 2)

    grids = {}
    t_s, modes_s = caches["sosra"][0], caches["sosra"][1]
    for gname, grid in (("t05", T05), ("t10", T10)):
        (sub,) = core.subgrid(t_s, [modes_s], grid)
        grids["sosra", gname] = core.lag_arrays(sub, volume)
        dm_modes = caches["dm_t0p5" if gname == "t05" else "dm_t10"][1]
        grids["dm", gname] = core.lag_arrays(dm_modes, volume)

    stats = {key: rho_stats(tri, boot[key[0]]) for key, tri in grids.items()}
    results = {"L": L, "khat": khat, "mode_n": mode_n}

    for ds in ("dm", "sosra"):
        curves, taus, tau_boot, zinfo, z_subsets = dataset_curves(
            stats[ds, "t05"], stats[ds, "t10"], khat)
        tau_err = core.boot_sem(tau_boot)
        print(f"L{L} {ds} (N={nens['dm_t10' if ds == 'dm' else 'sosra']}): "
              f"tau = {taus.round(2)},  z = {fmt(*zinfo)}, "
              f"subsets = { {k: round(v, 3) for k, v in z_subsets.items()} }")

        gn, gn_boot, _ = gn_equal_time(caches, ds, volume, boot[ds])
        gamma0 = khat2 / gn
        gamma0_boot = khat2[None, :] / gn_boot
        r_fdt = gamma0 * taus
        r_fdt_boot = gamma0_boot * tau_boot
        good = np.isfinite(r_fdt)
        weights = core.boot_sem(r_fdt_boot)[good] ** -2.0
        r_mean = np.average(r_fdt[good], weights=weights)
        r_mean_boot = np.array([
            np.average(r_fdt_boot[b][good], weights=weights)
            for b in range(len(boot[ds]))])

        fits = {}
        xs, rs, sems, rbs = [], [], [], []
        for mode in list(curves):
            if not np.isfinite(taus[mode]):
                continue
            x, r, rb, sem = collapse_points(curves, taus, mode)
            fits[mode] = fit_all_forms(x, r, sem, rb, r_fdt[mode],
                                       r_fdt_boot[:, mode])
            xs.append(x), rs.append(r), sems.append(sem), rbs.append(rb)
        x_joint = np.concatenate(xs)
        fits["joint"] = fit_all_forms(
            x_joint, np.concatenate(rs), np.concatenate(sems),
            np.concatenate(rbs, axis=1), r_mean, r_mean_boot)
        joint_sum = fits["joint"]["bsum"]
        print(f"L{L} {ds}: bsum joint lam0={joint_sum['params'][0]:.3f} "
              f"a={joint_sum['params'][1]:.2f} b={joint_sum['params'][2]:.3f} "
              f"chi2/dof={joint_sum['chi2dof']:.2f} (r={r_mean:.3f})")

        gamma = gamma_from_curves(curves, taus)
        gamma_boot = np.array([gamma_from_boot(curves, taus, b)[1]
                               for b in range(len(boot[ds]))])
        # 相邻 ρ 的误差强相关，对数差分的误差直接 bootstrap；独立传播在小 x 端高估 2–4 倍
        gamma[2] = core.boot_sem(gamma_boot)
        lam0_tail, _, ntail = tail_fit(*gamma[:3])
        lam0_windows = {xmin: tail_fit(*gamma[:3], xmin=xmin)[0]
                        for xmin in (1.5, 2.0, 3.0)}
        free_full = tail_fit(*gamma[:3], xmin=BIN_TAIL_XMIN, fixed_slope=None)
        lam0_boot, free_boot = [], []
        for gb in gamma_boot:
            lam0_boot.append(tail_fit(gamma[0], gb, gamma[2])[0])
            free_boot.append(tail_fit(gamma[0], gb, gamma[2], xmin=BIN_TAIL_XMIN,
                                      fixed_slope=None)[:2])
        lam0_tail_err = core.boot_sem(np.array(lam0_boot))
        free_err = core.boot_sem(np.array(free_boot))
        results[ds] = {
            "curves": curves, "taus": taus, "tau_err": tau_err, "z": zinfo,
            "z_subsets": z_subsets, "fits": fits, "gn": (gn, core.boot_sem(gn_boot)),
            "gamma0": (gamma0, core.boot_sem(gamma0_boot)),
            "r_fdt": (r_fdt, core.boot_sem(r_fdt_boot)),
            "r_mean": (r_mean, core.boot_sem(r_mean_boot)),
            "gamma": gamma, "lam0_tail": (lam0_tail, lam0_tail_err, ntail),
            "lam0_windows": lam0_windows,
            "tail_free": (free_full, free_err),
        }
        print(f"L{L} {ds}: lam0_tail(x>=2, s=1/2) = {fmt(lam0_tail, lam0_tail_err)}, "
              f"free fit x>=1: lam={fmt(free_full[0], free_err[0])}, "
              f"s={fmt(free_full[1], free_err[1])}")

    # FDT 锚点：SOSRA 短时斜率 vs Γ0；DM 一步 rho vs SOSRA 同 Δt
    slopes = {}
    for k in SLOPE_KS:
        slopes[k] = np.array([
            short_time_slope(stats["sosra", "t05"], mode, k) for mode in range(nmode)])
    slope_boot = np.stack([
        np.array([
            short_time_slope((core.rho_of(grids["sosra", "t05"], idx), None,
                              stats["sosra", "t05"][2]), mode, 4)
            for mode in range(nmode)])
        for idx in boot["sosra"]])
    results["slope"] = {"by_k": slopes, "err": core.boot_sem(slope_boot)}

    # 一步对比：中心值用各自全样本，差值及其误差在共享前 n_shared 条上成对算
    onestep = {}
    shared = np.arange(n_shared)
    for gname, dt in (("t05", 0.5), ("t10", 10.0)):
        row = {}
        for ds in ("dm", "sosra"):
            rho, _, sem = stats[ds, gname]
            row[ds] = (rho[1], sem[1])
        diff = (core.rho_of(grids["dm", gname], shared)[1]
                - core.rho_of(grids["sosra", gname], shared)[1])
        diff_boot = np.stack([
            core.rho_of(grids["dm", gname], idx)[1]
            - core.rho_of(grids["sosra", gname], idx)[1]
            for idx in boot_shared])
        onestep[dt] = (row["dm"][0], row["dm"][1], row["sosra"][0],
                       row["sosra"][1], diff, core.boot_sem(diff_boot))
    results["onestep"] = onestep

    # DM 体检（sosra 同批计算作对照；平稳性用 t10 网格的前/后三分之一）
    health = {}
    for tag in ("dm", "sosra"):
        if tag == "dm":
            times, modes, m, s = caches["dm_t10"]
        else:
            t_all, modes_all, m_all, s_all = caches["sosra"]
            (modes, m, s) = core.subgrid(t_all, [modes_all, m_all, s_all], T10)
            times = T10
        power = (modes * modes.conj()).real.astype(np.float64).mean(axis=3) / volume
        third = len(times) // 3
        n_tag = m.shape[1]
        gsig = np.empty(nmode)
        for mode in range(nmode):
            a, b = power[1:third, :, mode], power[-third:, :, mode]
            gsig[mode] = abs(a.mean() - b.mean()) / np.hypot(
                a.mean(axis=0).std(ddof=1) / np.sqrt(n_tag),
                b.mean(axis=0).std(ddof=1) / np.sqrt(n_tag))
        sa, sb = s[1:third].mean(axis=0), s[-third:].mean(axis=0)
        ssig = abs(sa.mean() - sb.mean()) / np.hypot(
            sa.std(ddof=1) / np.sqrt(n_tag), sb.std(ddof=1) / np.sqrt(n_tag))
        health[tag] = {
            "gsig": gsig, "ssig": ssig,
            "drift": np.abs(m - m[0]).max(),
            "var_ratio": m.var(axis=1).max() / m[0].var() - 1.0,
            "window": window_drift(modes, volume, boot[tag]),
            "s_traj": s[1:].mean(axis=0),
            "g_traj": power[1:].mean(axis=0),
        }
    # DM vs SOSRA 的静态量偏差：在共享前 n_shared 条同初态轨道上成对算
    for key, label in (("s_traj", "s_offset"), ("g_traj", "g_offset")):
        diff = health["dm"][key][:n_shared] - health["sosra"][key][:n_shared]
        diff_boot = np.stack([diff[idx].mean(axis=0) for idx in boot_shared])
        health[label] = np.max(np.abs(diff.mean(axis=0) / core.boot_sem(diff_boot)))

    # m 逐轨道守恒，rollout 的 m 分布就是初态值；KS 用 DM 的全部初态
    hmc_m = load_hmc_m(L)
    health["ks_p"] = ks_2samp(m0["dm"], hmc_m).pvalue if len(hmc_m) else np.nan
    results["health"] = health

    # m 分箱：各数据集用自己的 |m0| 四分位；箱内 Γ_eff 尾部直线拟合（斜率自由）
    rng = np.random.default_rng(core.BOOT_SEED + 1)
    bin_summary = {}
    for ds in ("dm", "sosra"):
        absm = np.abs(m0[ds])
        edges = np.quantile(absm, [0.25, 0.5, 0.75])
        bin_members = [np.flatnonzero(np.digitize(absm, edges) == b)
                       for b in range(4)]
        bin_boot = [members[rng.integers(0, len(members),
                                         (core.NBOOT, len(members)))]
                    for members in bin_members]
        taus = results[ds]["taus"]
        rows = []
        for b, members in enumerate(bin_members):
            stats05 = rho_stats_idx(grids[ds, "t05"], members, bin_boot[b])
            stats10 = rho_stats_idx(grids[ds, "t10"], members, bin_boot[b])
            curves_b = {}
            for mode in range(nmode):
                merged = merged_mode(stats05, stats10, mode)
                if merged is not None:
                    curves_b[mode] = merged
            gamma_b = gamma_from_curves(curves_b, taus)
            lam, slope, npts = tail_fit(*gamma_b[:3], xmin=BIN_TAIL_XMIN,
                                        fixed_slope=None)
            lam_boot, slope_boot_b = [], []
            for bb in range(core.NBOOT):
                gb = gamma_from_boot(curves_b, taus, bb)
                lam_b, slope_b, _ = tail_fit(*gb, xmin=BIN_TAIL_XMIN,
                                             fixed_slope=None)
                lam_boot.append(lam_b), slope_boot_b.append(slope_b)
            row = {
                "m2": float(np.mean(m0[ds][members] ** 2)),
                "absm": float(np.mean(np.abs(m0[ds][members]))),
                "lam": lam, "dlam": core.boot_sem(np.array(lam_boot)),
                "slope": slope, "dslope": core.boot_sem(np.array(slope_boot_b)),
                "npts": npts,
            }
            rows.append(row)
        bin_summary[ds] = rows
    results["bins"] = bin_summary
    return results


def rho_stats_idx(triplet, members, boot_members):
    rho = core.rho_of(triplet, members)
    rho_boot = np.stack([core.rho_of(triplet, idx) for idx in boot_members])
    return rho, rho_boot, core.boot_sem(rho_boot)


def gamma_from_boot(curves, taus, b):
    us, gs, es = [], [], []
    for mode in curves:
        if not np.isfinite(taus[mode]):
            continue
        t, r, rb, sem = curves[mode]
        xm, g, err = core.gamma_points(t / taus[mode], rb[b], sem, seg=coarse_segment(t))
        us.append(1.0 / xm), gs.append(g), es.append(err)
    return [np.concatenate(v) for v in (us, gs, es)]


def load_hmc_m(L):
    if L != 32:
        return np.empty(0)   # L64 无独立 HMC 系综，初态本身就是仅有的 512 个
    path = (ROOT / "trainingdata"
            / "modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2")
    with h5py.File(path, "r") as handle:
        key = [k for k in handle if "cfgs" in k.lower() or "phi" in k.lower()]
        data = np.asarray(handle[key[0]], dtype=np.float64)
    axes = tuple(i for i, n in enumerate(data.shape) if n == L)
    return data.mean(axis=axes)


def main_table(res):
    lines = ["主拟合为和值规则约束 bsum：b=(r−λ0)/a，λ0+a·b=r 严格成立；"
             "b_free 列为无约束拟合的 b，作交叉检查。", ""]
    columns = ("tau_n", "lambda0", "a", "b_sum", "chi2/dof", "b_free", "r_FDT")
    lines.append("| row | " + " | ".join(
        f"{c} DM | {c} SOSRA" for c in columns) + " |")
    lines.append("|" + "---|" * (2 * len(columns) + 1))

    def fit_cells(r, fit, mode):
        bsum, bfree = fit["bsum"], fit["bfree"]
        return [
            fmt(r["taus"][mode], r["tau_err"][mode]) if mode is not None
            else f"z={fmt(*r['z'])}",
            fmt(bsum["params"][0], bsum["boot"][0]),
            fmt(bsum["params"][1], bsum["boot"][1]),
            fmt(bsum["params"][2], bsum["boot"][2]),
            f"{bsum['chi2dof']:.2f}",
            fmt(bfree["params"][2], bfree["boot"][2]),
            fmt(r["r_fdt"][0][mode], r["r_fdt"][1][mode]) if mode is not None
            else fmt(*r["r_mean"]),
        ]

    ncol = len(columns)
    for mode in range(len(res["khat"])):
        pair = []
        for ds in ("dm", "sosra"):
            r = res[ds]
            pair.append(fit_cells(r, r["fits"][mode], mode)
                        if mode in r["fits"] else ["--"] * ncol)
        cells = [f"n={res['mode_n'][mode]} ({res['khat'][mode]:.3f})"]
        for column in range(ncol):
            cells += [pair[0][column], pair[1][column]]
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["joint"]
    for column in range(ncol):
        for ds in ("dm", "sosra"):
            r = res[ds]
            cells.append(fit_cells(r, r["fits"]["joint"], None)[column])
    lines.append("| " + " | ".join(cells) + " |")
    return lines


def stability_tables(res):
    lines = ["**z 的模窗口稳定性**（全样本值）", ""]
    lines.append("| dataset | n1-4 | n1-3 | n2-4 |")
    lines.append("|---|---|---|---|")
    for ds in ("dm", "sosra"):
        r = res[ds]
        lines.append(f"| {DS_LABEL[ds]} | {fmt(*r['z'])} | "
                     f"{r['z_subsets']['n1-3']:.3f} | {r['z_subsets']['n2-4']:.3f} |")
    lines += ["", "**joint 塌缩拟合的窗口稳定性**（bfree 与 bsum）", ""]
    lines.append("| dataset | form | window | lambda0 | a | b | chi2/dof |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in ("dm", "sosra"):
        fits = res[ds]["fits"]["joint"]
        for form in ("bfree", "bsum"):
            p = fits[form]["params"]
            lines.append(f"| {DS_LABEL[ds]} | {form} | all | {p[0]:.3f} | "
                         f"{p[1]:.2f} | {p[2]:.3f} | {fits[form]['chi2dof']:.2f} |")
            for (wname, wform), (params, chi2dof) in fits.get("windows", {}).items():
                if wform != form:
                    continue
                lines.append(f"| {DS_LABEL[ds]} | {form} | {wname} | {params[0]:.3f} | "
                             f"{params[1]:.2f} | {params[2]:.3f} | {chi2dof:.2f} |")
    lines += ["", "**Γ_eff 尾部拟合 λ0_tail**（slope 固定 1/2，窗口稳定性）", ""]
    lines.append("| dataset | x≥1.5 | x≥2 (err) | x≥3 |")
    lines.append("|---|---|---|---|")
    for ds in ("dm", "sosra"):
        r = res[ds]
        lam0, err, _ = r["lam0_tail"]
        lines.append(f"| {DS_LABEL[ds]} | {r['lam0_windows'][1.5]:.3f} | "
                     f"{fmt(lam0, err)} | {r['lam0_windows'][3.0]:.3f} |")
    lines += ["", "**joint 四种形式**（AIC 相对同数据最优；bsum 的 b 为派生量）", ""]
    lines.append("| dataset | form | params | chi2/dof | dAIC |")
    lines.append("|---|---|---|---|---|")
    for ds in ("dm", "sosra"):
        fits = res[ds]["fits"]["joint"]
        forms = ("bfree", "bhalf", "2exp", "bsum")
        best = min(fits[f]["aic"] for f in forms)
        for form in forms:
            p = fits[form]["params"]
            text = ", ".join(f"{v:.3g}" for v in p)
            lines.append(f"| {DS_LABEL[ds]} | {form} | {text} | "
                         f"{fits[form]['chi2dof']:.2f} | "
                         f"{fits[form]['aic'] - best:.1f} |")
    return lines


def anchor_tables(res):
    lines = ["**FDT 锚点（SOSRA）**：短时斜率 vs Γ0 = k̂²/G_n（M=T=1，源码约定）", ""]
    lines.append("| n | Γ0 | slope K=3 | K=4 (err) | K=5 | K=6 | slope/Γ0 (K=4) |")
    lines.append("|---|---|---|---|---|---|---|")
    slopes = res["slope"]
    g0, dg0 = res["sosra"]["gamma0"]
    for mode in range(len(res["khat"])):
        by_k = [slopes["by_k"][k][mode] for k in SLOPE_KS]
        ratio = by_k[1] / g0[mode]
        lines.append(
            f"| {res['mode_n'][mode]} | {fmt(g0[mode], dg0[mode])} | {by_k[0]:.3f} | "
            f"{fmt(by_k[1], slopes['err'][mode])} | {by_k[2]:.3f} | {by_k[3]:.3f} | "
            f"{ratio:.3f} |")
    lines += ["", "**DM 一步 ρ(Δt) vs SOSRA 同 Δt**（中心值各自全样本；"
              "diff/σ 在共享初态前缀上成对算）", ""]
    lines.append("| Δt | n | DM | SOSRA | diff/σ |")
    lines.append("|---|---|---|---|---|")
    for dt in (0.5, 10.0):
        dm, dm_sem, so, so_sem, diff, dsem = res["onestep"][dt]
        for mode in range(len(res["khat"])):
            sig = diff[mode] / dsem[mode]
            lines.append(f"| {dt:g} | {res['mode_n'][mode]} | "
                         f"{fmt(dm[mode], dm_sem[mode])} | "
                         f"{fmt(so[mode], so_sem[mode])} | {sig:+.1f} |")
    return lines


def bin_table(res):
    lines = ["**m 分箱**（|m(0)| 四分位；箱内 Γ_eff = λ + s·(1/x)，"
             f"窗口 x≥{BIN_TAIL_XMIN}；x 归一用全样本 τ_n；all 行为不分箱参照）", ""]
    lines.append("| bin | <|m|> | <m²> | λ DM | s DM | pts | λ SOSRA | s SOSRA | pts |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in range(4):
        dm, so = res["bins"]["dm"][b], res["bins"]["sosra"][b]
        lines.append(
            f"| Q{b + 1} | {dm['absm']:.3f} | {dm['m2']:.3f} | "
            f"{fmt(dm['lam'], dm['dlam'])} | {fmt(dm['slope'], dm['dslope'])} | "
            f"{dm['npts']} | {fmt(so['lam'], so['dlam'])} | "
            f"{fmt(so['slope'], so['dslope'])} | {so['npts']} |")
    cells = ["| all | -- | -- "]
    for ds in ("dm", "sosra"):
        (lam, slope, npts), err = res[ds]["tail_free"]
        cells.append(f"| {fmt(lam, err[0])} | {fmt(slope, err[1])} | {npts} ")
    lines.append("".join(cells) + "|")
    return lines


def health_list(res):
    health = res["health"]
    entries = []
    for tag in ("dm", "sosra"):
        h = health[tag]
        entries.append((f"{DS_LABEL[tag]} G_n 前/后三分一致性",
                        float(h["gsig"].max()), 3.0, "max|Δ|/σ"))
        entries.append((f"{DS_LABEL[tag]} 作用量密度平稳性",
                        float(h["ssig"]), 3.0, "|Δ|/σ"))
        entries.append((f"{DS_LABEL[tag]} ρ_n 原点窗口前/后半漂移",
                        float(np.nanmax(h["window"])), 4.0, "max|Δρ|/σ"))
        entries.append((f"{DS_LABEL[tag]} 守恒漂移 max|m(t)-m(0)|",
                        float(h["drift"]), 1e-3, "abs"))
        entries.append((f"{DS_LABEL[tag]} Var[m(t)] 相对变化",
                        float(abs(h["var_ratio"])), 0.05, "abs"))
    lines = ["**体检 PASS/FLAG**", ""]
    for name, value, threshold, unit in entries:
        status = "PASS" if value < threshold else "FLAG"
        lines.append(f"- {status}: {name} = {value:.3g} ({unit}, 阈 {threshold:g})")
    for key, name in (("s_offset", "DM−SOSRA 作用量密度成对偏差"),
                      ("g_offset", "DM−SOSRA 等时 G_n 成对偏差 (max over n)")):
        value = float(health[key])
        status = "PASS" if value < 3.0 else "FLAG"
        lines.append(f"- {status}: {name} = {value:.3g} (|Δ|/σ, 阈 3)")
    ks = health["ks_p"]
    if np.isfinite(ks):
        status = "PASS" if ks > 0.01 else "FLAG"
        lines.append(f"- {status}: 守恒 m（全部 DM 链）vs HMC 系综 P(m) 的 KS p = {ks:.3g}")
    else:
        lines.append("- SKIP: P(m) 对比（该 L 无独立 HMC 系综）")
    return lines


def write_summary(all_results, conclusions):
    lines = ["# DM vs SOSRA summary", ""]
    lines.append(
        "统计约定：误差为轨道级 bootstrap（B=200；L32 的 DM 为 512+1536 合并批 "
        "N=2048，SOSRA N=512；两者共享前 512 个初态，DM−SOSRA 的差值及其误差在"
        "该共享前缀上成对计算，中心值用各自全样本）。塌缩拟合 "
        "f(x)=exp(-λ0·x)(1+a·x)^(-b) 对合并可信段（5σ 且时间原点≥30）加权，"
        "主拟合施加和值规则约束 λ0+a·b=r（即 Γ_eff(0)=r），b=(r−λ0)/a 由约束派生，"
        "r=Γ0·τ_n 独立测定（joint 拟合用各模 r 的加权平均）；无约束 bfree 保留作"
        "交叉检查。χ² 用对角误差、忽略相邻 Δt 相关，偏小。τ_n 为 1/e 时间（对 ρ "
        "线性插值）。Γ0 = M·T·k̂²/G_n，M=1/γ=1、T=1、k̂²=8sin²(πn/L) 取自积分器"
        "源码（近邻 ∇² 与守恒噪声同一 stencil），G_n 为各自数据的等时估计值。"
        "短时斜率：-ln ρ 对前 K 个 δt=0.5 点拟 Γ·Δt+c·Δt²。")
    lines.append("")
    for res in all_results:
        lines.append(f"## L={res['L']}")
        lines.append("")
        lines += main_table(res)
        lines.append("")
        lines += stability_tables(res)
        lines.append("")
        lines += anchor_tables(res)
        lines.append("")
        lines += bin_table(res)
        lines.append("")
        lines += health_list(res)
        lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(conclusions)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


CONCLUSIONS = (
    "L=32（DM 为 512+1536 合并批，N=2048）：统计误差减半后 DM 与 SOSRA 的一致性"
    "依旧成立——z=3.792(19) vs 3.764(36)，约束拟合参数逐模 1–2σ 内重合，守恒 m "
    "对 HMC 大系综 KS p=0.51（顺带验证新 canonical 初态批同分布），仅存 FLAG 是 "
    "0.3–0.4% 量级的静态成对偏差（作用量密度 4.5σ、等时 G_n 3.4σ）与 L64 一步 ρ。"
    "和值规则 λ0+a·b=r（r=Γ0·τ_n 独立测定）与两套数据相容：joint 约束拟合 DM 给 "
    "λ0=0.327(15)、a=3.18(17)、b=0.473(28)，SOSRA 给 0.279(26)、2.73(29)、"
    "0.556(65)，b 被锚到 1/2 附近；但逐模 b_sum 从 0.15(n=1) 升到 0.49(n=4)、"
    "joint χ²/dof 随统计放大（DM 13.2），且尾部自由斜率 s=0.562(18) 与 1/2 有 3σ "
    "张力，说明单一 f(x) 的普适性在这些 k̂ 上只是近似。m 分箱（DM 每箱 512 条）"
    "箱内斜率不降（s_bin=0.53–0.64 ≈ 全样本）、λ(m) 仅缓升 0.233(18)→0.324(33)，"
    "进一步排除守恒 m 混合作为非指数尾主因。L64 把 z 窗口扩到 n=2–8（n=1 的 "
    "τ~3×10³ 不可测自动缺席，n≥5 由 δt=0.5 细网格测定）后 DM z=3.855(21) 与 "
    "SOSRA 3.867(23) 一致，旧 3 点窗口 n=2–4 的偏高（3.995(74) vs 3.788(76)）"
    "退化为低 k 子窗口效应，与 rollout 未平衡集中于长波模一致。健康 FLAG 不变"
    "（G_1 单调漂移、作用量密度下滑、一步 ρ(10) 在 n=2–5 偏快 4–8σ），L64 低 k "
    "端 τ 与静态量仍是未平衡数据。")


def main():
    targets = [int(v) for v in sys.argv[1:]] or [32]
    all_results = [analyze(L) for L in targets]
    write_summary(all_results, CONCLUSIONS)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
