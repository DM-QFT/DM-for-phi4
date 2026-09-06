"""Plot time-origin averaged correlations for the mixed-L extrapolations."""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagonal_modes import (
    load_fields as shared_load_fields,
    load_series,
    series_index,
    diagonal_fft,
    mode_layout,
    power,
    statistics,
)


plt.style.use(Path(__file__).resolve().parents[2] / "plots" / "cjp.mplstyles")

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/multiL"
OUTPUT = RUN / "analysis"
N = 512
LS = (32, 64, 128)
COLORS = {
    "modelb_t10": "black", "modelb_t0p5": "black",
    "t10": "#1f77b4", "t0p5": "#d62728",
}
LABELS = {
    "modelb_t10": "Model B", "modelb_t0p5": "Model B",
    "t10": r"mixed-L, $\delta t=10$",
    "t0p5": r"mixed-L, $\delta t=0.5$",
}


MODES_SHOWN = 4          # n>=5 的 rho 在 Δt~2 已落到噪声，画出来只是左边缘一撮点
MIN_ORIGINS = 30         # 时间原点数掉到这个以下，rho 的误差棒已不可信


def leading_run(ok):
    """从第一个点起连续显著的一段；避免远处噪声偶然过判据留下孤立点。"""
    keep = np.zeros_like(ok)
    for i, good in enumerate(ok):
        if not good:
            break
        keep[i] = True
    return keep


def initial_path(lattice):
    if lattice == 32:
        # 合并后的 2048 初态；本脚本仍只用前 N=512（与 SOSRA 参考同初态）
        return RUN / "hmc_L32_N2048_t0.npz"
    if lattice == 128:
        return RUN / "hmc_L128_N512_t0.npz"
    return (
        ROOT
        / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
        / "samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"
    )


def load_modes(series, times, initial, ky, kx, label):
    """t=0 不在序列里，用初态文件补上。"""
    series_times, fields = series
    result = []
    for i, t in enumerate(times):
        if i % 20 == 0:
            print(f"{label}: {i}/{len(times) - 1}", flush=True)
        block = (shared_load_fields(initial, count=N) if t == 0
                 else np.asarray(fields[series_index(series_times, t), :N],
                                 dtype=np.float32))
        result.append(diagonal_fft(block, ky, kx))
    return np.stack(result)


def lag_statistics(modes, volume):
    powers = power(modes, volume)
    nt, nensemble, nmodes = powers.shape
    rho = np.empty((nt, nmodes))
    sem = np.empty_like(rho)
    leaves = []
    for lag in range(nt):
        count = nt - lag
        left = modes[:count]
        right = modes[lag:]
        cross_values = np.mean(
            (left * np.conjugate(right)).real, axis=(0, 3)
        ) / volume
        values = statistics(
            cross_values,
            powers[:count].mean(axis=0),
            powers[lag:].mean(axis=0),
        )
        rho[lag], sem[lag] = values[2], values[3]
        leaves.append(values[4])
    return rho, sem, leaves


def analyze_lattice(lattice):
    mode_n, ky, kx, _, khat, _ = mode_layout(lattice)
    t10_times = np.arange(0, 1001, 10)
    t05_times = np.arange(0, 50.5, 0.5)
    modelb = load_series(ROOT / f"results/modelb_reference/multiL_L{lattice}.h5")
    series = {
        "t10": load_series(RUN / f"t10_L{lattice}.h5"),
        "t0p5": load_series(RUN / f"t0p5_L{lattice}.h5"),
        "modelb_t10": modelb,
        "modelb_t0p5": modelb,
    }
    times = {
        "t10": t10_times, "t0p5": t05_times,
        "modelb_t10": t10_times, "modelb_t0p5": t05_times,
    }
    result = {"mode_n": mode_n, "khat": khat, "times": times}
    for name in ("t10", "t0p5", "modelb_t10", "modelb_t0p5"):
        modes = load_modes(series[name], times[name], initial_path(lattice),
                           ky, kx, f"L{lattice} {name}")
        result[name] = lag_statistics(modes, lattice * lattice)
    return result


def plot_short(results):
    """单栏单联：对数纵轴把各模的指数衰减拉直，k 依赖一目了然。

    Model B 画线，DM 画点（实心 δt=10、空心 δt=0.5）。三者重叠即为符合；
    符合精度（绝对偏差 <=0.03，1--2 sigma）写正文，不占图面。
    """
    for lattice in LS:
        data = results[lattice]
        colors = plt.cm.viridis(np.linspace(0.08, 0.8, MODES_SHOWN))
        figure, axis = plt.subplots(figsize=(3.375, 2.6))
        series = (
            ("modelb_t0p5", dict(ls="-", lw=0.9, marker="")),
            ("t0p5", dict(ls="none", marker="o", ms=2.4, mfc="none", mew=0.6)),
            ("t10", dict(ls="none", marker="s", ms=2.9, mew=0.6)),
        )
        for mode in range(MODES_SHOWN):
            # 三个序列的时间网格不同，各自的可信上限也不同；取共同上限，
            # 否则稀疏序列会在参照线消失之后继续画出孤立点
            limits = []
            for name, _ in series:
                times = np.asarray(data["times"][name])
                rho, sem, _ = data[name]
                n_origins = len(times) - np.arange(len(times))
                ok = ((times > 0) & (times <= 50)
                      & (rho[:, mode] > 5 * sem[:, mode])
                      & (n_origins >= MIN_ORIGINS))
                run = leading_run(ok[times > 0])
                limits.append(times[times > 0][run][-1] if run.any() else 0.0)
            t_max = min(limits)
            for name, style in series:
                times = np.asarray(data["times"][name])
                rho, _, _ = data[name]
                keep = (times > 0) & (times <= t_max)
                axis.plot(times[keep], rho[keep, mode], color=colors[mode], **style)
        axis.set_yscale("log")
        axis.set(xlim=(0, 50), ylim=(2e-2, 1.6),
                 xlabel=r"$\Delta t$", ylabel=r"$\rho_n(\Delta t)$")
        mode_handles = [
            plt.Line2D([], [], color=colors[m], lw=1.4, marker="",
                       label=fr"${data['mode_n'][m]}$  ({data['khat'][m]:.2f})")
            for m in range(MODES_SHOWN)
        ]
        axis.legend(handles=mode_handles, title=r"$n$  ($\hat k$)", loc="lower right",
                    fontsize=6, title_fontsize=6, handlelength=1.1,
                    labelspacing=0.15, borderaxespad=0.4)
        handles = [
            plt.Line2D([], [], color="0.3", ls="-", lw=0.9, marker="", label="Model B"),
            plt.Line2D([], [], color="0.3", ls="none", marker="s", ms=2.9, label=r"DM $\delta t=10$"),
            plt.Line2D([], [], color="0.3", ls="none", marker="o", ms=2.4,
                       mfc="none", mew=0.6, label=r"DM $\delta t=0.5$"),
        ]
        figure.legend(handles=handles, loc="outside upper center", ncol=3,
                      title=fr"$L={lattice}$, $\kappa=0.34034$, $\lambda=1$")
        figure.savefig(OUTPUT / f"rho_L{lattice}_short.pdf")
        plt.close(figure)


def reliable_run(times, rho, sem, mode, t_hi):
    """从首点起连续满足 5 sigma 且时间原点足够的一段。"""
    n_origins = len(times) - np.arange(len(times))
    ok = ((times > 0) & (times <= t_hi) & (rho[:, mode] > 5 * sem[:, mode])
          & (n_origins >= MIN_ORIGINS))
    keep = np.zeros(len(times), dtype=bool)
    for i, good in enumerate(ok):
        if times[i] <= 0:
            continue
        if not good:
            break
        keep[i] = True
    return keep


def plot_loglog(results):
    """线性 Δt、对数 rho，量程到 1000：指数衰减在此是直线。

    delta t=0.5 与 delta t=10 两套网格拼起来覆盖 Δt=0.5..1000。高模的
    tau 只有 O(1)，在这个横轴上必然挤在左端；标度坍缩和 Gamma_eff
    分析在 dm_vs_sosra 管线里（N=2048、bootstrap 误差）。
    """
    for lattice in LS:
        data = results[lattice]
        colors = plt.cm.viridis(np.linspace(0.08, 0.8, MODES_SHOWN))
        figure, axis = plt.subplots(figsize=(3.375, 2.7))
        for mode in range(MODES_SHOWN):
            for name, style in (
                ("modelb_t0p5", dict(ls="-", lw=0.9, marker="")),
                ("modelb_t10", dict(ls="-", lw=0.9, marker="")),
                ("t0p5", dict(ls="none", marker="o", ms=2.4, mfc="none", mew=0.6)),
                ("t10", dict(ls="none", marker="s", ms=2.9, mew=0.6)),
            ):
                times = np.asarray(data["times"][name], dtype=float)
                rho, sem, _ = data[name]
                keep = reliable_run(times, rho, sem, mode, 1000.0)
                if keep.sum() < 3:      # 孤立一两点无从判读，不画
                    continue
                axis.plot(times[keep], rho[keep, mode], color=colors[mode], **style)
        axis.set(yscale="log", xlim=(0, 1000), ylim=(1e-2, 1.8),
                 xlabel=r"$\Delta t$", ylabel=r"$\rho_n(\Delta t)$")
        mode_handles = [
            plt.Line2D([], [], color=colors[m], lw=1.4, marker="",
                       label=fr"${data['mode_n'][m]}$  ({data['khat'][m]:.2f})")
            for m in range(MODES_SHOWN)
        ]
        axis.legend(handles=mode_handles, title=r"$n$  ($\hat k$)", loc="lower left",
                    fontsize=6, title_fontsize=6, handlelength=1.1,
                    labelspacing=0.15, borderaxespad=0.4)
        handles = [
            plt.Line2D([], [], color="0.3", ls="-", lw=0.9, marker="", label="Model B"),
            plt.Line2D([], [], color="0.3", ls="none", marker="s", ms=2.9, label=r"DM $\delta t=10$"),
            plt.Line2D([], [], color="0.3", ls="none", marker="o", ms=2.4,
                       mfc="none", mew=0.6, label=r"DM $\delta t=0.5$"),
        ]
        figure.legend(handles=handles, loc="outside upper center", ncol=3,
                      title=fr"$L={lattice}$, $\kappa=0.34034$, $\lambda=1$")
        figure.savefig(OUTPUT / f"rho_L{lattice}_t1000.pdf")
        plt.close(figure)


def write_tables(results):
    with (OUTPUT / "rho_time_origin_average.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("L", "chain", "lag", "mode_n", "khat", "rho", "rho_sem", "n_time_origins"))
        for lattice in LS:
            data = results[lattice]
            for name in ("modelb_t10", "t10", "modelb_t0p5", "t0p5"):
                times = data["times"][name]
                rho, sem, _ = data[name]
                for i, lag in enumerate(times):
                    for mode, n in enumerate(data["mode_n"]):
                        writer.writerow((lattice, name, lag, n, data["khat"][mode], rho[i, mode], sem[i, mode], len(times) - i))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {lattice: analyze_lattice(lattice) for lattice in LS}
    write_tables(results)
    plot_short(results)
    plot_loglog(results)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
