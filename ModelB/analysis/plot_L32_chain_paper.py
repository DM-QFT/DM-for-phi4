"""S5：rho_time_origin_average.tsv + zlog hmc_diagonal_lags.tsv → L32_chain_diagonal_lag_average.pdf。"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import ALT, DM, REFERENCE, ZERO  # noqa: E402

plt.style.use(PLOTS / "cjp.mplstyles")

TABLE = ROOT / "results/multiL/analysis/rho_time_origin_average.tsv"
SINGLE = {
    horizon: ROOT / (
        f"results/dm_vs_modelb/modelBcond_L32_k0.34034_l1.0_t{horizon:g}_ncsnpp_corrgauss_zlog_noskip/"
        "hmc_diagonal_lags.tsv"
    )
    for horizon in (10.0, 0.5)
}
OUT = ROOT / "results/multiL/analysis/L32_chain_diagonal_lag_average.pdf"

# multi-L 用 vermillion+o，single-L 用 blue+s
METHOD = {
    "modelb": (REFERENCE, "", "SOSRA"),
    "single": (DM, "s", r"single-$L$"),
    "mixed": (ALT, "o", r"multi-$L$"),
}
STEP = {10.0: "--", 0.5: "-"}
# 每点至少 30 个时间原点。下排 single-L 的链到 t=20，因此时间差只取到 5.5。
ROWS = (
    (10.0, "t10", "modelb_t10", 710.0, (9.0, 1100.0)),
    (0.5, "t0p5", "modelb_t0p5", 10.0, (0.45, 11.0)),
)

table = np.genfromtxt(TABLE, delimiter="\t", names=True, dtype=None, encoding=None)
single = {horizon: np.genfromtxt(path, delimiter="\t", names=True)
          for horizon, path in SINGLE.items()}


def select(rows, lag_max):
    rows = rows[np.argsort(rows["lag"])]
    return rows[(rows["lag"] > 0) & (rows["lag"] <= lag_max) & (rows["n_time_origins"] >= 30)]


def draw(axis, lag, rho, sem, color, marker, linestyle, nmark):
    # SOSRA 只画线；核的 marker 按对数间隔抽稀，single-L 用 15 个与 multi-L 错开
    axis.plot(lag, rho, color=color, linestyle=linestyle, marker="")
    if marker:
        idx = np.unique(np.geomspace(1, len(lag), min(nmark, len(lag))).astype(int) - 1)
        axis.errorbar(lag[idx], rho[idx], yerr=sem[idx], color=color, marker=marker, linestyle="none")


figure, axes = plt.subplots(2, 4, figsize=(7.0, 3.3))
for i, axis in enumerate(axes.flat):
    n = i + 1
    horizon, learned, reference, lag_max, xlim = ROWS[i // 4]
    # SOSRA 点线放在最底层，multi-L 放在最上层。核的线型区分 Δt。
    for model in ("modelb", "single", "mixed"):
        color, marker, _ = METHOD[model]
        linestyle = ":" if model == "modelb" else STEP[horizon]
        if model == "single":
            data = single[horizon]
            rows = select(data[data["mode_n"] == n], lag_max)
            draw(axis, rows["lag"], rows["rho_dm"], rows["rho_dm_sem"], color, marker, linestyle, 15)
            continue
        chain = reference if model == "modelb" else learned
        rows = select(table[(table["L"] == 32) & (table["chain"] == chain) & (table["mode_n"] == n)], lag_max)
        khat = rows["khat"][0]
        draw(axis, rows["lag"], rows["rho"], rows["rho_sem"], color, marker, linestyle, 20)
    axis.axhline(0.0, color=ZERO, linestyle=":", zorder=0)
    axis.set(xscale="log", xlim=xlim)
    lower, upper = axis.get_ylim()
    axis.set_ylim(lower, upper + 0.32 * (upper - lower))
    axis.text(0.03, 0.97, f"({chr(ord('a') + i)})", transform=axis.transAxes, ha="left", va="top")
    axis.text(0.97, 0.97, f"$n={n}$\n$\\hat k={khat:.3f}$", transform=axis.transAxes, ha="right", va="top")
    if i % 4 == 0:
        axis.set_ylabel(r"$\rho_n(\Delta t)$")
    axis.set_xlabel(r"$\Delta t$")

# 图例放在 (c)、(g) 的空白处。Δt 的线型由 caption 说明。
for axis, models, horizon in ((axes[0, 2], ("modelb", "single", "mixed"), 10.0),
                              (axes[1, 2], ("modelb", "single", "mixed"), 0.5)):
    handles = [Line2D([], [], color=METHOD[m][0], marker=METHOD[m][1],
                      linestyle=":" if m == "modelb" else STEP[horizon], label=METHOD[m][2]) for m in models]
    axis.legend(handles=handles)
figure.savefig(OUT)
print("saved", OUT)
