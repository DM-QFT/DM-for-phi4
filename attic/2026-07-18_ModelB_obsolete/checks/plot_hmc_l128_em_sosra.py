#!/usr/bin/env python3
"""Plot L=128 EM/SOSRA work-precision fronts from the benchmark TSV files."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"EM": "#D55E00", "SOSRA": "#0072B2"}
MARKERS = {"EM": "o", "SOSRA": "s"}


def read_tsv(name):
    with (HERE / name).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def speed_map():
    out = {}
    for name in (
        "hmc_l128_em_sosra_N512_throughput.tsv",
        "hmc_l128_em_sosra_equal_N512_throughput.tsv",
    ):
        for r in read_tsv(name):
            out[(r["method"], round(float(r["dt"]), 7))] = float(r["wall_per_t"])
    return out


def weak_points(speed):
    out = {}
    for name in (
        "hmc_l128_em_sosra_N512_accuracy.tsv",
        "hmc_l128_em_sosra_equal_N512_accuracy.tsv",
    ):
        for r in read_tsv(name):
            if r["observable"] != "action":
                continue
            key = (r["method"], round(float(r["dt"]), 7))
            if key not in speed:
                continue
            err = 100.0 * abs(float(r["relative_delta"]))
            se = 100.0 * float(r["paired_se"]) / abs(float(r["initial"]))
            out[key] = (speed[key], err, se)
    return out


def strong_points(speed):
    grouped = defaultdict(list)
    for name in ("hmc_paired_L128_T1.tsv", "hmc_paired_L128_equal_cost.tsv"):
        for r in read_tsv(name):
            if r["method"] not in ("EM", "SOSRA"):
                continue
            key = (r["method"], round(float(r["dt"]), 7))
            grouped[key].append(float(r["relrmse"]))
    out = {}
    for key, vals in grouped.items():
        if key in speed:
            out[key] = (speed[key], 100.0 * sum(vals) / len(vals))
    return out


def draw_method(ax, points, method, ylabel, weak=False):
    data = [(dt, *values) for (m, dt), values in points.items() if m == method]
    data.sort(key=lambda item: item[1], reverse=True)
    x = [d[1] for d in data]
    y = [d[2] for d in data]
    ax.plot(
        x, y, marker=MARKERS[method], color=COLORS[method],
        linewidth=1.7, markersize=5.5, label=method,
    )
    if weak:
        yerr = [d[3] for d in data]
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=COLORS[method],
                    alpha=0.55, capsize=2)
    for d in data:
        ax.annotate(f"{d[0]:g}", (d[1], d[2]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=COLORS[method])
    ax.set_ylabel(ylabel)


def main():
    speed = speed_map()
    weak = weak_points(speed)
    strong = strong_points(speed)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for method in ("EM", "SOSRA"):
        draw_method(axes[0], weak, method,
                    r"$|\Delta\langle S/V\rangle|/\langle S/V\rangle$  [%]",
                    weak=True)
        draw_method(axes[1], strong, method,
                    "paired field RMSE  [%]")

    axes[0].axhspan(0.0, 0.06, color="0.7", alpha=0.22,
                    label="paired statistical resolution")
    axes[0].set_title("HMC stationarity, t=5")
    axes[1].set_title("Common-noise strong error")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("GH200 wall seconds / physical time")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("L=128 Model B: EM vs SOSRA (labels are dt)", fontsize=12)

    png = HERE / "fig_hmc_l128_em_sosra_pareto.png"
    pdf = HERE / "fig_hmc_l128_em_sosra_pareto.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
