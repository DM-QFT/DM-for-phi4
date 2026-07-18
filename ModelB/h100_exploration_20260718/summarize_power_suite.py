#!/usr/bin/env python3
"""Build a compact cross-case CSV from all completed power runs."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


def extract_mode_and_timing(log_path: Path) -> tuple[str, float, int]:
    text = log_path.read_text()
    mode_match = re.search(r"POWER_SUMMARY mode=([^\s]+)", text)
    if mode_match:
        mode = mode_match.group(1)
    elif "GEMM_CONTROL" in text:
        mode = "gemm_control"
    else:
        mode = "unknown"

    solve_times = [
        float(value)
        for value in re.findall(r"POWER_SOLVE[^\n]* seconds=([0-9.eE+-]+)", text)
    ]
    if solve_times:
        return mode, statistics.median(solve_times), len(solve_times)

    fft_match = re.search(r"seconds_per_drift=([0-9.eE+-]+)", text)
    if fft_match:
        return mode, float(fft_match.group(1)), 0
    gemm_match = re.search(
        r"GEMM_CONTROL[^\n]*iterations=([0-9]+)[^\n]*elapsed=([0-9.eE+-]+)", text
    )
    if gemm_match:
        iterations = int(gemm_match.group(1))
        return mode, float(gemm_match.group(2)) / iterations, 0
    return mode, math.nan, 0


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "power_results")
    output = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "power_summary.csv")
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        data = json.loads(summary_path.read_text())
        mode, unit_seconds, solve_count = extract_mode_and_timing(summary_path.parent / "workload.log")
        dcgm = data["dcgm"]
        if solve_count:
            energy_per_unit = data["estimated_energy_j"] / solve_count
        elif math.isfinite(unit_seconds):
            energy_per_unit = dcgm["power_w"]["mean"] * unit_seconds
        else:
            energy_per_unit = math.nan
        rows.append(
            {
                "mode": mode,
                "run_directory": summary_path.parent.name,
                "window_seconds": data["window_seconds"],
                "dcgm_samples": data["dcgm_samples"],
                "solve_count": solve_count,
                "seconds_per_solve_or_drift": unit_seconds,
                "power_mean_w": dcgm["power_w"]["mean"],
                "power_median_w": dcgm["power_w"]["median"],
                "power_p95_w": dcgm["power_w"]["p95"],
                "power_peak_w": dcgm["power_w"]["max"],
                "instant_power_peak_w": dcgm["power_instant_w"]["max"],
                "total_energy_counter_average_w": data["total_energy_counter_average_w"],
                "estimated_energy_per_solve_or_drift_j": energy_per_unit,
                "sm_clock_mean_mhz": dcgm["sm_clock_mhz"]["mean"],
                "gr_active_percent": 100.0 * dcgm["gr_active"]["mean"],
                "sm_active_percent": 100.0 * dcgm["sm_active"]["mean"],
                "sm_occupancy_percent": 100.0 * dcgm["sm_occupancy"]["mean"],
                "dram_active_percent": 100.0 * dcgm["dram_active"]["mean"],
                "fp32_active_percent": 100.0 * dcgm["fp32_active"]["mean"],
                "integer_active_percent": 100.0 * dcgm["integer_active"]["mean"],
                "tensor_active_percent": 100.0 * dcgm["tensor_active"]["mean"],
                "rolling_peak_fraction_enforced_limit": data[
                    "rolling_peak_fraction_of_enforced_limit"
                ],
                "instant_peak_fraction_enforced_limit": data[
                    "instant_peak_fraction_of_enforced_limit"
                ],
                "power_violation_delta_ns": data["power_violation_delta_ns"],
            }
        )
    if not rows:
        raise RuntimeError(f"no summary.json files below {root}")

    mode_order = {
        "baseline_sync": 0,
        "rswm1_async": 1,
        "direct_fused_rswm1": 2,
        "shared_fused_rswm1": 3,
        "direct_packed_philox_rswm1": 4,
        "custom_fused_philox": 5,
        "fft_drift_only": 6,
        "gemm_control": 7,
    }
    rows.sort(key=lambda row: mode_order.get(str(row["mode"]), 99))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
