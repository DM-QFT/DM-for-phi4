#!/usr/bin/env python3
"""Summarize the marker-bounded DCGM power telemetry for one run."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else math.nan,
        "median": statistics.median(values) if values else math.nan,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else math.nan,
    }


def parse_window(path: Path) -> tuple[float, float]:
    text = path.read_text()
    start_match = re.search(r"POWER_WINDOW_START epoch=([0-9.eE+-]+)", text)
    end_match = re.search(r"POWER_WINDOW_END epoch=([0-9.eE+-]+)", text)
    if not start_match or not end_match:
        raise RuntimeError(f"measurement markers are missing in {path}")
    return float(start_match.group(1)), float(end_match.group(1))


def parse_dcgm(path: Path, start: float, end: float) -> list[dict[str, float]]:
    names = [
        "power_w", "total_energy_mj", "power_instant_w", "enforced_limit_w",
        "power_violation_ns", "sm_clock_mhz", "memory_clock_mhz", "gr_active",
        "sm_active", "sm_occupancy", "tensor_active", "dram_active",
        "fp32_active", "integer_active",
    ]
    rows: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        # timestamp, GPU, 0, then the twelve requested DCGM fields
        if len(fields) < 17 or fields[1:3] != ["GPU", "0"]:
            continue
        try:
            timestamp = float(fields[0])
            values = [float(value) for value in fields[3:17]]
        except ValueError:
            continue
        if start <= timestamp <= end:
            rows.append({"timestamp": timestamp, **dict(zip(names, values))})
    return rows


def parse_smi(path: Path, start: float, end: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 9:
                continue
            try:
                # nvidia-smi uses local time with millisecond precision.
                timestamp = datetime.strptime(
                    fields[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp()
                numeric = [float(value.strip()) for value in fields[1:7]]
            except ValueError:
                continue
            if start <= timestamp <= end:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "power_w": numeric[0],
                        "sm_clock_mhz": numeric[1],
                        "memory_clock_mhz": numeric[2],
                        "gpu_util_percent": numeric[3],
                        "memory_util_percent": numeric[4],
                        "temperature_c": numeric[5],
                        "pstate": fields[7].strip(),
                        "clock_event": fields[8].strip(),
                    }
                )
    return rows


def main() -> None:
    result_dir = Path(sys.argv[1]).resolve()
    start, end = parse_window(result_dir / "workload.log")
    dcgm = parse_dcgm(result_dir / "dcgm.log", start, end)
    smi = parse_smi(result_dir / "nvidia_smi.csv", start, end)
    if not dcgm:
        raise RuntimeError("no DCGM rows fell inside the marker window")

    fields = [
        "power_w", "power_instant_w", "enforced_limit_w", "sm_clock_mhz",
        "memory_clock_mhz", "gr_active", "sm_active", "sm_occupancy",
        "tensor_active", "dram_active", "fp32_active", "integer_active",
    ]
    summary: dict[str, object] = {
        "window_start_epoch": start,
        "window_end_epoch": end,
        "window_seconds": end - start,
        "dcgm_samples": len(dcgm),
        "nvidia_smi_samples": len(smi),
        "dcgm": {field: stats([row[field] for row in dcgm]) for field in fields},
        "nvidia_smi": {},
    }
    if smi:
        smi_fields = [
            "power_w", "sm_clock_mhz", "memory_clock_mhz", "gpu_util_percent",
            "memory_util_percent", "temperature_c",
        ]
        summary["nvidia_smi"] = {
            field: stats([float(row[field]) for row in smi]) for field in smi_fields
        }
        summary["clock_events"] = sorted(
            {str(row["clock_event"]) for row in smi if row["clock_event"] != "None"}
        )

    # On this GH200 the total-energy counter visibly includes module/Grace
    # overhead.  Integrate DCGM's rolling GPU-power field instead; it also
    # agrees with the independent nvidia-smi power.draw stream and avoids
    # aliasing the highly bursty instantaneous samples at 100 ms.
    mean_power = summary["dcgm"]["power_w"]["mean"]  # type: ignore[index]
    summary["estimated_energy_j"] = mean_power * (end - start)
    energy_values = [row["total_energy_mj"] for row in dcgm]
    summary["total_energy_counter_delta_j"] = (
        max(energy_values) - min(energy_values)
    ) / 1000.0
    counter_span = dcgm[-1]["timestamp"] - dcgm[0]["timestamp"]
    summary["total_energy_counter_span_seconds"] = counter_span
    summary["total_energy_counter_average_w"] = (
        summary["total_energy_counter_delta_j"] / counter_span
        if counter_span > 0.0 else math.nan
    )
    violation_values = [row["power_violation_ns"] for row in dcgm]
    summary["power_violation_delta_ns"] = max(violation_values) - min(violation_values)
    limit = summary["dcgm"]["enforced_limit_w"]["median"]  # type: ignore[index]
    instant_peak = summary["dcgm"]["power_instant_w"]["max"]  # type: ignore[index]
    rolling_peak = summary["dcgm"]["power_w"]["max"]  # type: ignore[index]
    summary["instant_peak_fraction_of_enforced_limit"] = instant_peak / limit
    summary["rolling_peak_fraction_of_enforced_limit"] = rolling_peak / limit

    output = result_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
