#!/usr/bin/env python3
"""Compute scalar observables for 2D phi4 multi-L L=128 samples.

This script is intended to run on compute nodes, not on a login node.  It
streams configurations in chunks, converts each ensemble to the Z2-symmetric
statistics convention, computes DM and HMC observables, and compares the HMC
numbers with the reference values from the user's earlier table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import h5py
import numpy as np


def cumulants_from_raw_moments(moments: List[object]) -> List[object]:
    """Return cumulants from raw moments m1..mn.

    Works for scalars and numpy arrays.
    """
    cumulants: List[object] = []
    for n, mn in enumerate(moments, start=1):
        kn = mn
        for j in range(1, n):
            kn = kn - math.comb(n - 1, j - 1) * cumulants[j - 1] * moments[n - j - 1]
        cumulants.append(kn)
    return cumulants


KAPPAS_DEFAULT = ["0.26", "0.2705", "0.28"]
LAMBDA_DEFAULT = "0.022"
L_DEFAULT = 128
SAMPLE_TAG = "crossL128_multiL_ep10000_em_log_steps2000_2048"


# HMC rows from the table supplied in the prompt.
# Errors are encoded where the table gave parentheses.
REFERENCE_HMC_L128 = {
    "0.26": {
        "abs_phi_bar": (0.0312, 0.0006),
        "chi": (9.07, 0.37),
        "S_over_V": (0.4512, 0.0001),
        "U4": (-0.03, 0.06),
        "kappa4": (2.2e-7, 3.9e-7),
        "kappa6": (5.3e-9, 5.7e-9),
        "kappa8": (4.5e-11, 4.2e-11),
        "kappa2_s": (0.9108, None),
        "kappa4_s": (-0.2745, None),
        "kappa6_s": (0.6107, None),
        "kappa8_s": (-3.356, None),
        "U4_s": (0.1094, None),
    },
    "0.2705": {
        "abs_phi_bar": (0.438, 0.008),
        "chi": (738.0, 19.0),
        "S_over_V": (0.3988, 0.0003),
        "U4": (0.480, 0.010),
        "kappa4": (-0.0807, 0.0063),
        "kappa6": (0.125, 0.015),
        "kappa8": (-0.420, 0.067),
        "kappa2_s": (1.417, None),
        "kappa4_s": (-1.427, None),
        "kappa6_s": (8.774, None),
        "kappa8_s": (-120.6, None),
        "U4_s": (0.2362, None),
    },
    "0.28": {
        "abs_phi_bar": (1.4548, 0.0004),
        "chi": (4.30, 0.14),
        "S_over_V": (0.2383, 0.0001),
        "U4": (0.666470, 0.000006),
        "kappa4": (-8.971, 0.009),
        "kappa6": (152.0, 0.2),
        "kappa8": (-5473.0, 11.0),
        "kappa2_s": (2.689, None),
        "kappa4_s": (-9.796, None),
        "kappa6_s": (171.0, None),
        "kappa8_s": (-6400.0, None),
        "U4_s": (0.4514, None),
    },
}


OBS_ORDER = [
    "abs_phi_bar",
    "chi",
    "S_over_V",
    "U4",
    "kappa2",
    "kappa4",
    "kappa6",
    "kappa8",
    "abs_phi_ps",
    "kappa2_s_ps",
    "kappa4_s_ps",
    "kappa6_s_ps",
    "kappa8_s_ps",
    "U4_s_ps",
]

TEX_NAMES = {
    "abs_phi_bar": r"\langle|\bar\phi|\rangle",
    "chi": r"\chi",
    "S_over_V": r"\langle S\rangle/V",
    "U4": r"U_4",
    "kappa2": r"\kappa_2",
    "kappa4": r"\kappa_4",
    "kappa6": r"\kappa_6",
    "kappa8": r"\kappa_8",
    "abs_phi_ps": r"\langle|\phi|\rangle",
    "kappa2_s_ps": r"\kappa_2^{(s)}",
    "kappa4_s_ps": r"\kappa_4^{(s)}",
    "kappa6_s_ps": r"\kappa_6^{(s)}",
    "kappa8_s_ps": r"\kappa_8^{(s)}",
    "U4_s_ps": r"U_4^{(s)}",
}


STAT_KEYS = [
    "abs_m",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    "m8",
    "S",
    "min_phi",
    "max_phi",
    "max_abs_phi",
]


def guess_sample_axis(shape: Tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"expected 3D array, got shape={shape}")
    return int(np.argmax(shape))


def iter_npy_configs(path: Path, chunk_size: int) -> Iterator[np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    sample_axis = guess_sample_axis(arr.shape)
    n = arr.shape[sample_axis]
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        if sample_axis == 0:
            x = arr[start:stop, :, :]
        elif sample_axis == 1:
            x = np.moveaxis(arr[:, start:stop, :], 1, 0)
        else:
            x = np.moveaxis(arr[:, :, start:stop], 2, 0)
        yield np.asarray(x)


def iter_h5_configs(path: Path, dataset: str, chunk_size: int) -> Iterator[np.ndarray]:
    with h5py.File(path, "r") as f:
        dset = f[dataset]
        sample_axis = guess_sample_axis(dset.shape)
        n = dset.shape[sample_axis]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            if sample_axis == 0:
                x = dset[start:stop, :, :]
            elif sample_axis == 1:
                x = np.moveaxis(dset[:, start:stop, :], 1, 0)
            else:
                x = np.moveaxis(dset[:, :, start:stop], 2, 0)
            yield np.asarray(x)


def action_2d(cfgs: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    x = cfgs.astype(np.float64, copy=False)
    x2 = x * x
    x4 = x2 * x2
    nb = np.roll(x, 1, axis=1) + np.roll(x, 1, axis=2)
    dens = -2.0 * kappa * x * nb + (1.0 - 2.0 * lam) * x2 + lam * x4
    return dens.sum(axis=(1, 2))


def chunk_stats(cfgs: np.ndarray, kappa: float, lam: float) -> Dict[str, np.ndarray]:
    x = cfgs.astype(np.float64, copy=False)
    if x.ndim != 3 or x.shape[1] != x.shape[2]:
        raise ValueError(f"expected configs as (N,L,L), got {x.shape}")
    m = x.mean(axis=(1, 2))

    min_phi = x.min(axis=(1, 2))
    max_phi = x.max(axis=(1, 2))
    stats = {
        "abs_m": np.abs(m),
        "S": action_2d(x, kappa, lam),
        "min_phi": min_phi,
        "max_phi": max_phi,
        "max_abs_phi": np.maximum(np.abs(min_phi), np.abs(max_phi)),
    }
    mp = m.copy()
    for n in range(1, 9):
        stats[f"m{n}"] = mp.copy()
        if n != 8:
            mp = mp * m
    return stats


def load_stats(
    kind: str,
    path: Path,
    kappa: float,
    lam: float,
    chunk_size: int,
    h5_dataset: str = "cfgs",
) -> Tuple[Dict[str, np.ndarray], int]:
    pieces: Dict[str, List[np.ndarray]] = {k: [] for k in STAT_KEYS}
    L_seen: Optional[int] = None
    iterator: Iterable[np.ndarray]
    if kind == "npy":
        iterator = iter_npy_configs(path, chunk_size)
    elif kind == "h5":
        iterator = iter_h5_configs(path, h5_dataset, chunk_size)
    else:
        raise ValueError(f"unknown kind={kind}")

    n_total = 0
    for cfgs in iterator:
        if cfgs.ndim != 3 or cfgs.shape[1] != cfgs.shape[2]:
            raise ValueError(f"{path}: expected chunk (N,L,L), got {cfgs.shape}")
        if L_seen is None:
            L_seen = int(cfgs.shape[1])
        elif L_seen != int(cfgs.shape[1]):
            raise ValueError(f"{path}: inconsistent L in chunks")
        st = chunk_stats(cfgs, kappa, lam)
        for key in STAT_KEYS:
            pieces[key].append(st[key])
        n_total += cfgs.shape[0]
        print(f"  streamed {path.name}: {n_total} configs", flush=True)

    if L_seen is None:
        raise ValueError(f"no configs read from {path}")
    return {k: np.concatenate(v) for k, v in pieces.items()}, L_seen


def apply_mask(stats: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[mask] for k, v in stats.items()}


def moments_from_means(means: Dict[str, float], prefix: str, z2: bool, V: int = 1) -> List[float]:
    out = []
    for n in range(1, 9):
        value = means[f"{prefix}{n}"] / V
        if z2 and n % 2 == 1:
            value = 0.0
        out.append(value)
    return out


def obs_from_means(means: Dict[str, float], V: int, z2: bool) -> Dict[str, float]:
    m_mom = moments_from_means(means, "m", z2=z2)
    m_kappa = cumulants_from_raw_moments(m_mom)

    return {
        "abs_phi_bar": means["abs_m"],
        "chi": V * (m_mom[1] - means["abs_m"] ** 2),
        "S_over_V": means["S"] / V,
        "U4": 1.0 - m_mom[3] / (3.0 * m_mom[1] * m_mom[1]),
        "kappa2": m_kappa[1],
        "kappa4": m_kappa[3],
        "kappa6": m_kappa[5],
        "kappa8": m_kappa[7],
    }


def mean_stats(stats: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    keys = [
        "abs_m",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
        "S",
    ]
    if idx is None:
        return {k: float(stats[k].mean()) for k in keys}
    return {k: float(stats[k][idx].mean()) for k in keys}


def bootstrap_observables(
    stats: Dict[str, np.ndarray],
    L: int,
    n_boot: int,
    seed: int,
    z2: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    n = len(stats["m2"])
    V = L * L
    vals = obs_from_means(mean_stats(stats), V, z2=z2)
    if n_boot <= 0 or n <= 1:
        return vals, {k: float("nan") for k in vals}

    rng = np.random.default_rng(seed)
    boot = {k: np.empty(n_boot, dtype=np.float64) for k in vals}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ob = obs_from_means(mean_stats(stats, idx), V, z2=z2)
        for key, value in ob.items():
            boot[key][b] = value
    errs = {k: float(v.std(ddof=1)) for k, v in boot.items()}
    return vals, errs


def n_configs_in_file(kind: str, path: Path, h5_dataset: str = "cfgs") -> int:
    if kind == "npy":
        arr = np.load(path, mmap_mode="r")
        return int(arr.shape[guess_sample_axis(arr.shape)])
    if kind == "h5":
        with h5py.File(path, "r") as f:
            dset = f[h5_dataset]
            return int(dset.shape[guess_sample_axis(dset.shape)])
    raise ValueError(f"unknown kind={kind}")


def configs_iterator(kind: str, path: Path, chunk_size: int, h5_dataset: str = "cfgs") -> Iterator[np.ndarray]:
    if kind == "npy":
        yield from iter_npy_configs(path, chunk_size)
    elif kind == "h5":
        yield from iter_h5_configs(path, h5_dataset, chunk_size)
    else:
        raise ValueError(f"unknown kind={kind}")


def per_site_obs_from_moments(moment_arrays: List[np.ndarray], abs_array: np.ndarray) -> Dict[str, float]:
    cumulants = cumulants_from_raw_moments(moment_arrays)
    m2 = moment_arrays[1]
    m4 = moment_arrays[3]
    U4_site = 1.0 - m4 / (3.0 * m2 * m2)
    return {
        "abs_phi_ps": float(abs_array.mean()),
        "kappa2_s_ps": float(cumulants[1].mean()),
        "kappa4_s_ps": float(cumulants[3].mean()),
        "kappa6_s_ps": float(cumulants[5].mean()),
        "kappa8_s_ps": float(cumulants[7].mean()),
        "U4_s_ps": float(U4_site.mean()),
    }


def per_site_observables(
    kind: str,
    path: Path,
    L: int,
    chunk_size: int,
    n_blocks: int,
    z2: bool,
    h5_dataset: str = "cfgs",
    mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    n_total = n_configs_in_file(kind, path, h5_dataset)
    if mask is None:
        mask = np.ones(n_total, dtype=bool)
    if len(mask) != n_total:
        raise ValueError(f"mask length {len(mask)} does not match n_configs {n_total} for {path}")
    n_used = int(mask.sum())
    if n_used == 0:
        raise ValueError(f"no configs selected for per-site observables: {path}")

    B = max(1, min(int(n_blocks), n_used))
    sums = np.zeros((B, 8, L, L), dtype=np.float64)
    abs_sums = np.zeros((B, L, L), dtype=np.float64)
    counts = np.zeros(B, dtype=np.int64)

    global_start = 0
    selected_seen = 0
    for cfgs in configs_iterator(kind, path, chunk_size, h5_dataset):
        n_chunk = cfgs.shape[0]
        chunk_mask = mask[global_start:global_start + n_chunk]
        global_start += n_chunk
        if not chunk_mask.any():
            continue
        x = cfgs[chunk_mask].astype(np.float64, copy=False)
        n_sel = x.shape[0]
        ranks = np.arange(selected_seen, selected_seen + n_sel)
        block_ids = np.minimum((ranks * B) // n_used, B - 1)
        selected_seen += n_sel

        for b in np.unique(block_ids):
            xb = x[block_ids == b]
            counts[b] += xb.shape[0]
            abs_sums[b] += np.abs(xb).sum(axis=0)
            xp = xb
            for n in range(1, 9):
                if (not z2) or (n % 2 == 0):
                    sums[b, n - 1] += xp.sum(axis=0)
                if n != 8:
                    xp = xp * xb

    total_count = int(counts.sum())
    if total_count != n_used:
        raise RuntimeError(f"selected count mismatch for {path}: saw {total_count}, expected {n_used}")

    total_sums = sums.sum(axis=0)
    total_abs = abs_sums.sum(axis=0)
    full_moments = [total_sums[n] / total_count for n in range(8)]
    if z2:
        for n in range(0, 8, 2):
            full_moments[n] = np.zeros((L, L), dtype=np.float64)
    full_abs = total_abs / total_count
    vals = per_site_obs_from_moments(full_moments, full_abs)

    if B <= 1:
        return vals, {k: float("nan") for k in vals}

    jk_vals: Dict[str, List[float]] = {k: [] for k in vals}
    for b in range(B):
        leave_count = total_count - int(counts[b])
        if leave_count <= 0:
            continue
        leave_sums = total_sums - sums[b]
        leave_abs = total_abs - abs_sums[b]
        moments = [leave_sums[n] / leave_count for n in range(8)]
        if z2:
            for n in range(0, 8, 2):
                moments[n] = np.zeros((L, L), dtype=np.float64)
        ob = per_site_obs_from_moments(moments, leave_abs / leave_count)
        for key, value in ob.items():
            jk_vals[key].append(value)
    errs = {}
    for key, samples in jk_vals.items():
        arr = np.asarray(samples, dtype=np.float64)
        errs[key] = float(np.sqrt((len(arr) - 1) * np.mean((arr - vals[key]) ** 2))) if len(arr) > 1 else float("nan")
    return vals, errs


def fmt_value(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.6e}"
    return f"{x:.8g}"


def fmt_pm(value: float, error: Optional[float]) -> str:
    if error is None or not np.isfinite(error):
        return fmt_value(value)
    return f"{fmt_value(value)} +/- {fmt_value(error)}"


def format_err_latex(value: float, error: Optional[float], sig_digits: int = 1) -> str:
    if error is None or not np.isfinite(error) or error == 0:
        return fmt_value(value)
    exp_err = int(math.floor(math.log10(abs(error))))
    factor = 10 ** (exp_err - sig_digits + 1)
    err_rounded = round(error / factor) * factor
    if err_rounded == 0:
        return fmt_value(value)
    exp_err = int(math.floor(math.log10(abs(err_rounded))))
    decimals = max(0, -(exp_err - sig_digits + 1))
    if decimals == 0:
        val_str = str(int(round(value)))
        err_str = str(int(round(err_rounded)))
    else:
        val_str = f"{value:.{decimals}f}"
        err_str = str(int(round(err_rounded * 10 ** decimals)))
    return f"{val_str}({err_str})"


def result_record(
    kappa: str,
    source: str,
    n_raw: int,
    n_used: int,
    L: int,
    z2: bool,
    vals: Dict[str, float],
    errs: Dict[str, float],
    filters: Dict[str, object],
) -> Dict[str, object]:
    return {
        "kappa": kappa,
        "source": source,
        "L": L,
        "n_raw": n_raw,
        "n_used": n_used,
        "n_z2_effective": 2 * n_used if z2 else n_used,
        "z2_symmetrized": z2,
        "filters": filters,
        "values": vals,
        "errors": errs,
    }


def paths_for_kappa(base: Path, kappa: str, lam: str) -> Tuple[Path, Path]:
    run = base / "runs" / f"phi4_Lmulti8-16-32-64_k{kappa}_l{lam}_ncsnpp"
    dm_path = run / "data_crossL" / f"samples_{SAMPLE_TAG}.npy"
    hmc_path = base / "trainingdata" / f"cfgs_wolff_fahmc_k={kappa}_l={lam}_128^2.jld2"
    return dm_path, hmc_path


def merge_observables(
    global_vals: Dict[str, float],
    global_errs: Dict[str, float],
    per_site_vals: Dict[str, float],
    per_site_errs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    vals = dict(global_vals)
    errs = dict(global_errs)
    vals.update(per_site_vals)
    errs.update(per_site_errs)
    return vals, errs


def compute_one(args: argparse.Namespace, kappa: str) -> List[Dict[str, object]]:
    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lam = float(args.lam)
    k_float = float(kappa)
    dm_path, hmc_path = paths_for_kappa(base, kappa, args.lam)
    if not dm_path.exists():
        raise FileNotFoundError(dm_path)
    if not hmc_path.exists():
        raise FileNotFoundError(hmc_path)

    print(f"\n========== kappa={kappa} ==========")
    print("compute precision: numpy float64")
    print(f"HMC: {hmc_path}")
    hmc_stats, L_hmc = load_stats("h5", hmc_path, k_float, lam, args.chunk_size, args.h5_dataset)
    print(f"DM : {dm_path}")
    dm_stats, L_dm = load_stats("npy", dm_path, k_float, lam, args.chunk_size)
    if L_hmc != args.L or L_dm != args.L:
        raise ValueError(f"unexpected L for k={kappa}: HMC L={L_hmc}, DM L={L_dm}")

    hmc_filters = {"description": "none"}
    hmc_raw_global_vals, hmc_raw_global_errs = bootstrap_observables(
        hmc_stats, args.L, args.n_boot, args.seed + int(float(kappa) * 100000),
        z2=False,
    )
    hmc_raw_ps_vals, hmc_raw_ps_errs = per_site_observables(
        "h5", hmc_path, args.L, args.chunk_size, args.n_site_blocks,
        z2=False, h5_dataset=args.h5_dataset,
    )
    hmc_raw_vals, hmc_raw_errs = merge_observables(
        hmc_raw_global_vals, hmc_raw_global_errs, hmc_raw_ps_vals, hmc_raw_ps_errs
    )
    hmc_raw_record = result_record(
        kappa, "HMC raw", len(hmc_stats["m2"]), len(hmc_stats["m2"]), args.L,
        False, hmc_raw_vals, hmc_raw_errs, hmc_filters,
    )

    hmc_z2_global_vals, hmc_z2_global_errs = bootstrap_observables(
        hmc_stats, args.L, args.n_boot, args.seed + 3 + int(float(kappa) * 100000),
        z2=True,
    )
    hmc_z2_ps_vals, hmc_z2_ps_errs = per_site_observables(
        "h5", hmc_path, args.L, args.chunk_size, args.n_site_blocks,
        z2=True, h5_dataset=args.h5_dataset,
    )
    hmc_z2_vals, hmc_z2_errs = merge_observables(
        hmc_z2_global_vals, hmc_z2_global_errs, hmc_z2_ps_vals, hmc_z2_ps_errs
    )
    hmc_z2_record = result_record(
        kappa, "HMC Z2", len(hmc_stats["m2"]), len(hmc_stats["m2"]), args.L,
        True, hmc_z2_vals, hmc_z2_errs, hmc_filters,
    )

    dm_mask = np.ones(len(dm_stats["m2"]), dtype=bool)
    filter_desc = []
    filter_names = []
    if args.filter_dm_site_support:
        site_lim = float(hmc_stats["max_abs_phi"].max())
        dm_mask &= dm_stats["max_abs_phi"] <= site_lim
        filter_names.append("site_abs_max_from_hmc_z2")
        filter_desc.append({"name": "site_abs_max_from_hmc_z2", "limit": site_lim})
    if args.filter_dm_magnetization_support:
        m_lim = float(hmc_stats["abs_m"].max())
        dm_mask &= dm_stats["abs_m"] <= m_lim
        filter_names.append("abs_m_max_from_hmc_z2")
        filter_desc.append({"name": "abs_m_max_from_hmc_z2", "limit": m_lim})
    dm_used = apply_mask(dm_stats, dm_mask)
    if len(dm_used["m2"]) == 0:
        raise RuntimeError(f"all DM configs filtered out for k={kappa}")
    dm_global_vals, dm_global_errs = bootstrap_observables(
        dm_used, args.L, args.n_boot, args.seed + 17 + int(float(kappa) * 100000),
        z2=True,
    )
    dm_ps_vals, dm_ps_errs = per_site_observables(
        "npy", dm_path, args.L, args.chunk_size, args.n_site_blocks,
        z2=True, h5_dataset=args.h5_dataset, mask=dm_mask,
    )
    dm_vals, dm_errs = merge_observables(
        dm_global_vals, dm_global_errs, dm_ps_vals, dm_ps_errs
    )
    dm_filters = {
        "description": "none" if not filter_names else " and ".join(filter_names),
        "details": filter_desc,
        "n_dropped": int((~dm_mask).sum()),
    }
    dm_record = result_record(
        kappa, "DM multi-L Z2", len(dm_stats["m2"]), len(dm_used["m2"]), args.L,
        True, dm_vals, dm_errs, dm_filters,
    )

    records = [hmc_raw_record, hmc_z2_record, dm_record]
    single_json = out_dir / f"multiL_L128_k{kappa}_observables.json"
    with single_json.open("w") as f:
        json.dump({"records": records}, f, indent=2)
    print(f"saved {single_json}")

    write_single_report(out_dir / f"multiL_L128_k{kappa}_observables.txt", records)
    return records


def write_single_report(path: Path, records: List[Dict[str, object]]) -> None:
    kappa = str(records[0]["kappa"])
    lines = [f"2D phi4 L=128 observables, kappa={kappa}", ""]
    for rec in records:
        lines.append(
            f"{rec['source']}: N_raw={rec['n_raw']} N_used={rec['n_used']} "
            f"N_effective={rec['n_z2_effective']} z2={rec['z2_symmetrized']}"
        )
        vals = rec["values"]
        errs = rec["errors"]
        for obs in OBS_ORDER:
            lines.append(f"  {obs:>12s}: {fmt_pm(vals[obs], errs[obs])}")
        lines.append("")

    lines.append("HMC Z2 comparison against prompt reference:")
    lines.append("  PASS means the center value is consistent with the prompt reference.")
    lines.append("  WARN means this run did not strictly reproduce that table entry.")
    hmc = next(r for r in records if r["source"] == "HMC Z2")
    ref = REFERENCE_HMC_L128.get(kappa, {})
    validation_pairs = [
        ("abs_phi_bar", "abs_phi_bar"),
        ("chi", "chi"),
        ("S_over_V", "S_over_V"),
        ("U4", "U4"),
        ("kappa4", "kappa4"),
        ("kappa6", "kappa6"),
        ("kappa8", "kappa8"),
        ("kappa2_s_ps", "kappa2_s"),
        ("kappa4_s_ps", "kappa4_s"),
        ("kappa6_s_ps", "kappa6_s"),
        ("kappa8_s_ps", "kappa8_s"),
        ("U4_s_ps", "U4_s"),
    ]
    for obs, ref_obs in validation_pairs:
        if ref_obs not in ref:
            continue
        got = hmc["values"][obs]
        got_err = hmc["errors"][obs]
        ref_val, ref_err = ref[ref_obs]
        diff = got - ref_val
        if ref_err is not None and ref_err > 0:
            z = diff / math.sqrt(got_err * got_err + ref_err * ref_err)
            status = "PASS" if abs(z) <= 3.0 else "WARN"
            lines.append(
                f"  {status:4s} {obs:>12s}: got {fmt_pm(got, got_err)} | "
                f"ref {fmt_pm(ref_val, ref_err)} | diff={fmt_value(diff)} | z={z:+.2f}"
            )
        else:
            rel = diff / ref_val if ref_val != 0 else float("nan")
            status = "PASS" if abs(rel) <= 0.01 else "WARN"
            lines.append(
                f"  {status:4s} {obs:>12s}: got {fmt_pm(got, got_err)} | "
                f"ref {fmt_value(ref_val)} | diff={fmt_value(diff)} | rel={rel:+.3%}"
            )
    path.write_text("\n".join(lines) + "\n")
    print(f"saved {path}")


def collect_records(out_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(out_dir.glob("multiL_L128_k*_observables.json")):
        with path.open() as f:
            data = json.load(f)
        records.extend(data["records"])
    source_order = {"HMC raw": 0, "HMC Z2": 1, "DM multi-L Z2": 2}
    records.sort(key=lambda r: (float(r["kappa"]), source_order.get(str(r["source"]), 99)))
    return records


def write_combined_outputs(out_dir: Path, records: List[Dict[str, object]]) -> None:
    csv_path = out_dir / "multiL_L128_observables_summary.csv"
    ref_key = {
        "kappa2_s_ps": "kappa2_s",
        "kappa4_s_ps": "kappa4_s",
        "kappa6_s_ps": "kappa6_s",
        "kappa8_s_ps": "kappa8_s",
        "U4_s_ps": "U4_s",
    }
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "kappa", "source", "observable", "value", "error",
            "n_raw", "n_used", "n_z2_effective", "ref_value", "ref_error",
            "delta_vs_ref", "z_vs_ref",
        ])
        for rec in records:
            kappa = str(rec["kappa"])
            ref = REFERENCE_HMC_L128.get(kappa, {}) if rec["source"] == "HMC Z2" else {}
            for obs in OBS_ORDER:
                val = rec["values"][obs]
                err = rec["errors"][obs]
                mapped = ref_key.get(obs, obs)
                ref_val = ref.get(mapped, (None, None))[0]
                ref_err = ref.get(mapped, (None, None))[1]
                delta = None if ref_val is None else val - ref_val
                z = None
                if delta is not None and ref_err is not None and ref_err > 0:
                    z = delta / math.sqrt(err * err + ref_err * ref_err)
                w.writerow([
                    kappa, rec["source"], obs, val, err,
                    rec["n_raw"], rec["n_used"], rec["n_z2_effective"],
                    ref_val, ref_err, delta, z,
                ])
    print(f"saved {csv_path}")

    txt_path = out_dir / "multiL_L128_observables_summary.txt"
    lines = ["Combined 2D phi4 multi-L L=128 observable summary", ""]
    for rec in records:
        lines.append(
            f"k={rec['kappa']}  {rec['source']}  "
            f"N={rec['n_used']} raw, N_effective={rec['n_z2_effective']}, "
            f"z2={rec['z2_symmetrized']}"
        )
        cells = [f"{obs}={fmt_pm(rec['values'][obs], rec['errors'][obs])}" for obs in OBS_ORDER]
        lines.extend("  " + c for c in cells)
        lines.append("")
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"saved {txt_path}")

    tex_path = out_dir / "multiL_L128_observables_summary.tex"
    lines = [
        r"% Auto-generated by compute_multiL_L128_observables.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Scalar observables for 2D $\phi^4$ at $L=128$, $\lambda=0.022$. "
        r"All ensembles are evaluated with $Z_2$ symmetrization.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(OBS_ORDER) + r"}",
        r"\toprule",
        r"$\kappa$ & source & " + " & ".join(f"${TEX_NAMES[o]}$" for o in OBS_ORDER) + r" \\",
        r"\midrule",
    ]
    last_k = None
    for rec in records:
        kappa = str(rec["kappa"])
        if last_k is not None and kappa != last_k:
            lines.append(r"\midrule")
        last_k = kappa
        vals = rec["values"]
        errs = rec["errors"]
        row = [f"${kappa}$", str(rec["source"])]
        row.extend(f"${format_err_latex(vals[o], errs[o])}$" for o in OBS_ORDER)
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"saved {tex_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="/work/gw97/w97000/project/DM/2Dphi4")
    p.add_argument("--out_dir", default="results/multiL_L128_observables")
    p.add_argument("--kappas", nargs="+", default=KAPPAS_DEFAULT)
    p.add_argument("--L", type=int, default=L_DEFAULT)
    p.add_argument("--lam", default=LAMBDA_DEFAULT)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--n_boot", type=int, default=800)
    p.add_argument("--n_site_blocks", type=int, default=32,
                   help="Number of config blocks for per-site spatial-average jackknife errors.")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--h5_dataset", default="cfgs")
    p.add_argument("--filter_dm_site_support", action="store_true")
    p.add_argument("--filter_dm_magnetization_support", action="store_true")
    p.add_argument("--combine_only", action="store_true")
    p.add_argument("--no_combine", action="store_true",
                   help="Write per-kappa outputs only; useful for MPI workers.")
    p.add_argument("--allow_login", action="store_true",
                   help="Allow direct execution without PBS_NODEFILE. Use only for tiny tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_login and not os.environ.get("PBS_NODEFILE"):
        sys.exit(
            "Refusing to compute without PBS_NODEFILE. Enter an interactive PBS "
            "allocation first, then run the shell driver."
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.combine_only:
        for kappa in args.kappas:
            compute_one(args, kappa)

    if not args.no_combine:
        records = collect_records(out_dir)
        if records:
            write_combined_outputs(out_dir, records)
        else:
            print(f"no per-kappa JSON files found in {out_dir}")


if __name__ == "__main__":
    main()
