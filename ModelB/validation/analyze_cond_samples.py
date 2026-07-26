"""Validate conditional DM samples against the true Model B evolution.

Compares, at matched ensemble indices: zero-mode/conservation, field moments,
radially averaged structure factors, and the two-time correlator
``C(0,h) = <(1/V) sum_x phi0 phi_h>``.

Usage (both forms are supported)::

    python validation/analyze_cond_samples.py --samples <run_dir>/samples/<...>.npz
    python -m validation.analyze_cond_samples --samples <run_dir>/samples/<...>.npz
"""

import argparse
from datetime import datetime, timezone
import json
import os
import tempfile

import h5py
import numpy as np


_HDF_META_ALIASES = {
    "L": ("L", "N"),
    "kappa": ("kappa", "κ"),
    "lambda": ("lambda", "lam", "λ"),
    "time": ("time", "horizon"),
    "source_path": ("source_path",),
}

_JSON_META_ALIASES = {
    "L": ("L",),
    "kappa": ("kappa",),
    "lambda": ("lambda", "lam"),
    "time": ("time", "horizon"),
    "source_path": ("source_path", "cond_path"),
}


def _decode_scalar(value):
    """Convert an HDF5/NumPy scalar into a plain Python scalar."""
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f"expected scalar metadata, got shape {value.shape}")
    value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def hdf_metadata(handle):
    """Read the canonical Model B metadata fields present in an HDF5 file."""
    result = {}
    for canonical, aliases in _HDF_META_ALIASES.items():
        for key in aliases:
            if key in handle:
                result[canonical] = _decode_scalar(handle[key][()])
                break
            if key in handle.attrs:
                result[canonical] = _decode_scalar(handle.attrs[key])
                break
    return result


def canonical_json_metadata(mapping):
    """Return canonical metadata names from a DM JSON mapping."""
    result = {}
    for canonical, aliases in _JSON_META_ALIASES.items():
        for key in aliases:
            if key in mapping and mapping[key] is not None:
                result[canonical] = mapping[key]
                break
    return result


def _require_metadata(meta, fields, source):
    missing = [field for field in fields if field not in meta]
    if missing:
        raise ValueError(f"{source} is missing metadata: {', '.join(missing)}")


def _check_numeric_match(field, left, left_source, right, right_source):
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {field} metadata") from exc
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        raise ValueError(f"non-finite {field} metadata")
    if not np.isclose(left_value, right_value, rtol=2e-6, atol=2e-7):
        raise ValueError(
            f"{field} mismatch: {left_source}={left!r}, "
            f"{right_source}={right!r}"
        )


def check_metadata_match(left, left_source, right, right_source, fields):
    """Require selected canonical numeric fields to agree."""
    _require_metadata(left, fields, left_source)
    _require_metadata(right, fields, right_source)
    for field in fields:
        _check_numeric_match(
            field, left[field], left_source, right[field], right_source
        )


def resolve_metadata_path(path, metadata_path):
    """Resolve a possibly-relative path with respect to its metadata file."""
    path = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(metadata_path)), path)
    return os.path.realpath(os.path.abspath(path))


def check_source_match(left, left_metadata_path, right, right_metadata_path):
    """Require two source paths to resolve to the same file-system path."""
    left_resolved = resolve_metadata_path(left, left_metadata_path)
    right_resolved = resolve_metadata_path(right, right_metadata_path)
    if left_resolved != right_resolved:
        raise ValueError(
            "source_path mismatch: "
            f"{left_metadata_path} -> {left_resolved}, "
            f"{right_metadata_path} -> {right_resolved}"
        )
    return left_resolved


def _as_positive_int_scalar(value, name):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    scalar = array.item()
    try:
        integer = int(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if integer <= 0 or integer != scalar:
        raise ValueError(f"{name} must be a positive integer, got {scalar!r}")
    return integer


def validate_indices(indices, name="cond_indices"):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype, got {indices.dtype}")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0):
        raise ValueError(f"{name} contains a negative index")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate conditions")
    return indices


def _validate_field_array(array, name):
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N,L,L), got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must be non-empty and square, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got {array.dtype}")
    for start in range(0, array.shape[0], 64):
        if not np.isfinite(array[start:start + 64]).all():
            raise ValueError(f"{name} contains NaN or Inf")
    return array


def load_dm_archive(path, require_dphi=True):
    """Load and validate a conditional sample archive without dtype expansion."""
    required = {"phi_gen", "cond_indices", "n_per_cond"}
    if require_dphi:
        required.update(("dphi", "zero_mode_raw"))
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        phi_gen = _validate_field_array(archive["phi_gen"], "phi_gen")
        cond_indices = validate_indices(archive["cond_indices"])
        n_per_cond = _as_positive_int_scalar(
            archive["n_per_cond"], "n_per_cond"
        )
        result = {
            "phi_gen": phi_gen,
            "cond_indices": cond_indices,
            "n_per_cond": n_per_cond,
        }
        expected = cond_indices.size * n_per_cond
        if phi_gen.shape[0] != expected:
            raise ValueError(
                f"phi_gen has {phi_gen.shape[0]} samples, expected "
                f"{cond_indices.size} x {n_per_cond} = {expected}"
            )
        if require_dphi:
            dphi = _validate_field_array(archive["dphi"], "dphi")
            if dphi.shape != phi_gen.shape:
                raise ValueError(
                    f"dphi shape {dphi.shape} != phi_gen shape {phi_gen.shape}"
                )
            zero_mode_raw = np.asarray(archive["zero_mode_raw"])
            if zero_mode_raw.shape != (expected,):
                raise ValueError(
                    "zero_mode_raw must have shape "
                    f"({expected},), got {zero_mode_raw.shape}"
                )
            if not np.issubdtype(zero_mode_raw.dtype, np.number):
                raise ValueError("zero_mode_raw must be numeric")
            if not np.isfinite(zero_mode_raw).all():
                raise ValueError("zero_mode_raw contains NaN or Inf")
            result.update(dphi=dphi, zero_mode_raw=zero_mode_raw)
    return result


def load_json_object(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"required metadata file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_indexed_cfgs(path, indices, dtype=np.float64):
    """Read only selected ensemble members from an HDF5 ``cfgs`` dataset.

    h5py requires increasing indices for fancy indexing, so unique sorted rows
    are read once and then restored to the requested order in memory.  The full
    HDF5 dataset is never converted to float64.
    """
    indices = validate_indices(indices)
    with h5py.File(path, "r") as handle:
        if "cfgs" not in handle:
            raise ValueError(f"{path} has no 'cfgs' dataset")
        dataset = handle["cfgs"]
        if dataset.ndim != 3 or dataset.shape[1] != dataset.shape[2]:
            raise ValueError(
                f"{path}: cfgs must have h5py shape (N,L,L), got {dataset.shape}"
            )
        if indices[-1] >= dataset.shape[0] or indices.max() >= dataset.shape[0]:
            raise IndexError(
                f"condition index {indices.max()} is outside cfgs[0:{dataset.shape[0]}]"
            )
        unique, inverse = np.unique(indices, return_inverse=True)
        selected = dataset[unique.tolist(), :, :]
        if not np.array_equal(indices, unique):
            selected = selected[inverse]
        if dtype is not None:
            selected = np.asarray(selected, dtype=dtype)
        else:
            selected = np.asarray(selected)
        metadata = hdf_metadata(handle)
    selected = _validate_field_array(selected, f"{path}:cfgs[selected]")
    return selected, metadata


def radial_sk(fields, fft_batch_size=64):
    """Return the radial structure factor for ``(N,L,L)`` fields.

    Shell ``m`` contains modes with ``m-0.5 <= |k| < m+0.5``.  Only nonzero
    modes satisfying ``|k| <= L/2`` are eligible.  In particular, modes in the
    square Fourier-grid corners are excluded instead of being folded into the
    final Nyquist shell.
    """
    fields = _validate_field_array(fields, "fields")
    if fft_batch_size <= 0:
        raise ValueError("fft_batch_size must be positive")
    n, lattice_size, _ = fields.shape
    power_sum = np.zeros((lattice_size, lattice_size), dtype=np.float64)
    for start in range(0, n, fft_batch_size):
        fourier = np.fft.fft2(fields[start:start + fft_batch_size], axes=(-2, -1))
        power_sum += np.sum(fourier.real**2 + fourier.imag**2, axis=0)
    power = power_sum / (n * lattice_size * lattice_size)

    frequencies = np.fft.fftfreq(lattice_size) * lattice_size
    ky, kx = np.meshgrid(frequencies, frequencies, indexing="ij")
    kmag = np.hypot(kx, ky)
    max_shell = lattice_size // 2
    shell = np.floor(kmag + 0.5).astype(np.int64)
    eligible = (kmag > 0.0) & (kmag <= lattice_size / 2.0)
    eligible &= (shell >= 1) & (shell <= max_shell)

    shell_ids = shell[eligible]
    counts = np.bincount(shell_ids, minlength=max_shell + 1)
    k_sum = np.bincount(
        shell_ids, weights=kmag[eligible], minlength=max_shell + 1
    )
    power_shell_sum = np.bincount(
        shell_ids, weights=power[eligible], minlength=max_shell + 1
    )
    present = counts[1:] > 0
    kk = k_sum[1:][present] / counts[1:][present]
    sk = power_shell_sum[1:][present] / counts[1:][present]
    return kk, sk


def sem(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("sem expects a 1-D array")
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def _spatial_moment(fields, order, batch_size=64):
    fields = _validate_field_array(fields, "fields")
    result = np.empty(fields.shape[0], dtype=np.float64)
    for start in range(0, fields.shape[0], batch_size):
        stop = min(start + batch_size, fields.shape[0])
        chunk = np.asarray(fields[start:stop], dtype=np.float64)
        result[start:stop] = np.mean(chunk**order, axis=(1, 2))
    return result


def _spatial_std(fields, batch_size=64):
    fields = _validate_field_array(fields, "fields")
    result = np.empty(fields.shape[0], dtype=np.float64)
    for start in range(0, fields.shape[0], batch_size):
        stop = min(start + batch_size, fields.shape[0])
        result[start:stop] = np.std(
            fields[start:stop], axis=(1, 2), dtype=np.float64
        )
    return result


def _condition_average(rep_values, ncond, n_per_cond):
    values = np.asarray(rep_values, dtype=np.float64)
    if values.shape != (ncond * n_per_cond,):
        raise ValueError("replicate statistic has an unexpected shape")
    return values.reshape(ncond, n_per_cond).mean(axis=1)


def _relative_deviation(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    if not np.any(valid):
        raise ValueError("no positive finite reference S(k) bins")
    return numerator[valid] / denominator[valid] - 1.0, valid


def _safe_corrcoef(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def ensure_outputs_writable(paths, overwrite):
    existing = [path for path in paths if os.path.lexists(path)]
    if existing and not overwrite:
        joined = "\n  ".join(existing)
        raise FileExistsError(
            "refusing to overwrite existing validation output(s):\n  "
            f"{joined}\nPass --overwrite to replace them explicitly."
        )


def _publish_file(temporary, path, overwrite):
    if overwrite:
        os.replace(temporary, path)
    else:
        # Atomic no-replace publication; the temporary lives in the same dir.
        os.link(temporary, path)
        os.remove(temporary)


def atomic_savetxt(path, values, header, overwrite=False):
    descriptor, temporary = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}."
    )
    os.close(descriptor)
    try:
        np.savetxt(temporary, values, header=header)
        os.chmod(temporary, 0o640)
        _publish_file(temporary, path, overwrite)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_json_dump(path, value, overwrite=False):
    descriptor, temporary = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}."
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o640)
        _publish_file(temporary, path, overwrite)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _validate_dm_json(meta, meta_path, archive, lattice_size):
    if meta.get("n_per_cond") is not None:
        json_n = _as_positive_int_scalar(meta["n_per_cond"], "JSON n_per_cond")
        if json_n != archive["n_per_cond"]:
            raise ValueError(
                f"n_per_cond mismatch: {meta_path}={json_n}, "
                f"archive={archive['n_per_cond']}"
            )
    if meta.get("n_cond") is not None:
        json_ncond = _as_positive_int_scalar(meta["n_cond"], "JSON n_cond")
        if json_ncond != archive["cond_indices"].size:
            raise ValueError(
                f"n_cond mismatch: {meta_path}={json_ncond}, "
                f"archive={archive['cond_indices'].size}"
            )
    canonical = canonical_json_metadata(meta)
    _require_metadata(
        canonical, ("kappa", "lambda", "time", "source_path"), meta_path
    )
    if "L" in canonical:
        _check_numeric_match("L", canonical["L"], meta_path, lattice_size, "archive")
    canonical["L"] = lattice_size
    return canonical


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=str, required=True, help="Path to cond_samples_*.npz"
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="True phi(t0+h) file (default: sample/run metadata)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace validation outputs with the same name",
    )
    args = parser.parse_args(argv)

    samples_path = os.path.realpath(os.path.abspath(args.samples))
    archive = load_dm_archive(samples_path, require_dphi=True)
    phi_gen = archive["phi_gen"]
    dphi = archive["dphi"]
    cond_indices = archive["cond_indices"]
    n_per_cond = archive["n_per_cond"]
    zero_mode_raw = archive["zero_mode_raw"]
    ncond = cond_indices.size
    lattice_size = phi_gen.shape[-1]

    sample_meta_path = os.path.splitext(samples_path)[0] + ".json"
    sample_meta = load_json_object(sample_meta_path)
    dm_meta = _validate_dm_json(
        sample_meta, sample_meta_path, archive, lattice_size
    )
    cond_path = resolve_metadata_path(dm_meta["source_path"], sample_meta_path)

    run_dir = os.path.dirname(os.path.dirname(samples_path))
    run_meta_path = os.path.join(run_dir, "meta.json")
    run_meta = load_json_object(run_meta_path) if os.path.isfile(run_meta_path) else {}
    out_dir = os.path.join(run_dir, "validation")
    base = os.path.splitext(os.path.basename(samples_path))[0]
    sk_path = os.path.join(out_dir, base + "_sk.dat")
    scatter_path = os.path.join(out_dir, base + "_c0h_scatter.dat")
    provenance_path = os.path.join(out_dir, base + "_analysis.json")
    # Reject a collision before loading HDF5 data or doing FFTs.
    ensure_outputs_writable(
        (sk_path, scatter_path, provenance_path), args.overwrite
    )
    if run_meta:
        run_canonical = canonical_json_metadata(run_meta)
        check_metadata_match(
            dm_meta,
            sample_meta_path,
            run_canonical,
            run_meta_path,
            ("L", "kappa", "lambda", "time"),
        )
        _require_metadata(run_canonical, ("source_path",), run_meta_path)
        check_source_match(
            dm_meta["source_path"],
            sample_meta_path,
            run_canonical["source_path"],
            run_meta_path,
        )

    target_value = args.target_path or sample_meta.get("target_path")
    target_metadata_path = sample_meta_path
    if target_value is None and run_meta:
        target_value = run_meta.get("target_path")
        target_metadata_path = run_meta_path
    if target_value is None:
        raise ValueError(
            "no target_path: pass --target_path or provide it in sample/run metadata"
        )
    target_path = resolve_metadata_path(target_value, target_metadata_path)

    phi0, cond_hdf_meta = read_indexed_cfgs(cond_path, cond_indices)
    phi_true, target_hdf_meta = read_indexed_cfgs(target_path, cond_indices)
    if phi0.shape != (ncond, lattice_size, lattice_size):
        raise ValueError(f"condition shape {phi0.shape} is incompatible with DM samples")
    if phi_true.shape != phi0.shape:
        raise ValueError(
            f"target shape {phi_true.shape} != condition shape {phi0.shape}"
        )

    check_metadata_match(
        dm_meta,
        sample_meta_path,
        cond_hdf_meta,
        cond_path,
        ("L", "kappa", "lambda"),
    )
    check_metadata_match(
        dm_meta,
        sample_meta_path,
        target_hdf_meta,
        target_path,
        ("L", "kappa", "lambda", "time"),
    )
    _require_metadata(target_hdf_meta, ("source_path",), target_path)
    check_source_match(
        dm_meta["source_path"],
        sample_meta_path,
        target_hdf_meta["source_path"],
        target_path,
    )

    phi_gen_by_cond = phi_gen.reshape(
        ncond, n_per_cond, lattice_size, lattice_size
    )
    dphi_by_cond = dphi.reshape(ncond, n_per_cond, lattice_size, lattice_size)
    relation_error = 0.0
    for condition in range(ncond):
        discrepancy = (
            np.asarray(phi_gen_by_cond[condition], dtype=np.float64)
            - phi0[condition]
            - dphi_by_cond[condition]
        )
        relation_error = max(relation_error, float(np.max(np.abs(discrepancy))))
    scale = max(1.0, float(np.max(np.abs(phi_gen))))
    if relation_error > 2e-5 * scale:
        raise ValueError(
            "phi_gen is inconsistent with phi0 + dphi: "
            f"max error {relation_error:.3e}"
        )

    dphi_true = phi_true - phi0
    volume = lattice_size * lattice_size

    print(f"=== {os.path.basename(samples_path)} ===")
    print(
        f"{ncond} conditions x {n_per_cond} samples, L={lattice_size}, "
        f"horizon t={dm_meta['time']}, kappa={dm_meta['kappa']}"
    )
    print(f"condition source: {cond_path}")
    print(f"true target:     {target_path}")

    print("\n--- zero mode / conservation ---")
    print(
        "raw DM zero mode of dphi (physical): "
        f"mean|.| = {np.abs(zero_mode_raw).mean():.4e}, "
        f"max|.| = {np.abs(zero_mode_raw).max():.4e}"
    )
    true_std = float(dphi_true.std(dtype=np.float64))
    relative_zero_mode = (
        float(np.abs(zero_mode_raw).mean()) / true_std
        if true_std > 0
        else float("nan")
    )
    print(
        f"  relative to per-site dphi std {true_std:.4f}: "
        f"{relative_zero_mode:.3e}"
    )
    generated_magnetization = phi_gen_by_cond.mean(axis=(2, 3), dtype=np.float64)
    initial_magnetization = phi0.mean(axis=(1, 2), dtype=np.float64)
    cons_err = float(
        np.max(np.abs(generated_magnetization - initial_magnetization[:, None]))
    )
    print(f"saved-sample |M(phi_gen) - M(phi0)| max = {cons_err:.3e}")

    # Replicates are first averaged inside each condition; uncertainties then
    # use conditions as the independent sampling level.
    print("\n--- ensemble observables (all DM replicates; condition-level SEM) ---")
    p2g = _condition_average(
        _spatial_moment(phi_gen, 2), ncond, n_per_cond
    )
    p2t = _spatial_moment(phi_true, 2)
    print(
        f"<phi^2>: gen {p2g.mean():.5f} +- {sem(p2g):.5f}   "
        f"true {p2t.mean():.5f} +- {sem(p2t):.5f}"
    )
    p4g = _condition_average(
        _spatial_moment(phi_gen, 4), ncond, n_per_cond
    )
    p4t = _spatial_moment(phi_true, 4)
    print(
        f"<phi^4>: gen {p4g.mean():.5f} +- {sem(p4g):.5f}   "
        f"true {p4t.mean():.5f} +- {sem(p4t):.5f}"
    )
    dsg = _condition_average(_spatial_std(dphi), ncond, n_per_cond)
    dst = _spatial_std(dphi_true)
    std_ratio = float(dsg.mean() / dst.mean()) if dst.mean() > 0 else float("nan")
    print(
        f"dphi spatial std: gen {dsg.mean():.5f} +- {sem(dsg):.5f}   "
        f"true {dst.mean():.5f} +- {sem(dst):.5f}   ratio {std_ratio:.4f}"
    )

    kk, sk_gen = radial_sk(phi_gen)
    kk_true, sk_true = radial_sk(phi_true)
    kk_dgen, skd_gen = radial_sk(dphi)
    kk_dtrue, skd_true = radial_sk(dphi_true)
    if not (
        np.array_equal(kk, kk_true)
        and np.array_equal(kk, kk_dgen)
        and np.array_equal(kk, kk_dtrue)
    ):
        raise RuntimeError("internal radial bin mismatch")
    rel, rel_valid = _relative_deviation(sk_gen, sk_true)
    rel_k = kk[rel_valid]
    print(
        f"S(k) rel. deviation: max|.| = {np.abs(rel).max()*100:.2f}%  "
        f"(k={rel_k[np.abs(rel).argmax()]:.1f}), "
        f"mean|.| = {np.abs(rel).mean()*100:.2f}%"
    )

    print(f"\n--- two-time correlator C(0,{dm_meta['time']}) ---")
    c_gen_rep = np.empty((ncond, n_per_cond), dtype=np.float64)
    for condition in range(ncond):
        c_gen_rep[condition] = np.einsum(
            "nij,ij->n",
            phi_gen_by_cond[condition],
            phi0[condition],
            dtype=np.float64,
            optimize=True,
        ) / volume
    c_gen = c_gen_rep.mean(axis=1)
    c_gen_mc_sem = (
        c_gen_rep.std(axis=1, ddof=1) / np.sqrt(n_per_cond)
        if n_per_cond > 1
        else np.full(ncond, np.nan)
    )
    c_true = np.einsum(
        "nij,nij->n", phi0, phi_true, dtype=np.float64, optimize=True
    ) / volume
    print(
        f"C(0,{dm_meta['time']}): gen {c_gen.mean():.5f} +- {sem(c_gen):.5f}   "
        f"true {c_true.mean():.5f} +- {sem(c_true):.5f}"
    )
    correlation = _safe_corrcoef(c_gen, c_true)
    print(f"per-condition corr(C_i^gen_mean, C_i^true) = {correlation:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    provenance_lines = (
        f"samples={samples_path}\n"
        f"sample_metadata={sample_meta_path}\n"
        f"condition_source={cond_path}\n"
        f"true_target={target_path}"
    )
    atomic_savetxt(
        sk_path,
        np.column_stack([kk, sk_gen, sk_true, skd_gen, skd_true]),
        header=(
            "k  S_gen(phi)  S_true(phi)  S_gen(dphi)  S_true(dphi)\n"
            "radial_domain=0<|k|<=L/2; corner modes excluded\n"
            + provenance_lines
        ),
        overwrite=args.overwrite,
    )
    atomic_savetxt(
        scatter_path,
        np.column_stack([cond_indices, c_gen, c_gen_mc_sem, c_true]),
        header=(
            "index  C_gen_cond_mean  C_gen_MC_SEM  C_true\n" + provenance_lines
        ),
        overwrite=args.overwrite,
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": samples_path,
        "sample_metadata": sample_meta_path,
        "run_metadata": run_meta_path if run_meta else None,
        "condition_source": cond_path,
        "true_target": target_path,
        "n_conditions": int(ncond),
        "n_per_condition": int(n_per_cond),
        "condition_indices": cond_indices.tolist(),
        "L": int(lattice_size),
        "kappa": float(dm_meta["kappa"]),
        "lambda": float(dm_meta["lambda"]),
        "horizon": float(dm_meta["time"]),
        "all_dm_replicates_used": True,
        "uncertainty_level": "SEM across condition-level replicate means",
        "radial_domain": "0 < |k| <= L/2; Fourier-grid corner modes excluded",
        "max_abs_phi_relation_error": relation_error,
        "max_abs_conservation_error": cons_err,
        "c0h_condition_correlation": correlation,
        "outputs": [sk_path, scatter_path],
    }
    atomic_json_dump(provenance_path, provenance, overwrite=args.overwrite)
    print(f"\nSaved validation outputs and provenance to {out_dir}/{base}_*")


if __name__ == "__main__":
    main()
