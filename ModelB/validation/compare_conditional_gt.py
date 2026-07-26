"""Compare DM conditional samples with Julia conditional ground truth.

For each fixed condition ``phi0``, this compares ``n_per_cond`` DM samples
with the same number of independent Model B reruns.  Both direct-script and
module execution are supported::

    python validation/compare_conditional_gt.py --dm samples.npz --gt truth.jld2
    python -m validation.compare_conditional_gt --dm samples.npz --gt truth.jld2
"""

import argparse
from datetime import datetime, timezone
import os

import h5py
import numpy as np

try:  # Package/module execution.
    from .analyze_cond_samples import (
        _relative_deviation,
        _require_metadata,
        _validate_dm_json,
        _validate_field_array,
        atomic_json_dump,
        atomic_savetxt,
        check_metadata_match,
        check_source_match,
        ensure_outputs_writable,
        hdf_metadata,
        load_dm_archive,
        load_json_object,
        radial_sk,
        read_indexed_cfgs,
        validate_indices,
    )
except ImportError:  # ``python validation/compare_conditional_gt.py``.
    from analyze_cond_samples import (  # type: ignore
        _relative_deviation,
        _require_metadata,
        _validate_dm_json,
        _validate_field_array,
        atomic_json_dump,
        atomic_savetxt,
        check_metadata_match,
        check_source_match,
        ensure_outputs_writable,
        hdf_metadata,
        load_dm_archive,
        load_json_object,
        radial_sk,
        read_indexed_cfgs,
        validate_indices,
    )


def _read_gt(path):
    with h5py.File(path, "r") as handle:
        required = ("cfgs", "cond_indices_0based", "n_per_cond")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"{path} is missing datasets: {', '.join(missing)}")
        phi_gt = _validate_field_array(handle["cfgs"][...], f"{path}:cfgs")
        idx_gt = validate_indices(
            handle["cond_indices_0based"][...], "cond_indices_0based"
        )
        n_value = np.asarray(handle["n_per_cond"][()])
        if n_value.shape != ():
            raise ValueError(f"{path}: n_per_cond must be scalar")
        try:
            n_gt = int(n_value.item())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid n_per_cond") from exc
        if n_gt <= 1 or n_gt != n_value.item() or n_gt % 2:
            raise ValueError(
                f"{path}: n_per_cond must be an even integer greater than one "
                "for the split-half noise floor"
            )
        metadata = hdf_metadata(handle)
    expected = idx_gt.size * n_gt
    if phi_gt.shape[0] != expected:
        raise ValueError(
            f"{path}: cfgs has {phi_gt.shape[0]} members, expected "
            f"{idx_gt.size} x {n_gt} = {expected}"
        )
    return phi_gt, idx_gt, n_gt, metadata


def _pooled_difference_moments(samples, reference, batch_size=32):
    """Compute pooled central moments without materializing all differences."""
    if samples.ndim != 4 or reference.ndim != 3:
        raise ValueError("unexpected conditional field shapes")
    if samples.shape[0] != reference.shape[0] or samples.shape[2:] != reference.shape[1:]:
        raise ValueError("sample/reference shapes are incompatible")
    raw = np.zeros(4, dtype=np.float64)
    count = 0
    for condition in range(samples.shape[0]):
        for start in range(0, samples.shape[1], batch_size):
            difference = (
                np.asarray(
                    samples[condition, start:start + batch_size], dtype=np.float64
                )
                - reference[condition]
            )
            square = difference * difference
            raw[0] += difference.sum(dtype=np.float64)
            raw[1] += square.sum(dtype=np.float64)
            raw[2] += (square * difference).sum(dtype=np.float64)
            raw[3] += (square * square).sum(dtype=np.float64)
            count += difference.size
    raw /= count
    mean = raw[0]
    variance = raw[1] - mean * mean
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("pooled dphi variance is not positive and finite")
    third = raw[2] - 3 * mean * raw[1] + 2 * mean**3
    fourth = raw[3] - 4 * mean * raw[2] + 6 * mean**2 * raw[1] - 3 * mean**4
    std = np.sqrt(variance)
    return float(mean), float(std), float(third / std**3), float(fourth / variance**2 - 3)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dm", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace validation outputs with the same name",
    )
    args = parser.parse_args(argv)

    dm_path = os.path.realpath(os.path.abspath(args.dm))
    gt_path = os.path.realpath(os.path.abspath(args.gt))
    archive = load_dm_archive(dm_path, require_dphi=False)
    phi_dm = archive["phi_gen"]
    idx_dm = archive["cond_indices"]
    n_dm = archive["n_per_cond"]
    lattice_size = phi_dm.shape[-1]

    dm_meta_path = os.path.splitext(dm_path)[0] + ".json"
    dm_json = load_json_object(dm_meta_path)
    dm_meta = _validate_dm_json(dm_json, dm_meta_path, archive, lattice_size)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(dm_path)), "validation")
    base = os.path.splitext(os.path.basename(dm_path))[0]
    sk_path = os.path.join(out_dir, base + "_condsk.dat")
    provenance_path = os.path.join(out_dir, base + "_condcompare.json")
    # Reject a collision before loading the (potentially large) GT ensemble.
    ensure_outputs_writable((sk_path, provenance_path), args.overwrite)

    phi_gt, idx_gt, n_gt, gt_meta = _read_gt(gt_path)
    _require_metadata(
        gt_meta, ("L", "kappa", "lambda", "time", "source_path"), gt_path
    )
    check_metadata_match(
        dm_meta,
        dm_meta_path,
        gt_meta,
        gt_path,
        ("L", "kappa", "lambda", "time"),
    )
    cond_path = check_source_match(
        dm_meta["source_path"],
        dm_meta_path,
        gt_meta["source_path"],
        gt_path,
    )

    if not np.array_equal(idx_dm, idx_gt):
        raise ValueError(
            "condition-index mismatch:\n"
            f"  DM: {idx_dm.tolist()}\n  GT: {idx_gt.tolist()}"
        )
    if n_dm != n_gt:
        raise ValueError(f"n_per_cond mismatch: DM={n_dm}, GT={n_gt}")
    if phi_gt.shape[-1] != lattice_size:
        raise ValueError(
            f"lattice-size mismatch: DM={lattice_size}, GT={phi_gt.shape[-1]}"
        )

    ncond = idx_gt.size
    phi_dm = phi_dm.reshape(ncond, n_dm, lattice_size, lattice_size)
    phi_gt = phi_gt.reshape(ncond, n_gt, lattice_size, lattice_size)
    phi0, cond_hdf_meta = read_indexed_cfgs(cond_path, idx_gt)
    check_metadata_match(
        dm_meta,
        dm_meta_path,
        cond_hdf_meta,
        cond_path,
        ("L", "kappa", "lambda"),
    )
    if phi0.shape != (ncond, lattice_size, lattice_size):
        raise ValueError(
            f"condition shape {phi0.shape} is incompatible with DM/GT fields"
        )

    print("=== conditional ground-truth comparison ===")
    print(f"{ncond} conditions x {n_dm} samples, L={lattice_size}")
    print(f"DM samples:       {dm_path}")
    print(f"ground truth:     {gt_path}")
    print(f"condition source: {cond_path}")
    print(
        f"{'idx':>6} {'relerr(mean)':>13} {'noisefloor':>11} "
        f"{'std ratio':>10} {'C(0,h)_DM':>10} {'C(0,h)_GT':>10}"
    )

    sk_dm_all, sk_gt_all = [], []
    rows = []
    kk = None
    volume = lattice_size * lattice_size
    for condition in range(ncond):
        m_dm = phi_dm[condition].mean(axis=0, dtype=np.float64)
        m_gt = phi_gt[condition].mean(axis=0, dtype=np.float64)
        drift = float(np.linalg.norm(m_gt - phi0[condition]))
        if not np.isfinite(drift) or drift <= 0:
            raise ValueError(
                f"condition {idx_gt[condition]} has zero/non-finite GT mean drift"
            )
        relerr = float(np.linalg.norm(m_dm - m_gt) / drift)

        # Two equal-size half means are farther apart by sqrt(2) than the
        # expected discrepancy between one full-size Monte Carlo mean and its
        # population mean.  The historical diagnostic omitted this factor.
        half = n_gt // 2
        half_difference = np.linalg.norm(
            phi_gt[condition, :half].mean(axis=0, dtype=np.float64)
            - phi_gt[condition, half:].mean(axis=0, dtype=np.float64)
        )
        floor = float(half_difference / (np.sqrt(2.0) * drift))

        s_dm = float(
            phi_dm[condition].std(axis=0, dtype=np.float64).mean()
        )
        s_gt = float(
            phi_gt[condition].std(axis=0, dtype=np.float64).mean()
        )
        if not np.isfinite(s_gt) or s_gt <= 0:
            raise ValueError(
                f"condition {idx_gt[condition]} has zero/non-finite GT std"
            )
        std_ratio = s_dm / s_gt
        c_dm = float(
            np.einsum(
                "nij,ij->",
                phi_dm[condition],
                phi0[condition],
                dtype=np.float64,
                optimize=True,
            )
            / (n_dm * volume)
        )
        c_gt = float(
            np.einsum(
                "nij,ij->",
                phi_gt[condition],
                phi0[condition],
                dtype=np.float64,
                optimize=True,
            )
            / (n_gt * volume)
        )
        print(
            f"{idx_gt[condition]:>6} {relerr:>13.4f} {floor:>11.4f} "
            f"{std_ratio:>10.4f} {c_dm:>10.5f} {c_gt:>10.5f}"
        )
        rows.append(
            {
                "condition_index": int(idx_gt[condition]),
                "relative_mean_map_error": relerr,
                "gt_split_half_noise_floor": floor,
                "mean_pixel_std_ratio": std_ratio,
                "c0h_dm": c_dm,
                "c0h_gt": c_gt,
            }
        )

        current_k, sk_dm = radial_sk(phi_dm[condition] - m_dm)
        gt_k, sk_gt = radial_sk(phi_gt[condition] - m_gt)
        if not np.array_equal(current_k, gt_k):
            raise RuntimeError("internal radial bin mismatch")
        if kk is not None and not np.array_equal(kk, current_k):
            raise RuntimeError("radial bins differ between conditions")
        kk = current_k
        sk_dm_all.append(sk_dm)
        sk_gt_all.append(sk_gt)

    sk_dm_mean = np.mean(sk_dm_all, axis=0)
    sk_gt_mean = np.mean(sk_gt_all, axis=0)
    rel, valid = _relative_deviation(sk_dm_mean, sk_gt_mean)
    valid_k = kk[valid]
    print(
        f"\nconditional fluctuation S(k): max|rel| = {np.abs(rel).max()*100:.2f}% "
        f"(k={valid_k[np.abs(rel).argmax()]:.1f}), "
        f"mean|rel| = {np.abs(rel).mean()*100:.2f}%"
    )

    pooled = {}
    for name, fields in (("DM", phi_dm), ("GT", phi_gt)):
        mean, std, skew, kurtosis = _pooled_difference_moments(fields, phi0)
        pooled[name] = {
            "mean": mean,
            "std": std,
            "skew": skew,
            "excess_kurtosis": kurtosis,
        }
        print(
            f"dphi pooled [{name}]: mean {mean:+.5f}  std {std:.5f}  "
            f"skew {skew:+.4f}  ex.kurt {kurtosis:+.4f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    provenance_lines = (
        f"dm_samples={dm_path}\n"
        f"dm_metadata={dm_meta_path}\n"
        f"ground_truth={gt_path}\n"
        f"condition_source={cond_path}"
    )
    atomic_savetxt(
        sk_path,
        np.column_stack([kk, sk_dm_mean, sk_gt_mean]),
        header=(
            "k  S_cond_DM  S_cond_GT\n"
            "radial_domain=0<|k|<=L/2; corner modes excluded\n"
            + provenance_lines
        ),
        overwrite=args.overwrite,
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dm_samples": dm_path,
        "dm_metadata": dm_meta_path,
        "ground_truth": gt_path,
        "condition_source": cond_path,
        "n_conditions": int(ncond),
        "n_per_condition": int(n_dm),
        "condition_indices": idx_gt.tolist(),
        "L": int(lattice_size),
        "kappa": float(dm_meta["kappa"]),
        "lambda": float(dm_meta["lambda"]),
        "horizon": float(dm_meta["time"]),
        "gt_split_half_noise_floor_correction": "difference / sqrt(2)",
        "radial_domain": "0 < |k| <= L/2; Fourier-grid corner modes excluded",
        "conditions": rows,
        "pooled_dphi": pooled,
        "outputs": [sk_path],
    }
    atomic_json_dump(provenance_path, provenance, overwrite=args.overwrite)
    print(f"Saved conditional S(k) and provenance to {out_dir}/{base}_cond*")


if __name__ == "__main__":
    main()
