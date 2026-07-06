"""
Read all acceptance_multiL_*.csv summary blocks under
results/calibration/ and emit a tidy table.
"""

import re
from pathlib import Path
import csv
import sys

CALIB = Path(__file__).resolve().parent.parent / "results/calibration"


def parse_run(name):
    """Return (k, Lset, sigma_tag) parsed from filename or None."""
    m = re.match(r"acceptance_multiL_phi4_3d_Lmulti([\d-]+)_k([\d.]+)_l0\.9_ncsnpp(?:_(\w+))?\.csv", name)
    if not m:
        return None
    Lset = m.group(1)
    k = float(m.group(2))
    suf = m.group(3) or ""
    return k, Lset, suf


def read_summary(path):
    """Return {L: (mean, sd, n_ep)} from summary block of csv."""
    out = {}
    in_summary = False
    with open(path) as f:
        for line in f:
            if line.startswith("# summary"):
                in_summary = True
                continue
            if in_summary:
                if line.startswith("L,"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    L = int(parts[0])
                    out[L] = (float(parts[1]), float(parts[2]), int(parts[3]))
    return out


def main():
    rows = []
    for csv_path in sorted(CALIB.glob("acceptance_multiL_*.csv")):
        meta = parse_run(csv_path.name)
        if not meta:
            continue
        k, Lset, suf = meta
        summary = read_summary(csv_path)
        rows.append((k, Lset, suf, summary))

    # Sort by (k, Lset_size)
    def sort_key(row):
        k, Lset, suf, _ = row
        n_train = len(Lset.split("-"))
        return (k, n_train, suf)
    rows.sort(key=sort_key)

    print(f"{'kappa':>6}  {'Lset':>10}  {'sigma':>8}  "
          f"{'L=4':>10}  {'L=8':>10}  {'L=16':>10}  {'L=32':>10}  {'L=64':>10}")
    print("-" * 95)
    for k, Lset, suf, summary in rows:
        sigma_str = suf or "default"
        cells = []
        for L in [4, 8, 16, 32, 64]:
            if L in summary:
                m, s, n = summary[L]
                cells.append(f"{m:.3f}({int(round(s*1000)):d})")
            else:
                cells.append("-")
        print(f"{k:>6.4f}  {Lset:>10}  {sigma_str:>8}  "
              f"{cells[0]:>10}  {cells[1]:>10}  {cells[2]:>10}  "
              f"{cells[3]:>10}  {cells[4]:>10}")

    # Also dump a flat csv
    out_path = CALIB / "acceptance_multiL_summary.csv"
    with open(out_path, "w") as f:
        f.write("k,Lset,sigma_tag,L_eval,acc_mean,acc_sd_over_ep,n_ep\n")
        for k, Lset, suf, summary in rows:
            for L, (m, s, n) in sorted(summary.items()):
                f.write(f"{k},{Lset},{suf},{L},{m:.6f},{s:.6f},{n}\n")
    print(f"\nFlat csv: {out_path}")


if __name__ == "__main__":
    main()
