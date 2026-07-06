"""Format the DM+MALA rows for tab:multiL_HMC_cumulants (16-col 3D layout)."""
import csv
import importlib.util
import sys
from pathlib import Path

CMP = Path(__file__).resolve().parents[1] / "physical observable" / \
    "compare_L64_multiL_HMC_cumulants.py"
spec = importlib.util.spec_from_file_location("cmp_pipeline", CMP)
cmp_mod = importlib.util.module_from_spec(spec)
sys.modules["cmp_pipeline"] = cmp_mod
spec.loader.exec_module(cmp_mod)
ve = cmp_mod._val_err

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "results/L64_multiL_HMC_cumulants_raw_z2_em_linear_steps2000_ep10000.csv")

COLS = [("abs_M", "d_abs_M"), ("chi", "d_chi"), ("S_V", "d_S_V"),
        ("U4", "d_U4"), ("k2", "d_k2"), ("k4", "d_k4"), ("k6", "d_k6"),
        ("k8", "d_k8"), ("site_abs_phi", "d_site_abs_phi"),
        ("site_k2", "d_site_k2"), ("site_k4", "d_site_k4"),
        ("site_k6", "d_site_k6"), ("site_k8", "d_site_k8"),
        ("site_U4_site", "d_site_U4_site")]

rows = list(csv.DictReader(open(CSV)))
LABEL = r"DM\,$\{4,8,16,32\}$+MALA"
for kappa in ("0.18", "0.1923", "0.2"):
    r = next((x for x in rows if x["kappa"] == kappa
              and x["source"] == "DM_multiL4-8-16-32_MALA"), None)
    if r is None:
        print(f"% missing {kappa}")
        continue
    cells = [ve(float(r[v]), float(r[e])) for v, e in COLS]
    print(f"%% kappa={kappa}  N={r['N']}")
    print(f" & {LABEL} & " + " & ".join(cells) + r" \\")
    print()
