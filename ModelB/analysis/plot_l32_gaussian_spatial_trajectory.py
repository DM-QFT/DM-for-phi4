"""Plot one Gaussian-initial spatial trajectory for Model B and iterated DM."""

from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CHAIN = ROOT / (
    "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_all_data/samples/"
    "gaussian_chain_ep2257_N5120_seed20260721"
)
OUTPUT = CHAIN / "spatial_trajectories"
TIMES = range(0, 101, 10)


def dm_path(time):
    if time == 0:
        return CHAIN / "gaussian_ep2257_t0_N5120_seed20260721.npz"
    return CHAIN / (
        f"gaussian_chain_ep2257_t{time}_N5120_steps2000_log_"
        f"seed{20260721 + time // 10}.npz"
    )


def physical_path(time):
    return CHAIN / "physical_reference" / (
        f"modelB_L32_Ne5120_gaussian_t{time}_chain_"
        "dt0.002_seed20260801.jld2"
    )


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dm = []
    physical = []
    for time in TIMES:
        with np.load(dm_path(time), allow_pickle=False) as archive:
            field = np.asarray(archive["phi_gen"][index], dtype=np.float32)
        dm.append(field)
        if time == 0:
            physical.append(field)
        else:
            with h5py.File(physical_path(time), "r") as handle:
                physical.append(np.asarray(handle["cfgs"][index], dtype=np.float32))

    dm = np.stack(dm)
    physical = np.stack(physical)
    limit = float(max(np.max(np.abs(dm)), np.max(np.abs(physical))))
    norm = Normalize(-limit, limit)

    figure, axes = plt.subplots(
        2, len(dm), figsize=(24, 5.2), constrained_layout=True,
    )
    image = None
    for column, time in enumerate(TIMES):
        for row, fields in enumerate((physical, dm)):
            axis = axes[row, column]
            image = axis.imshow(
                fields[column], origin="lower", cmap="RdBu_r", norm=norm,
                interpolation="nearest", rasterized=True,
            )
            axis.set_xticks(())
            axis.set_yticks(())
        axes[0, column].set_title(fr"$t={time}$")
    axes[0, 0].set_ylabel("Model B", fontsize=12)
    axes[1, 0].set_ylabel("Iterated DM", fontsize=12)
    figure.colorbar(
        image, ax=axes, shrink=0.92, pad=0.008,
        label=fr"$\phi(x,y)$, shared range $[-{limit:.2f},{limit:.2f}]$",
    )
    figure.suptitle(f"L=32 Gaussian trajectory index {index}", fontsize=14)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / f"gaussian_modelb_dm_spatial_trajectory_index{index:04d}"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(stem)


if __name__ == "__main__":
    main()
