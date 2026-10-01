"""Compare rectangular Wilson loops in saved DM and heatbath ensembles."""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import ive

from sample_su2_em import wilson
from train_su2_dm import load_cfgs

ROOT = Path(__file__).resolve().parents[2]
DM_FILE = ROOT / "results/su2_L16_beta2.0_lcnn_hk_c16l3_20261001/samples_epoch0100_steps2000.h5"
HB_FILE = ROOT / "trainningdata/cfgs_su2_beta=2.0_16^2.h5"
RT = sorted(((r, t) for r in range(1, 5) for t in range(1, 5)), key=lambda rt: (rt[0] * rt[1], *rt))
BATCH = 256
HB_BLOCK = 50  # Consecutive saved configurations per block. Check stability when changing ensembles.
OUT = DM_FILE.with_name(DM_FILE.stem + "_wilson")


@torch.no_grad()
def measure(path):
    # Reuse the sampler's loop definition. Average sites before estimating ensemble errors.
    U = load_cfgs(path=path, dtype=torch.float64)
    return torch.cat([
        torch.stack([wilson(batch, r, t).mean((1, 2)) for r, t in RT], dim=1)
        for batch in U.split(BATCH)
    ]).cpu().numpy()


def mean_error(values, block):
    n_blocks = len(values) // block
    if n_blocks < 2:
        raise ValueError("At least two complete blocks are needed to estimate the error")
    # Discard only the incomplete last block. The mean and error use identical configurations.
    means = values[:n_blocks * block].reshape(n_blocks, block, -1).mean(axis=1)
    return means.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def main():
    with h5py.File(DM_FILE) as dm, h5py.File(HB_FILE) as hb:
        beta, L = float(dm.attrs["beta"]), int(dm.attrs["L"])
        if beta != float(hb.attrs["beta"]) or L != int(hb.attrs["L"]):
            raise ValueError("DM and heatbath must have the same beta and lattice size")
    if any(not (0 < r < L and 0 < t < L) for r, t in RT):
        raise ValueError("Loop sides must be positive and smaller than L")

    # Infinite-volume result for simple contractible loops. At beta=2 and L=16,
    # finite-volume corrections for these loops are negligible. See arXiv 1403.1770.
    area = np.prod(RT, axis=1)
    exact = (ive(2, beta) / ive(1, beta)) ** area
    dm_values, hb_values = measure(DM_FILE), measure(HB_FILE)
    dm_mean, dm_se = mean_error(dm_values, 1)
    hb_mean, hb_se = mean_error(hb_values, HB_BLOCK)
    n_dm, n_hb = len(dm_values), len(hb_values) // HB_BLOCK * HB_BLOCK
    print(f"beta {beta:g}   L {L}   DM {n_dm}   heatbath {n_hb}   heatbath block {HB_BLOCK}")
    print(" R  T        analytic              DM +/- SE                 heatbath +/- SE")
    for i, (r, t) in enumerate(RT):
        print(f"{r:2d} {t:2d}   {exact[i]:.8e}   {dm_mean[i]: .8e} +/- {dm_se[i]:.3e}   "
              f"{hb_mean[i]: .8e} +/- {hb_se[i]:.3e}")
    np.savetxt(OUT.with_suffix(".csv"), np.column_stack([
        np.asarray(RT), area, exact, dm_mean, dm_se, hb_mean, hb_se,
    ]), delimiter=",", fmt=["%d"] * 3 + ["%.10e"] * 5,
        header="R,T,area,analytic_infinite_volume,dm_mean,dm_se,heatbath_mean,heatbath_se", comments="")

    plt.style.use(ROOT.parent / "plots" / "cjp.mplstyles")
    fig, (ax, residual) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.4),
                                      gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(RT))
    for mean, se, label in [
        (dm_mean, dm_se, f"DM  $N={n_dm}$"),
        (hb_mean, hb_se, f"Heatbath  $N={n_hb}$"),
    ]:
        ax.errorbar(x, mean, yerr=se, linestyle="none", label=label)
        residual.errorbar(x, mean - exact, yerr=se, linestyle="none")
    ax.plot(x, exact, label="Analytic, infinite volume", zorder=1)
    # Keep nonpositive estimates and error bars crossing zero visible.
    ax.set_yscale("symlog", linthresh=2e-3, linscale=0.7)
    ax.set_ylabel(r"$\langle W(R,T)\rangle$")
    ax.set_title(rf"$SU(2)\quad \beta={beta:g}\quad L={L}$" + "     Error bars show one standard error")
    ax.legend(loc="upper right")
    residual.axhline(0, color="black")
    residual.set_ylabel(r"$\langle W\rangle-W_{\mathrm{analytic}}$")
    residual.set_xlabel(r"Loop size $R\times T$")
    residual.set_xticks(x, [rf"${r}\times{t}$" for r, t in RT])
    residual.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a in (ax, residual):
        a.set_xlim(-0.6, len(RT) - 0.4)
        a.tick_params(axis="x", which="minor", bottom=False, top=False)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {OUT}.pdf, .png, .csv")


if __name__ == "__main__":
    main()
