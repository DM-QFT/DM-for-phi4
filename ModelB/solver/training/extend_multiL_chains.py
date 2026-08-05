"""Extend the mixed-L ep10000 kernels on 512 HMC trajectories."""

from pathlib import Path

from training import sample_modelB_chain as chain


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/multiL_ep10000_extrapolation"
T10 = ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/models/epoch=10000.ckpt"
T05 = ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip/models/epoch=10000.ckpt"
L32_INITIAL = OUT / "hmc_L32_N512_t0.npz"
L64_INITIAL = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"
L64_T200 = ROOT / "runs/multiL_ep10000_reverse_analysis/samples/t10/mixed_L64_chain_t200/chain_ep10000_t200_via_iterated_t10dm512_steps2000_log_seed2026080820.npz"


def run(checkpoint, initial, output, times, seeds):
    chain.TIMES = times
    chain.SEEDS = seeds
    chain.BATCH = 512
    chain.NUM_STEPS = 2000
    chain.SCHEDULE = "log"
    chain.main([
        "--checkpoint", str(checkpoint),
        "--input", str(initial),
        "--output_dir", str(output),
    ])


run(T10, L32_INITIAL, OUT / "t10/L32", range(10, 1001, 10), range(2026081101, 2026081201))
run(T10, L64_T200, OUT / "t10/L64_t210_t1000", range(210, 1001, 10), range(2026081201, 2026081281))
run(T05, L32_INITIAL, OUT / "t0p5/L32", [0.5 * i for i in range(1, 101)], range(2026082101, 2026082201))
run(T05, L64_INITIAL, OUT / "t0p5/L64", [0.5 * i for i in range(1, 101)], range(2026082201, 2026082301))
