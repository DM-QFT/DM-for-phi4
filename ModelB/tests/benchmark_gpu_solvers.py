from pathlib import Path
import csv
import functools
import gc
import os
import sys
import time
from typing import Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax
import numpy as np
import torch

jax.config.update("jax_enable_x64", False)


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "edm2"), str(ROOT), str(ROOT.parent)]

from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D
from diffusion_lightning import marginal_prob_std
from ncsnpp_dpm_solver_v3 import (
    COMPILE_MODE,
    NFE,
    dpm_solver_v3,
    prepare_solver,
    project,
)


INITIAL = (
    ROOT
    / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
    / "samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"
)
RUN = (
    ROOT
    / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
)
CHECKPOINT = RUN / "models/epoch=10000.ckpt"
EMS = RUN / "dpm_solver_v3_ems_L32_k256_n120.npz"
OUTPUT = ROOT / "results/gpu_solver_benchmark.tsv"
DEVICE = torch.device("cuda:0")
LATTICE = 64
ENSEMBLES = 512
FINAL_TIME = 100.0
# Matches the DPM-Solver-v3 single-transition action-density error at t=10.
EM_DT = 0.0004
SHARK_DT = 0.004
TRANSITIONS = 10
REPEATS = 1
KAPPA = 0.340340
LAMBDA = 1.0
M02 = 2.0 * (1.0 - 2.0 * LAMBDA) - 8.0 * KAPPA
NOISE_AMPLITUDE = np.float32(np.sqrt(2.0))
SQRT_EM_DT = np.float32(np.sqrt(EM_DT))
SQRT_SHARK_DT = np.float32(np.sqrt(SHARK_DT))
EM_STEPS = round(FINAL_TIME / EM_DT)
EM_STEPS_PER_TRANSITION = round((FINAL_TIME / TRANSITIONS) / EM_DT)
assert EM_STEPS == TRANSITIONS * EM_STEPS_PER_TRANSITION
SHARK_STEPS = round(FINAL_TIME / SHARK_DT)
SHARK_STEPS_PER_TRANSITION = round((FINAL_TIME / TRANSITIONS) / SHARK_DT)
assert SHARK_STEPS == TRANSITIONS * SHARK_STEPS_PER_TRANSITION
LINK_STRUCTURE = jax.ShapeDtypeStruct(
    (ENSEMBLES, 2, LATTICE, LATTICE), jnp.float32
)


# ShARK subtracts vmapped diffusion PyTrees.  The stencil must be static while
# the constant coefficient remains a dynamic leaf so that this difference is zero.
class ScaledStaticFunctionLinearOperator(lineax.FunctionLinearOperator):
    fn: Callable = eqx.field(static=True)
    scale: jax.Array

    def __init__(self, fn, input_structure, scale):
        super().__init__(fn, input_structure)
        self.scale = jnp.asarray(scale)

    def mv(self, vector):
        return self.scale * self.fn(vector)

    def as_matrix(self):
        return self.scale * super().as_matrix()

    def transpose(self):
        return self.scale * super().transpose()


def load_model():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    hparams = state["hyper_parameters"]
    sigma = float(hparams["sigma"])
    network = NCSNppCond2D(
        functools.partial(marginal_prob_std, sigma=sigma),
        in_channels=int(hparams.get("in_channels", 2)),
    )
    model = ConditionalDiffusionModel.load_from_checkpoint(
        CHECKPOINT,
        score_model=network,
        map_location="cpu",
        weights_only=False,
    ).to(DEVICE).eval()
    model.ema.to(DEVICE)
    return model, hparams


def timed(run):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def modelb_drift_jax(state):
    left = jnp.roll(state, 1, -1)
    right = jnp.roll(state, -1, -1)
    down = jnp.roll(state, 1, -2)
    up = jnp.roll(state, -1, -2)
    nearest = left + right + down + up
    diagonal = (
        jnp.roll(left, 1, -2)
        + jnp.roll(left, -1, -2)
        + jnp.roll(right, 1, -2)
        + jnp.roll(right, -1, -2)
    )
    second = (
        jnp.roll(state, 2, -1)
        + jnp.roll(state, -2, -1)
        + jnp.roll(state, 2, -2)
        + jnp.roll(state, -2, -2)
    )
    minus_biharmonic = -20.0 * state + 8.0 * nearest - 2.0 * diagonal - second
    laplacian = nearest - 4.0 * state
    laplacian_cube = left**3 + right**3 + down**3 + up**3 - 4.0 * state**3
    return 2.0 * KAPPA * minus_biharmonic + M02 * laplacian + 4.0 * LAMBDA * laplacian_cube


def link_divergence(links):
    return (
        links[:, 0]
        - jnp.roll(links[:, 0], 1, -1)
        + links[:, 1]
        - jnp.roll(links[:, 1], 1, -2)
    )


def em_link_divergence(links):
    return (NOISE_AMPLITUDE * SQRT_EM_DT) * link_divergence(links)


EM_LINK_OPERATOR = ScaledStaticFunctionLinearOperator(
    link_divergence,
    LINK_STRUCTURE,
    NOISE_AMPLITUDE * SQRT_EM_DT,
)
SHARK_LINK_OPERATOR = ScaledStaticFunctionLinearOperator(
    link_divergence,
    LINK_STRUCTURE,
    NOISE_AMPLITUDE * SQRT_SHARK_DT,
)


def diffrax_em_solve(state, key, step_count):
    time_grid = jnp.arange(step_count + 1, dtype=jnp.float32)
    brownian = diffrax.UnsafeBrownianPath(shape=LINK_STRUCTURE, key=key)

    def drift(time_value, fields, args):
        del time_value, args
        return np.float32(EM_DT) * modelb_drift_jax(fields)

    def diffusion(time_value, fields, args):
        del time_value, fields, args
        return EM_LINK_OPERATOR

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )
    solution = diffrax.diffeqsolve(
        terms,
        diffrax.Euler(),
        t0=time_grid[0],
        t1=time_grid[-1],
        dt0=None,
        y0=state,
        stepsize_controller=diffrax.StepTo(ts=time_grid),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=step_count + 1,
        adjoint=diffrax.ForwardMode(),
    )
    return solution.ys[0], solution.stats["num_steps"]


def benchmark_diffrax_em(initial):
    state = jax.device_put(initial)
    compiled_solve = jax.jit(diffrax_em_solve, static_argnums=2)

    validation_key = jr.key(29_999)
    validation_brownian = diffrax.UnsafeBrownianPath(
        shape=LINK_STRUCTURE, key=validation_key
    )
    increment = validation_brownian.evaluate(0.0, 1.0)
    expected = (
        state
        + np.float32(EM_DT) * modelb_drift_jax(state)
        + em_link_divergence(increment)
    )
    actual, steps = compiled_solve(state, validation_key, 1)
    jax.block_until_ready((actual, steps))
    assert actual.dtype == jnp.float32
    assert int(steps) == 1
    assert bool(jnp.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6))
    mean_drift = jnp.max(jnp.abs(jnp.mean(actual - state, axis=(-2, -1))))
    assert float(mean_drift) < 1.0e-6

    compile_start = time.perf_counter()
    warm_steps_expected = EM_STEPS_PER_TRANSITION
    warm, warm_steps = compiled_solve(
        state, jr.key(30_000), warm_steps_expected
    )
    jax.block_until_ready((warm, warm_steps))
    compile_seconds = time.perf_counter() - compile_start
    assert int(warm_steps) == warm_steps_expected
    assert bool(jnp.isfinite(warm).all())
    print(
        f"Diffrax EM compile and warmup  {compile_seconds} s",
        flush=True,
    )

    times = []
    result = warm
    for repeat in range(REPEATS):
        start = time.perf_counter()
        result = state
        step_counts = []
        for transition in range(TRANSITIONS):
            result, steps = compiled_solve(
                result,
                jr.key(30_001 + repeat * TRANSITIONS + transition),
                EM_STEPS_PER_TRANSITION,
            )
            step_counts.append(steps)
        jax.block_until_ready((result, step_counts))
        seconds = time.perf_counter() - start
        assert sum(int(steps) for steps in step_counts) == EM_STEPS
        times.append(seconds)
        print(f"Diffrax EM repeat {repeat + 1}  {seconds} s", flush=True)
    assert bool(jnp.isfinite(result).all())
    return times


def diffrax_shark_solve(state, key, step_count):
    time_grid = jnp.arange(step_count + 1, dtype=jnp.float32)
    brownian = diffrax.UnsafeBrownianPath(
        shape=LINK_STRUCTURE,
        key=key,
        levy_area=diffrax.SpaceTimeLevyArea,
    )

    def drift(time_value, fields, args):
        del time_value, args
        return np.float32(SHARK_DT) * modelb_drift_jax(fields)

    def diffusion(time_value, fields, args):
        del time_value, fields, args
        return SHARK_LINK_OPERATOR

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )
    solution = diffrax.diffeqsolve(
        terms,
        diffrax.ShARK(),
        t0=time_grid[0],
        t1=time_grid[-1],
        dt0=None,
        y0=state,
        stepsize_controller=diffrax.StepTo(ts=time_grid),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=step_count + 1,
        adjoint=diffrax.ForwardMode(),
    )
    return solution.ys[0], solution.stats["num_steps"]


def benchmark_diffrax_shark(initial):
    state = jax.device_put(initial)
    compiled_solve = jax.jit(diffrax_shark_solve, static_argnums=2)

    one_step, one_step_count = compiled_solve(state, jr.key(31_999), 1)
    jax.block_until_ready((one_step, one_step_count))
    assert one_step.dtype == jnp.float32
    assert int(one_step_count) == 1
    assert bool(jnp.isfinite(one_step).all())
    mean_drift = jnp.max(jnp.abs(jnp.mean(one_step - state, axis=(-2, -1))))
    assert float(mean_drift) < 1.0e-6

    compile_start = time.perf_counter()
    warm, warm_steps = compiled_solve(
        state, jr.key(32_000), SHARK_STEPS_PER_TRANSITION
    )
    jax.block_until_ready((warm, warm_steps))
    compile_seconds = time.perf_counter() - compile_start
    assert int(warm_steps) == SHARK_STEPS_PER_TRANSITION
    assert bool(jnp.isfinite(warm).all())
    print(
        f"Diffrax ShARK compile and t=10 warmup  {compile_seconds} s",
        flush=True,
    )

    times = []
    result = warm
    for repeat in range(REPEATS):
        start = time.perf_counter()
        result = state
        step_counts = []
        for transition in range(TRANSITIONS):
            result, steps = compiled_solve(
                result,
                jr.key(32_001 + repeat * TRANSITIONS + transition),
                SHARK_STEPS_PER_TRANSITION,
            )
            step_counts.append(steps)
        jax.block_until_ready((result, step_counts))
        seconds = time.perf_counter() - start
        assert sum(int(steps) for steps in step_counts) == SHARK_STEPS
        times.append(seconds)
        print(f"Diffrax ShARK repeat {repeat + 1}  {seconds} s", flush=True)
    assert bool(jnp.isfinite(result).all())
    return times


def benchmark_dpm(initial, compile_mode):
    model, hparams = load_model()
    solver = prepare_solver(EMS)
    norm_res = float(hparams["norm_max"])
    norm_cond = float(hparams["cond_norm_max"])
    diffusion_sigma = float(hparams["sigma"])
    model.score_model = torch.compile(
        model.score_model.to(memory_format=torch.channels_last),
        mode=compile_mode,
        dynamic=False,
    )

    def transition(fields):
        condition = (fields[:, None] / norm_cond).to(
            memory_format=torch.channels_last
        )
        noise = project(torch.randn_like(condition))
        residual = dpm_solver_v3(
            model, condition, noise, solver, diffusion_sigma
        )[:, 0]
        return fields + norm_res * residual

    with model.ema.average_parameters():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            compile_seconds, warm = timed(lambda: transition(initial.clone()))
            warm = transition(warm)
            torch.cuda.synchronize()
            print(
                f"DPM-Solver-v3 {compile_mode} compile and first transition  "
                f"{compile_seconds} s",
                flush=True,
            )

            times = []
            result = warm
            for repeat in range(REPEATS):
                torch.manual_seed(40_000 + repeat)

                def run():
                    fields = initial.clone()
                    for _ in range(TRANSITIONS):
                        fields = transition(fields)
                    return fields

                seconds, result = timed(run)
                times.append(seconds)
                print(
                    f"DPM-Solver-v3 {compile_mode} repeat {repeat + 1}  "
                    f"{seconds} s",
                    flush=True,
                )
            assert torch.isfinite(result).all()
    return times, compile_seconds


def append_results(rows):
    if not OUTPUT.exists():
        raise FileNotFoundError(
            f"Run tests/benchmark_gpu_solvers.jl before this script. Missing {OUTPUT}"
        )
    with OUTPUT.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames
        julia_rows = [row for row in reader if row["backend"].startswith("Julia")]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in julia_rows:
            writer.writerow(row[name] for name in header)
        for row in rows:
            if len(row) != len(header):
                raise ValueError("benchmark row does not match the TSV header")
            writer.writerow(row)


def result_row(
    method,
    backend,
    precision,
    dt,
    steps,
    drift_nfe,
    transitions,
    network_nfe,
    mode,
    times,
):
    return (
        method,
        backend,
        precision,
        LATTICE,
        ENSEMBLES,
        FINAL_TIME,
        dt,
        steps,
        drift_nfe,
        transitions,
        network_nfe,
        mode,
        times[0],
    )


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    with np.load(INITIAL, allow_pickle=False) as archive:
        initial_numpy = np.asarray(archive["phi_gen"], dtype=np.float32)
    initial = torch.from_numpy(initial_numpy).to(DEVICE)
    assert tuple(initial.shape) == (ENSEMBLES, LATTICE, LATTICE)

    dpm_max_times, _ = benchmark_dpm(initial, COMPILE_MODE)
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    em_times = benchmark_diffrax_em(initial_numpy)
    shark_times = benchmark_diffrax_shark(initial_numpy)
    append_results((
        result_row(
            "ShARK",
            f"Diffrax {diffrax.__version__}, JAX {jax.__version__}",
            "Float32",
            SHARK_DT,
            SHARK_STEPS,
            2 * SHARK_STEPS,
            TRANSITIONS,
            0,
            "jax.jit XLA StepTo",
            shark_times,
        ),
        result_row(
            "EM",
            f"Diffrax {diffrax.__version__}, JAX {jax.__version__}",
            "Float32",
            EM_DT,
            EM_STEPS,
            EM_STEPS,
            TRANSITIONS,
            0,
            "jax.jit XLA StepTo",
            em_times,
        ),
        result_row(
            "DPM-Solver-v3",
            "PyTorch",
            "BF16 network, Float32 and Float64 solver",
            "",
            0,
            0,
            TRANSITIONS,
            TRANSITIONS * NFE,
            COMPILE_MODE,
            dpm_max_times,
        ),
    ))
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
