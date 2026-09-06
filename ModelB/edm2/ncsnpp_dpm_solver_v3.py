"""Generate the two formal L64 chains with DPM-Solver-v3 at 32 NFE."""

from pathlib import Path
import math
import sys
import time

import h5py
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT), str(ROOT.parent)]

from diffusion_lightning import marginal_prob_std
from ncsnpp_dpmpp2m import DEVICE, INITIAL, load_model, project


SPECS = (
    (
        ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip",
        ROOT / "results/multiL/t0p5_L64.h5",
        (
            ROOT / "trainingdata/modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2",
            ROOT / "trainingdata/modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0p5_dt0.002_seed2026080102.jld2",
        ),
        "hmc_L64_N512_t50_dpm_solver_v3_nfe32.h5",
        2026081202,
    ),
    (
        ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip",
        ROOT / "results/multiL/t10_L64.h5",
        (
            ROOT / "trainingdata/modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2",
            ROOT / "trainingdata/modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080102.jld2",
        ),
        "hmc_L64_N512_t1000_dpm_solver_v3_nfe32.h5",
        2026081201,
    ),
)
NFE = 32
STATISTICS_STEPS = 120
EMS_SAMPLES = 256
EMS_BATCH = 64
SAMPLE_BATCH = 512
EPS = 1e-5
COMPILE_MODE = "max-autotune"

# Exact indexes produced by the official implementation for 120 EMS intervals
# and 32 uniform-logSNR steps in this VE noise range.
SAMPLE_INDEX = np.array([
    0, 4, 7, 11, 15, 19, 22, 26, 30, 34, 38,
    41, 45, 49, 53, 56, 60, 64, 68, 71, 75, 79,
    82, 86, 90, 94, 98, 101, 105, 109, 112, 116, 120,
])


def smooth_derivative(values, gap):
    derivative = np.empty_like(values)
    derivative[0] = (values[1] - values[0]) / gap
    derivative[-1] = (values[-1] - values[-2]) / gap
    derivative[1:-1] = (values[2:] - values[:-2]) / (2 * gap)
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - 5)
        stop = min(len(values), index + 6)
        smoothed[index] = derivative[start:stop].mean()
    return smoothed


def ems_pairs(paths, hparams, seed):
    condition_path, target_path = paths
    with h5py.File(condition_path, "r") as handle:
        initial = np.asarray(handle["cfgs"], dtype=np.float32)
    with h5py.File(target_path, "r") as handle:
        target = np.asarray(handle["cfgs"], dtype=np.float32)
    index = np.random.default_rng(seed).choice(
        len(initial), EMS_SAMPLES, replace=False
    )
    condition = torch.from_numpy(
        initial[index, None] / float(hparams["cond_norm_max"])
    ).to(DEVICE, memory_format=torch.channels_last)
    residual = torch.from_numpy(
        (target[index] - initial[index])[:, None]
        / float(hparams["norm_max"])
    ).to(DEVICE, memory_format=torch.channels_last)
    return condition, project(residual), index


def compute_ems(model, hparams, paths, output, seed):
    if output.exists():
        print(f"loaded {output}", flush=True)
        return

    condition, clean, sample_index = ems_pairs(paths, hparams, seed)
    diffusion_sigma = float(hparams["sigma"])
    log_base = np.log(diffusion_sigma)
    sigma_max = marginal_prob_std(
        torch.tensor(1.0), diffusion_sigma
    ).item()
    sigma_min = marginal_prob_std(
        torch.tensor(EPS), diffusion_sigma
    ).item()
    lambdas = np.linspace(
        -np.log(sigma_max), -np.log(sigma_min), STATISTICS_STEPS + 1
    )
    sigmas = np.exp(-lambdas)
    gap = lambdas[1] - lambdas[0]

    def epsilon(state, sigma, cond):
        train_time = (
            torch.log1p(2 * log_base * sigma.square()) / (2 * log_base)
        )
        score = model(
            torch.cat((state, cond), dim=1),
            train_time.expand(len(state)),
        ).float()
        return project(-sigma * score)

    l_values = np.empty(len(sigmas), dtype=np.float64)
    start_all = time.perf_counter()
    with torch.no_grad():
        for point, sigma_value in enumerate(sigmas):
            torch.manual_seed(seed + point)
            sigma = torch.tensor(
                sigma_value, device=DEVICE, dtype=torch.float32
            )
            noisy = project(clean + sigma * torch.randn_like(clean))
            vector = project(
                2 * torch.randint_like(clean, 0, 2) - 1
            )
            numerator = torch.zeros((), device=DEVICE)
            denominator = torch.zeros((), device=DEVICE)
            for start in range(0, len(clean), EMS_BATCH):
                stop = start + EMS_BATCH
                state = noisy[start:stop]
                tangent = vector[start:stop]
                cond = condition[start:stop]
                with fwAD.dual_level():
                    dual = fwAD.make_dual(state, tangent)
                    _, jvp = fwAD.unpack_dual(epsilon(dual, sigma, cond))
                numerator += (sigma * jvp * tangent).sum()
                denominator += tangent.square().sum()
            l_values[point] = (numerator / denominator).item()
            if point % 20 == 0 or point == STATISTICS_STEPS:
                print(
                    f"EMS l {point}/{STATISTICS_STEPS} "
                    f"sigma={sigma_value:.5g} l={l_values[point]:+.6g}",
                    flush=True,
                )

        l_derivative = smooth_derivative(l_values, gap)
        moments = np.empty((4, len(sigmas)), dtype=np.float64)
        for point, sigma_value in enumerate(sigmas):
            torch.manual_seed(seed + 1000 + point)
            sigma = torch.tensor(
                sigma_value, device=DEVICE, dtype=torch.float32
            )
            noisy = project(clean + sigma * torch.randn_like(clean))
            totals = torch.zeros(4, device=DEVICE)
            count = 0
            for start in range(0, len(clean), EMS_BATCH):
                stop = start + EMS_BATCH
                state = noisy[start:stop]
                cond = condition[start:stop]
                noise = epsilon(state, sigma, cond)
                with fwAD.dual_level():
                    dual_state = fwAD.make_dual(
                        state, -sigma * noise
                    )
                    dual_sigma = fwAD.make_dual(sigma, -sigma)
                    _, noise_derivative = fwAD.unpack_dual(
                        epsilon(dual_state, dual_sigma, cond)
                    )
                l_value = float(l_values[point])
                f = sigma * noise - l_value * state
                f_derivative = (
                    sigma * ((l_value - 1) * noise + noise_derivative)
                    - float(l_derivative[point]) * state
                )
                totals += torch.stack((
                    f.sum(), f_derivative.sum(), f.square().sum(),
                    (f * f_derivative).sum(),
                ))
                count += f.numel()
            moments[:, point] = (totals / count).cpu().numpy()
            if point % 20 == 0 or point == STATISTICS_STEPS:
                variance = moments[2, point] - moments[0, point] ** 2
                print(
                    f"EMS f {point}/{STATISTICS_STEPS} "
                    f"sigma={sigma_value:.5g} var={variance:.6g}",
                    flush=True,
                )

    f_mean, fd_mean, ff_mean, ffd_mean = moments
    variance = ff_mean - f_mean ** 2
    s_values = (ffd_mean - f_mean * fd_mean) / variance
    b_values = fd_mean - s_values * f_mean
    for name, values in (
        ("l", l_values), ("s", s_values), ("b", b_values)
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite EMS {name}")
    np.savez_compressed(
        output, l=l_values, s=s_values, b=b_values,
        l_d=l_derivative, lambdas=lambdas, sigmas=sigmas,
        sample_index=sample_index, seed=seed,
        condition_path=str(paths[0]), target_path=str(paths[1]),
    )
    print(
        f"saved {output} in {time.perf_counter() - start_all:.1f} s "
        f"l=[{l_values.min():+.4g},{l_values.max():+.4g}] "
        f"s=[{s_values.min():+.4g},{s_values.max():+.4g}]",
        flush=True,
    )


def weighted_integral(exponent, x, weight, cumulative=True):
    maximum = np.max(exponent)
    value = weight * np.exp(exponent - maximum)
    interval = 0.5 * np.diff(x) * (value[1:] + value[:-1])
    if not cumulative:
        return interval.sum() * np.exp(maximum)
    result = np.cumsum(interval) * np.exp(maximum)
    return np.concatenate(([0.0], result))


def prepare_solver(ems_file):
    with np.load(ems_file) as archive:
        l = np.asarray(archive["l"], dtype=np.float64)
        s = np.asarray(archive["s"], dtype=np.float64)
        b = np.asarray(archive["b"], dtype=np.float64)
        statistics_sigmas = np.asarray(archive["sigmas"], dtype=np.float64)
    lambda_start = -torch.log(
        torch.tensor(float(statistics_sigmas[0]), dtype=torch.float32)
    ).double().item()
    lambda_stop = -torch.log(
        torch.tensor(float(statistics_sigmas[-1]), dtype=torch.float32)
    ).double().item()
    integration_grid = torch.linspace(
        lambda_start, lambda_stop, STATISTICS_STEPS + 1,
        dtype=torch.float32,
    )
    integration_sigmas = torch.exp(-integration_grid).double()
    lambdas = (-torch.log(integration_sigmas)).numpy()

    zeros = np.zeros_like(l)
    ones = np.ones_like(l)
    integral_l = weighted_integral(zeros, lambdas, l)
    integral_s = weighted_integral(zeros, lambdas, s)
    integral_one = weighted_integral(
        integral_l + integral_s, lambdas, ones
    )
    integral_b = weighted_integral(-integral_s, lambdas, b)
    integral_c = weighted_integral(
        integral_l + integral_s, lambdas, integral_b
    )
    coefficients = np.empty((2, NFE), dtype=np.float64)
    for step, (start, stop) in enumerate(
        zip(SAMPLE_INDEX[:-1], SAMPLE_INDEX[1:])
    ):
        x = lambdas[start:stop + 1]
        exponent = (
            integral_l[start:stop + 1]
            + integral_s[start:stop + 1]
            - integral_l[start] - integral_s[start]
        )
        for order in (2, 3):
            weight = (x - x[0]) ** (order - 1) / math.factorial(order - 1)
            coefficients[order - 2, step] = weighted_integral(
                exponent, x, weight, cumulative=False
            )
    arrays = (
        l, integral_l, integral_s, integral_one, integral_b,
        integral_c, lambdas, coefficients,
    )
    tensors = tuple(
        torch.from_numpy(value).to(DEVICE) for value in arrays
    )
    index = torch.arange(
        STATISTICS_STEPS + 1, device=DEVICE, dtype=torch.float32
    )
    sample_grid = (
        float(lambdas[0])
        + float(lambdas[-1] - lambdas[0]) * index / STATISTICS_STEPS
    )
    sigma_at_index = torch.exp(-sample_grid).double()
    lambda_at_index = -torch.log(sigma_at_index.float()).double()
    tensors += (sigma_at_index, lambda_at_index)
    sample_sigmas = sigma_at_index[SAMPLE_INDEX]
    initial_sigma = torch.tensor(
        float(statistics_sigmas[0]), device=DEVICE, dtype=torch.float32
    )
    return tensors, sample_sigmas, initial_sigma


def dpm_solver_v3(model, condition, noise, solver, diffusion_sigma):
    (
        l, integral_l, integral_s, integral_one, integral_b,
        integral_c, lambdas, coefficients,
        sigma_at_index, lambda_at_index,
    ), sigmas, initial_sigma = solver
    x = project(noise * initial_sigma).double()
    x_history = []
    epsilon_history = []
    index_history = []
    log_base = np.log(diffusion_sigma)

    for step in range(NFE):
        index = int(SAMPLE_INDEX[step])
        sigma = sigmas[step].float()
        train_time = (
            torch.log1p(2 * log_base * sigma.square()) / (2 * log_base)
        ).float()
        score = model(
            torch.cat((x.float(), condition), dim=1),
            train_time.expand(len(x)),
        ).float()
        epsilon = (-sigma * project(score)).double()
        x_history.append(x)
        epsilon_history.append(epsilon)
        index_history.append(index)
        if len(x_history) > 3:
            x_history.pop(0)
            epsilon_history.pop(0)
            index_history.pop(0)

        order = min(3, step + 1, NFE - step)
        reverse = tuple(range(len(x_history) - 1, len(x_history) - order - 1, -1))
        current = reverse[0]
        start_index = index_history[current]
        stop_index = int(SAMPLE_INDEX[step + 1])
        g_values = []
        for position in reverse:
            point = index_history[position]
            f = (
                sigma_at_index[point] * epsilon_history[position]
                - l[point] * x_history[position]
            )
            g_values.append(
                torch.exp(integral_s[start_index] - integral_s[point]) * f
                - torch.exp(integral_s[start_index])
                * (integral_b[point] - integral_b[start_index])
            )

        x = (
            torch.exp(integral_l[start_index] - integral_l[stop_index])
            * x_history[current]
            - torch.exp(-integral_l[stop_index] - integral_s[start_index])
            * (integral_one[stop_index] - integral_one[start_index])
            * g_values[0]
            - torch.exp(-integral_l[stop_index])
            * (
                integral_c[stop_index] - integral_c[start_index]
                - integral_b[start_index]
                * (integral_one[stop_index] - integral_one[start_index])
            )
        )
        if order > 1:
            lambda_values = torch.stack([
                lambda_at_index[index_history[position]]
                for position in reverse
            ])
            delta = lambda_values[1:] - lambda_values[0]
            matrix = torch.stack([
                delta ** power for power in range(1, order)
            ], dim=1)
            right = (
                torch.stack(g_values[1:]) - g_values[0]
            ).reshape(order - 1, -1)
            derivatives = (torch.linalg.inv(matrix) @ right).reshape(
                (order - 1,) + g_values[0].shape
            )
            for derivative in range(order - 1):
                x -= (
                    torch.exp(
                        integral_l[start_index] - integral_l[stop_index]
                    )
                    * coefficients[derivative, step]
                    * derivatives[derivative]
                    * math.factorial(derivative + 1)
                )
    return project(x.float())
def generate(model, hparams, timing_reference, output, ems_file, checkpoint):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with np.load(INITIAL, allow_pickle=False) as archive:
        initial = np.asarray(archive["phi_gen"], dtype=np.float32)
        cond_indices = np.asarray(
            archive["cond_indices"]
            if "cond_indices" in archive
            else np.arange(len(initial)),
            dtype=np.int64,
        )
    with h5py.File(timing_reference, "r") as handle:
        times = np.asarray(handle["times"], dtype=np.float64)
        seeds = np.asarray(handle["seeds"], dtype=np.int64)

    solver = prepare_solver(ems_file)
    sigmas = solver[1]
    fields = torch.from_numpy(initial).to(DEVICE)
    n, lattice = fields.shape[0], fields.shape[-1]
    norm_res = float(hparams["norm_max"])
    norm_cond = float(hparams["cond_norm_max"])
    diffusion_sigma = float(hparams["sigma"])
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "x") as handle:
        chain = handle.create_dataset(
            "phi_gen", (len(times), n, lattice, lattice), dtype="f4",
            chunks=(1, n, lattice, lattice),
        )
        handle["times"] = times
        handle["seeds"] = seeds
        handle["cond_indices"] = cond_indices
        handle["sigmas"] = sigmas.cpu().numpy()
        seconds = handle.create_dataset(
            "sampling_seconds", (len(times),), dtype="f4"
        )
        zero_mode = handle.create_dataset(
            "zero_mode_raw", (len(times), n), dtype="f4"
        )
        handle.attrs.update(
            checkpoint=str(checkpoint), parent=str(INITIAL),
            timing_reference=str(timing_reference), ems_file=str(ems_file),
            horizon=float(hparams["horizon"]), solver="DPM-Solver-v3",
            nfe=NFE, order=3, skip_type="logSNR",
            statistics_steps=STATISTICS_STEPS,
            ems_samples=EMS_SAMPLES, lower_order_final=True,
            ems_lattice=32, ems_distribution="training pairs",
            probability_flow=True, score_to_noise=True,
            compile_mode=COMPILE_MODE, L=lattice, n_samples=n,
            kappa=hparams.get("kappa"), lam=hparams.get("lam"),
        )

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            for transition, seed in enumerate(seeds):
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                raw_means = np.empty(n, dtype=np.float32)
                for start in range(0, n, SAMPLE_BATCH):
                    stop = min(start + SAMPLE_BATCH, n)
                    condition = (
                        fields[start:stop, None] / norm_cond
                    ).to(memory_format=torch.channels_last)
                    noise = project(torch.randn_like(condition))
                    residual = dpm_solver_v3(
                        model, condition, noise, solver, diffusion_sigma
                    )[:, 0]
                    residual *= norm_res
                    raw_means[start:stop] = (
                        residual.mean(dim=(-2, -1)).cpu().numpy()
                    )
                    fields[start:stop].add_(residual)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                chain[transition] = fields.cpu().numpy()
                seconds[transition] = elapsed
                zero_mode[transition] = raw_means
                handle.flush()
                print(
                    f"{output.name} t={times[transition]:g} "
                    f"{elapsed:.3f} s max|mean(dphi)|="
                    f"{np.abs(raw_means).max():.3e}",
                    flush=True,
                )
    print(f"saved {output}", flush=True)


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    for run, timing_reference, ems_paths, output_name, seed in SPECS:
        checkpoint = run / "models/epoch=10000.ckpt"
        ems_file = run / "dpm_solver_v3_ems_L32_k256_n120.npz"
        output = run / "samples" / output_name
        print(f"starting {run.name}", flush=True)
        model, hparams = load_model(checkpoint)
        with model.ema.average_parameters():
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            compute_ems(
                model, hparams, ems_paths, ems_file, seed
            )
            torch.cuda.empty_cache()
            model.score_model = torch.compile(
                model.score_model.to(memory_format=torch.channels_last),
                mode=COMPILE_MODE, dynamic=False,
            )
            generate(
                model, hparams, timing_reference, output,
                ems_file, checkpoint,
            )
        del model
        torch.compiler.reset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
