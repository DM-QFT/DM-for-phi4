"""Generate the two L64 chains with the original NCSN++ checkpoints at 32 NFE."""

from pathlib import Path
import functools
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D
from diffusion_lightning import marginal_prob_std


INITIAL = (
    ROOT
    / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
    / "samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"
)
SPECS = (
    (
        ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip",
        ROOT / "results/multiL/t0p5_L64.h5",
        "hmc_L64_N512_t50_dpmpp2m_nfe32_learned.h5",
        2026083102,
    ),
    (
        ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip",
        ROOT / "results/multiL/t10_L64.h5",
        "hmc_L64_N512_t1000_dpmpp2m_nfe32_learned.h5",
        2026083101,
    ),
)
DEVICE = torch.device("cuda:0")
NFE = 32
TEACHER_NFE = 128
ORIGINS = (0, 9, 49, 99)
PAIR_COUNT = 32
FIT_COUNT = 16
GRID_UPDATES = 100
GRID_LR = 3e-2
LOW_WEIGHT = 1.0
MOMENT_WEIGHT = 1.0
GRID_REGULARIZATION = 1e-4
EPS = 1e-5
BATCH_SIZE = 512


def project(x):
    return x - x.mean(dim=(-2, -1), keepdim=True)


def log_grid(nfe, sigma_min, sigma_max):
    sigmas = torch.exp(torch.linspace(
        np.log(sigma_max), np.log(sigma_min), nfe, device=DEVICE
    ))
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def learned_grid(logits, sigma_min, sigma_max):
    span = np.log(sigma_max / sigma_min)
    intervals = torch.softmax(logits, dim=0) * span
    log_sigmas = torch.cat((
        logits.new_tensor([np.log(sigma_max)]),
        np.log(sigma_max) - intervals.cumsum(0),
    ))
    return torch.cat((log_sigmas.exp(), logits.new_zeros(1)))


def dpmpp2m(model, condition, noise, sigmas, diffusion_sigma):
    residual = project(noise * sigmas[0])
    old_denoised = None
    h_last = None
    log_base = np.log(diffusion_sigma)

    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        t = torch.log1p(2 * log_base * sigma.square()) / (2 * log_base)
        score = model(
            torch.cat((residual, condition), dim=1), t.expand(len(residual))
        ).float()
        denoised = project(residual + sigma.square() * project(score))

        if sigma_next == 0:
            residual = denoised
        else:
            h = torch.log(sigma / sigma_next)
            if old_denoised is None:
                denoised_d = denoised
            else:
                r = h_last / h
                denoised_d = (
                    (1 + 0.5 / r) * denoised - 0.5 / r * old_denoised
                )
            residual = project(
                sigma_next / sigma * residual
                - torch.expm1(-h) * denoised_d
            )
            h_last = h
        old_denoised = denoised
    return project(residual)


def dpmpp2m_sde(model, condition, noise, sigmas, diffusion_sigma, eta=1.0):
    """DPM-Solver++ 2M SDE with midpoint correction."""
    residual = project(noise * sigmas[0])
    old_denoised = None
    h_last = None
    log_base = np.log(diffusion_sigma)

    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        t = torch.log1p(2 * log_base * sigma.square()) / (2 * log_base)
        score = model(
            torch.cat((residual, condition), dim=1), t.expand(len(residual))
        ).float()
        denoised = project(residual + sigma.square() * project(score))

        if sigma_next == 0:
            residual = denoised
        else:
            h = torch.log(sigma / sigma_next)
            eta_h = eta * h
            gain = -torch.expm1(-h - eta_h)
            residual = (
                sigma_next / sigma * torch.exp(-eta_h) * residual
                + gain * denoised
            )
            if old_denoised is not None:
                r = h_last / h
                residual = residual + 0.5 * gain / r * (
                    denoised - old_denoised
                )
            if eta:
                residual = residual + (
                    sigma_next * torch.sqrt(-torch.expm1(-2 * eta_h))
                    * project(torch.randn_like(residual))
                )
            residual = project(residual)
            h_last = h
        old_denoised = denoised
    return project(residual)


def prediction_errors(prediction, target):
    error = prediction - target
    field = error.square().mean()
    spectrum = torch.fft.fft2(error[:, 0], norm="ortho")
    n = torch.arange(1, 5, device=error.device)
    low = torch.cat((
        spectrum[:, n, n], spectrum[:, n, -n],
        spectrum[:, -n, n], spectrum[:, -n, -n],
    ), dim=1).abs().square().mean()
    return field, low


def moment_error(condition, prediction, target, norm_cond, norm_res):
    groups = len(ORIGINS)
    samples = len(condition) // groups
    shape = (groups, samples, *condition.shape[1:])
    base = condition.reshape(shape) * norm_cond
    prediction = base + prediction.reshape(shape) * norm_res
    target = base + target.reshape(shape) * norm_res
    prediction_phi2 = prediction.square().mean(dim=(1, 2, 3, 4))
    target_phi2 = target.square().mean(dim=(1, 2, 3, 4))
    prediction_phi4 = prediction.pow(4).mean(dim=(1, 2, 3, 4))
    target_phi4 = target.pow(4).mean(dim=(1, 2, 3, 4))
    relative_phi2 = prediction_phi2 / target_phi2 - 1
    relative_phi4 = prediction_phi4 / target_phi4 - 1
    loss = (relative_phi2.square() + relative_phi4.square()).mean()
    return loss, relative_phi2.mean(), relative_phi4.mean()


def evaluate(model, pairs, sigmas, diffusion_sigma, norm_cond, norm_res):
    condition, noise, target = pairs
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = dpmpp2m(
            model, condition, noise, sigmas, diffusion_sigma
        )
        field, low = prediction_errors(prediction, target)
        moment, dphi2, dphi4 = moment_error(
            condition, prediction, target, norm_cond, norm_res
        )
    return np.array([
        field.item(), low.item(), moment.item(), dphi2.item(), dphi4.item()
    ])


def teacher_pairs(model, reference, diffusion_sigma, sigma_min, sigma_max,
                  norm_cond, teacher_seed):
    with np.load(INITIAL, allow_pickle=False) as archive:
        initial = np.asarray(archive["phi_gen"], dtype=np.float32)
    sources = []
    with h5py.File(reference, "r") as handle:
        for origin in ORIGINS:
            sources.append(
                initial if origin == 0
                else np.asarray(handle["phi_gen"][origin - 1])
            )

    rng = np.random.default_rng(teacher_seed)
    conditions = []
    for source in sources:
        index = rng.permutation(len(source))[:PAIR_COUNT]
        conditions.append(source[index])
    condition = torch.from_numpy(np.stack(conditions))[:, :, None].to(DEVICE)
    condition = (condition / norm_cond).reshape(-1, 1, 64, 64)
    condition = condition.to(memory_format=torch.channels_last)
    torch.manual_seed(teacher_seed)
    torch.cuda.manual_seed_all(teacher_seed)
    noise = project(torch.randn_like(condition))
    teacher_sigmas = log_grid(TEACHER_NFE, sigma_min, sigma_max)
    print(f"generating {TEACHER_NFE} NFE teacher", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        target = dpmpp2m(
            model, condition, noise, teacher_sigmas, diffusion_sigma
        ).detach()

    shape = (len(ORIGINS), PAIR_COUNT, 1, 64, 64)
    values = tuple(x.reshape(shape) for x in (condition, noise, target))
    fit = tuple(
        x[:, :FIT_COUNT].reshape(-1, 1, 64, 64)
        .to(memory_format=torch.channels_last)
        for x in values
    )
    valid = tuple(
        x[:, FIT_COUNT:].reshape(-1, 1, 64, 64)
        .to(memory_format=torch.channels_last)
        for x in values
    )
    return fit, valid


def fit_grid(model, reference, diffusion_sigma, sigma_min, sigma_max,
             norm_cond, norm_res, teacher_seed):
    fit, valid = teacher_pairs(
        model, reference, diffusion_sigma, sigma_min, sigma_max,
        norm_cond, teacher_seed,
    )
    base = log_grid(NFE, sigma_min, sigma_max)
    logits = torch.nn.Parameter(torch.zeros(NFE - 1, device=DEVICE))
    optimizer = torch.optim.Adam((logits,), lr=GRID_LR)
    initial_fit = evaluate(
        model, fit, base, diffusion_sigma, norm_cond, norm_res
    )
    initial_valid = evaluate(
        model, valid, base, diffusion_sigma, norm_cond, norm_res
    )
    best_logits = logits.detach().clone()
    best_score = (
        initial_valid[0] + LOW_WEIGHT * initial_valid[1]
        + MOMENT_WEIGHT * initial_valid[2]
    )
    print(f"initial fit {initial_fit}", flush=True)
    print(f"initial valid {initial_valid}", flush=True)

    uniform = torch.full_like(logits, 1 / len(logits))
    condition, noise, target = fit
    for update in range(1, GRID_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        sigmas = learned_grid(logits, sigma_min, sigma_max)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = dpmpp2m(
                model, condition, noise, sigmas, diffusion_sigma
            )
            field, low = prediction_errors(prediction, target)
            moment, dphi2, dphi4 = moment_error(
                condition, prediction, target, norm_cond, norm_res
            )
            weights = torch.softmax(logits, dim=0)
            regularization = torch.log(weights / uniform).square().mean()
            loss = (
                field + LOW_WEIGHT * low + MOMENT_WEIGHT * moment
                + GRID_REGULARIZATION * regularization
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_((logits,), 1.0)
        optimizer.step()

        if update % 10 == 0:
            candidate = learned_grid(
                logits.detach(), sigma_min, sigma_max
            )
            valid_error = evaluate(
                model, valid, candidate, diffusion_sigma, norm_cond, norm_res
            )
            score = (
                valid_error[0] + LOW_WEIGHT * valid_error[1]
                + MOMENT_WEIGHT * valid_error[2]
            )
            if score < best_score:
                best_score = score
                best_logits = logits.detach().clone()
            print(
                f"update {update} field {field.item():.6g} "
                f"low {low.item():.6g} dphi2 {100*dphi2.item():+.3f}% "
                f"dphi4 {100*dphi4.item():+.3f}% valid {valid_error}",
                flush=True,
            )

    sigmas = learned_grid(best_logits, sigma_min, sigma_max).detach()
    final_fit = evaluate(
        model, fit, sigmas, diffusion_sigma, norm_cond, norm_res
    )
    final_valid = evaluate(
        model, valid, sigmas, diffusion_sigma, norm_cond, norm_res
    )
    print(f"learned fit {final_fit}", flush=True)
    print(f"learned valid {final_valid}", flush=True)
    errors = np.concatenate((
        initial_fit, initial_valid, final_fit, final_valid
    ))
    return sigmas, base, errors


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = state["hyper_parameters"]
    diffusion_sigma = float(hparams["sigma"])
    network = NCSNppCond2D(
        functools.partial(marginal_prob_std, sigma=diffusion_sigma),
        in_channels=int(hparams.get("in_channels", 2)),
    )
    model = ConditionalDiffusionModel.load_from_checkpoint(
        checkpoint, score_model=network, map_location="cpu", weights_only=False
    ).to(DEVICE).eval()
    model.ema.to(DEVICE)
    return model, hparams


def load_or_fit_grid(model, run, reference, hparams, teacher_seed):
    grid_file = run / "dpmpp2m_nfe32_sigmas.npz"
    diffusion_sigma = float(hparams["sigma"])
    sigma_max = marginal_prob_std(
        torch.tensor(1.0), diffusion_sigma
    ).item()
    sigma_min = marginal_prob_std(
        torch.tensor(EPS), diffusion_sigma
    ).item()
    if grid_file.exists():
        with np.load(grid_file) as archive:
            sigmas = torch.from_numpy(archive["sigmas"]).to(DEVICE)
        print(f"loaded {grid_file}", flush=True)
    else:
        sigmas, uniform, errors = fit_grid(
            model, reference, diffusion_sigma, sigma_min, sigma_max,
            float(hparams["cond_norm_max"]), float(hparams["norm_max"]),
            teacher_seed,
        )
        np.savez(
            grid_file,
            sigmas=sigmas.cpu().numpy(),
            uniform_sigmas=uniform.cpu().numpy(),
            errors=errors,
        )
        print(f"saved {grid_file}", flush=True)

    values = sigmas.detach().cpu().numpy()
    if (
        values.shape != (NFE + 1,)
        or not np.all(np.diff(values[:-1]) < 0)
        or not np.isclose(values[0], sigma_max, rtol=1e-5)
        or not np.isclose(values[-2], sigma_min, rtol=1e-5)
        or values[-1] != 0
    ):
        raise ValueError(f"invalid sigma grid in {grid_file}")
    return sigmas, grid_file


def generate(model, hparams, reference, output, sigmas, grid_file, checkpoint,
             sigma_grid="learned"):
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
    with h5py.File(reference, "r") as handle:
        times = np.asarray(handle["times"], dtype=np.float64)
        seeds = np.asarray(handle["seeds"], dtype=np.int64)

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
        metadata = dict(
            checkpoint=str(checkpoint),
            parent=str(INITIAL),
            reference_chain=str(reference),
            horizon=float(hparams["horizon"]),
            solver="DPM-Solver++ 2M",
            nfe=len(sigmas) - 1,
            sigma_grid=sigma_grid,
            score_to_denoiser=True,
            probability_flow=True,
            compile_mode="reduce-overhead",
            L=lattice,
            n_samples=n,
            kappa=hparams.get("kappa"),
            lam=hparams.get("lam"),
        )
        if grid_file is not None:
            metadata.update(
                sigma_grid_file=str(grid_file),
                teacher_nfe=TEACHER_NFE,
            )
        handle.attrs.update(metadata)

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            for transition, seed in enumerate(seeds):
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                raw_means = np.empty(n, dtype=np.float32)
                for start in range(0, n, BATCH_SIZE):
                    stop = min(start + BATCH_SIZE, n)
                    condition = (
                        fields[start:stop, None] / norm_cond
                    ).to(memory_format=torch.channels_last)
                    noise = project(torch.randn_like(condition))
                    residual = dpmpp2m(
                        model, condition, noise, sigmas, diffusion_sigma
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
    for run, reference, output_name, teacher_seed in SPECS:
        checkpoint = run / "models/epoch=10000.ckpt"
        output = run / "samples" / output_name
        print(f"starting {run.name}", flush=True)
        model, hparams = load_model(checkpoint)
        with model.ema.average_parameters():
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.score_model = torch.compile(
                model.score_model.to(memory_format=torch.channels_last),
                mode="reduce-overhead", dynamic=False,
            )
            sigmas, grid_file = load_or_fit_grid(
                model, run, reference, hparams, teacher_seed
            )
            torch.cuda.empty_cache()
            generate(
                model, hparams, reference, output, sigmas,
                grid_file, checkpoint,
            )
        del model
        torch.compiler.reset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
