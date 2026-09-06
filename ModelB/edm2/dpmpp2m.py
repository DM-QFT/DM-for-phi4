"""Learn the 32 NFE sigma grid and generate the final Model B chain."""

from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
MODEL_B = HERE.parent
sys.path[:0] = [str(HERE), str(MODEL_B), str(MODEL_B.parent)]

import h5py
import numpy as np
import torch

from cond_lightning import NCSNppCond2D
from edm_lightning import trivial_std
from train import CHANNELS, ConditionalEDM2


RUN = (
    MODEL_B
    / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_"
    "edm_ncsnpp_fahmc_noskip"
)
CHECKPOINT = RUN / "models/epoch=10000.ckpt"
INITIAL = (
    MODEL_B
    / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
    / "samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"
)
TEACHER = RUN / "samples/hmc_L64_N512_t1000_heun128_pairs.h5"
GRID_FILE = RUN / "dpmpp2m_nfe32_sigmas.npz"
OUTPUT = RUN / "samples/hmc_L64_N512_t1000_dpmpp2m_nfe32_learned.h5"
NFE = 32
TEACHER_TRANSITIONS = (0, 9, 49, 99)
FIT_SAMPLES_PER_TRANSITION = 16
GRID_UPDATES = 100
GRID_LR = 3e-2
MOMENT_UPDATES = 60
MOMENT_LR = 1e-2
MOMENT_WEIGHT = 0.1
GRID_REGULARIZATION = 1e-4
LOW_MODE_WEIGHT = 1.0
TARGET_TIME = 1000.0
BATCH_SIZE = 512
SEED = 2026082400
DEVICE = "cuda:0"


def karras_sigmas(nfe, sigma_min, sigma_max, rho=7.0, device=DEVICE):
    step = torch.arange(nfe, device=device, dtype=torch.float32)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + step / (nfe - 1)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ).pow(rho)
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def sigma_grid(logits, sigma_min, sigma_max):
    u_start = -np.log(sigma_max)
    span = np.log(sigma_max / sigma_min)
    intervals = torch.softmax(logits, dim=0) * span
    u = torch.cat((logits.new_tensor([u_start]), u_start + intervals.cumsum(0)))
    sigmas = torch.exp(-u)
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def dpmpp2m(model, condition, noise, sigmas):
    residual = model.project(noise * sigmas[0])
    old_denoised = None
    h_last = None
    batch = len(condition)

    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = model.denoise(residual, condition, sigma.expand(batch))
        sigma_next = sigmas[index + 1]
        if index == len(sigmas) - 2:
            residual = denoised
        else:
            h = torch.log(sigma / sigma_next)
            if old_denoised is None:
                denoised_d = denoised
            else:
                r = h_last / h
                denoised_d = (
                    (1.0 + 0.5 / r) * denoised
                    - 0.5 / r * old_denoised
                )
            residual = model.project(
                sigma_next / sigma * residual
                - torch.expm1(-h) * denoised_d
            )
            h_last = h
        old_denoised = denoised
    return model.project(residual)


@torch.no_grad()
def generate_teacher(model, checkpoint, teacher, target_time, seed):
    with np.load(INITIAL, allow_pickle=False) as archive:
        initial = np.asarray(archive["phi_gen"], dtype=np.float32)
    fields = torch.from_numpy(initial).to(DEVICE)
    n, lattice = fields.shape[0], fields.shape[-1]
    horizon = float(model.hparams.horizon)
    transitions = round(target_time / horizon)
    if max(TEACHER_TRANSITIONS) >= transitions:
        raise ValueError("teacher transition lies beyond the generated chain")

    teacher.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(teacher, "x") as handle:
        conditions = handle.create_dataset(
            "condition",
            (len(TEACHER_TRANSITIONS), n, lattice, lattice),
            dtype="f4",
            chunks=(1, n, lattice, lattice),
        )
        targets = handle.create_dataset(
            "target",
            (len(TEACHER_TRANSITIONS), n, lattice, lattice),
            dtype="f4",
            chunks=(1, n, lattice, lattice),
        )
        handle["transitions"] = TEACHER_TRANSITIONS
        handle["times"] = horizon * (np.asarray(TEACHER_TRANSITIONS) + 1)
        handle.attrs.update(
            checkpoint=str(checkpoint),
            initial=str(INITIAL),
            horizon=horizon,
            solver="Heun",
            num_steps=128,
            seed=seed,
            L=lattice,
            n_samples=n,
        )

        selected = {
            transition: i
            for i, transition in enumerate(TEACHER_TRANSITIONS)
        }
        for transition in range(transitions):
            pair = selected.get(transition)
            if pair is not None:
                conditions[pair] = fields.cpu().numpy()
            torch.manual_seed(seed + transition + 1)
            torch.cuda.manual_seed_all(seed + transition + 1)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            condition = (fields[:, None] / float(model.norm_cond)).to(
                memory_format=torch.channels_last
            )
            residual = model.sample(
                condition, num_steps=128, use_ema=False
            )[:, 0]
            fields.add_(residual * float(model.norm_res))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time
            if pair is not None:
                targets[pair] = fields.cpu().numpy()
                handle.flush()
            print(
                f"teacher t={(transition + 1) * horizon:g} {seconds:.3f} s",
                flush=True,
            )


def teacher_pairs(model, teacher, seed):
    fit = [[], [], []]
    valid = [[], [], []]
    with h5py.File(teacher, "r") as handle:
        for pair, transition in enumerate(TEACHER_TRANSITIONS):
            previous = np.asarray(handle["condition"][pair], dtype=np.float32)
            current = np.asarray(handle["target"][pair], dtype=np.float32)
            condition = torch.from_numpy(previous[:, None]).to(DEVICE)
            condition = (condition / float(model.norm_cond)).to(
                memory_format=torch.channels_last
            )
            target = torch.from_numpy((current - previous)[:, None]).to(DEVICE)
            target = model.project(target / float(model.norm_res))

            torch.manual_seed(seed + transition + 1)
            torch.cuda.manual_seed_all(seed + transition + 1)
            noise = model.project(torch.randn_like(condition))

            count = FIT_SAMPLES_PER_TRANSITION
            for destination, start in ((fit, 0), (valid, count)):
                stop = start + count
                destination[0].append(condition[start:stop])
                destination[1].append(noise[start:stop])
                destination[2].append(target[start:stop])

    pairs = []
    for group in (fit, valid):
        pairs.append(tuple(
            torch.cat(values).to(memory_format=torch.channels_last)
            for values in group
        ))
    return pairs


def errors(prediction, target):
    error = prediction - target
    field = error.square().mean()
    spectrum = torch.fft.fft2(error[:, 0], norm="ortho")
    n = torch.arange(1, 5, device=error.device)
    modes = torch.cat((
        spectrum[:, n, n], spectrum[:, n, -n],
        spectrum[:, -n, n], spectrum[:, -n, -n],
    ), dim=1)
    low = modes.abs().square().mean()
    return field, low


def moment_error(model, condition, prediction, target):
    groups = len(TEACHER_TRANSITIONS)
    samples = len(condition) // groups
    shape = (groups, samples, *condition.shape[1:])
    condition = condition.reshape(shape) * float(model.norm_cond)
    prediction = condition + prediction.reshape(shape) * float(model.norm_res)
    target = condition + target.reshape(shape) * float(model.norm_res)

    prediction_mean = prediction.mean(dim=(-2, -1), keepdim=True)
    target_mean = target.mean(dim=(-2, -1), keepdim=True)
    prediction_variance = (prediction - prediction_mean).square().mean(
        dim=(1, 2, 3, 4)
    )
    target_variance = (target - target_mean).square().mean(
        dim=(1, 2, 3, 4)
    )
    prediction_phi4 = prediction.pow(4).mean(dim=(1, 2, 3, 4))
    target_phi4 = target.pow(4).mean(dim=(1, 2, 3, 4))
    relative_variance = prediction_variance / target_variance - 1.0
    relative_phi4 = prediction_phi4 / target_phi4 - 1.0
    loss = (relative_variance.square() + relative_phi4.square()).mean()
    return loss, relative_variance.mean(), relative_phi4.mean()


def evaluate(model, pairs, sigmas):
    condition, noise, target = pairs
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = dpmpp2m(model, condition, noise, sigmas)
        field, low = errors(prediction, target)
        moment, relative_variance, relative_phi4 = moment_error(
            model, condition, prediction, target
        )
    return tuple(
        value.item()
        for value in (field, low, moment, relative_variance, relative_phi4)
    )


def learn_grid(model, fit, valid, nfe):
    base = karras_sigmas(nfe, model.sigma_min, model.sigma_max)
    base_intervals = -torch.diff(torch.log(base[:-1]))
    base_weights = base_intervals / base_intervals.sum()
    logits = torch.nn.Parameter(torch.log(base_weights))
    optimizer = torch.optim.Adam((logits,), lr=GRID_LR)

    initial_fit = evaluate(model, fit, base)
    initial_valid = evaluate(model, valid, base)
    best_valid = initial_valid[0] + LOW_MODE_WEIGHT * initial_valid[1]
    best_logits = logits.detach().clone()
    print(
        f"nfe={nfe} karras fit={initial_fit} valid={initial_valid}",
        flush=True,
    )

    condition, noise, target = fit
    for update in range(1, GRID_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        sigmas = sigma_grid(logits, model.sigma_min, model.sigma_max)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = dpmpp2m(model, condition, noise, sigmas)
            field, low = errors(prediction, target)
            weights = torch.softmax(logits, dim=0)
            regularization = torch.log(weights / base_weights).square().mean()
            loss = (
                field + LOW_MODE_WEIGHT * low
                + GRID_REGULARIZATION * regularization
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_((logits,), 1.0)
        optimizer.step()

        if update % 10 == 0:
            candidate = sigma_grid(
                logits.detach(), model.sigma_min, model.sigma_max
            )
            valid_error = evaluate(model, valid, candidate)
            valid_total = valid_error[0] + LOW_MODE_WEIGHT * valid_error[1]
            if valid_total < best_valid:
                best_valid = valid_total
                best_logits = logits.detach().clone()
            print(
                f"nfe={nfe} update={update} fit=({field.item():.6g}, "
                f"{low.item():.6g}) valid={valid_error[:2]}",
                flush=True,
            )

    with torch.no_grad():
        logits.copy_(best_logits)
    reference_weights = torch.softmax(logits.detach(), dim=0)
    optimizer = torch.optim.Adam((logits,), lr=MOMENT_LR)
    current = evaluate(
        model,
        valid,
        sigma_grid(logits.detach(), model.sigma_min, model.sigma_max),
    )
    best_valid = (
        current[0] + LOW_MODE_WEIGHT * current[1]
        + MOMENT_WEIGHT * current[2]
    )

    for update in range(1, MOMENT_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        sigmas = sigma_grid(logits, model.sigma_min, model.sigma_max)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = dpmpp2m(model, condition, noise, sigmas)
            field, low = errors(prediction, target)
            moment, relative_variance, relative_phi4 = moment_error(
                model, condition, prediction, target
            )
            weights = torch.softmax(logits, dim=0)
            regularization = torch.log(
                weights / reference_weights
            ).square().mean()
            loss = (
                field + LOW_MODE_WEIGHT * low + MOMENT_WEIGHT * moment
                + GRID_REGULARIZATION * regularization
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_((logits,), 1.0)
        optimizer.step()

        if update % 10 == 0:
            candidate = sigma_grid(
                logits.detach(), model.sigma_min, model.sigma_max
            )
            valid_error = evaluate(model, valid, candidate)
            valid_total = (
                valid_error[0] + LOW_MODE_WEIGHT * valid_error[1]
                + MOMENT_WEIGHT * valid_error[2]
            )
            if valid_total < best_valid:
                best_valid = valid_total
                best_logits = logits.detach().clone()
            print(
                f"nfe={nfe} moment={update} valid=({valid_error[0]:.6g}, "
                f"{valid_error[1]:.6g}) dvar={100 * valid_error[3]:+.3f}% "
                f"dphi4={100 * valid_error[4]:+.3f}%",
                flush=True,
            )

    learned = sigma_grid(best_logits, model.sigma_min, model.sigma_max).detach()
    final_fit = evaluate(model, fit, learned)
    final_valid = evaluate(model, valid, learned)
    print(
        f"nfe={nfe} learned fit={final_fit} valid={final_valid}",
        flush=True,
    )
    return learned, initial_fit, initial_valid, final_fit, final_valid


def generate(
    model, sigmas, output, checkpoint, grid_file, target_time, seed
):
    with np.load(INITIAL, allow_pickle=False) as archive:
        initial = np.asarray(archive["phi_gen"], dtype=np.float32)
    fields = torch.from_numpy(initial).to(DEVICE)
    n, lattice = fields.shape[0], fields.shape[-1]
    horizon = float(model.hparams.horizon)
    transitions = round(target_time / horizon)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as handle:
        chain = handle.create_dataset(
            "phi_gen",
            (transitions, n, lattice, lattice),
            dtype="f4",
            chunks=(1, n, lattice, lattice),
        )
        handle["times"] = horizon * np.arange(1, transitions + 1)
        handle["sigmas"] = sigmas.cpu().numpy()
        handle["sampling_seconds"] = np.empty(transitions)
        handle["max_abs_mean_increment"] = np.empty(transitions)
        handle.attrs.update(
            checkpoint=str(checkpoint),
            initial=str(INITIAL),
            horizon=horizon,
            solver="DPM-Solver++ 2M",
            nfe=len(sigmas) - 1,
            sigma_grid="learned",
            sigma_grid_file=str(grid_file),
            teacher_steps=128,
            seed=seed,
            L=lattice,
            n_samples=n,
            compile_mode="reduce-overhead",
        )

        for transition in range(transitions):
            torch.manual_seed(seed + transition + 1)
            torch.cuda.manual_seed_all(seed + transition + 1)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            max_mean = 0.0
            for start in range(0, n, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, n)
                condition = (
                    fields[start:stop, None] / float(model.norm_cond)
                ).to(memory_format=torch.channels_last)
                noise = model.project(torch.randn_like(condition))
                residual = dpmpp2m(model, condition, noise, sigmas)[:, 0]
                residual *= float(model.norm_res)
                max_mean = max(
                    max_mean,
                    residual.mean(dim=(-2, -1)).abs().max().item(),
                )
                fields[start:stop].add_(residual)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time
            chain[transition] = fields.cpu().numpy()
            handle["sampling_seconds"][transition] = seconds
            handle["max_abs_mean_increment"][transition] = max_mean
            handle.flush()
            print(
                f"nfe={len(sigmas) - 1} t={(transition + 1) * horizon:g} "
                f"{seconds:.3f} s max|mean(dphi)|={max_mean:.3e}",
                flush=True,
            )


def main(run=RUN, target_time=TARGET_TIME, seed=SEED):
    time_tag = f"{target_time:g}"
    checkpoint = run / "models/epoch=10000.ckpt"
    teacher = run / "samples" / f"hmc_L64_N512_t{time_tag}_heun128_pairs.h5"
    grid_file = run / "dpmpp2m_nfe32_sigmas.npz"
    output = (
        run / "samples"
        / f"hmc_L64_N512_t{time_tag}_dpmpp2m_nfe32_learned.h5"
    )
    existing = [path for path in (grid_file, output) if path.exists()]
    if existing:
        raise FileExistsError(f"output already exists: {existing}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    channels = list(state["hyper_parameters"].get("channels", CHANNELS))
    network = NCSNppCond2D(
        trivial_std,
        in_channels=2,
        channels=channels,
        use_attention=False,
        periodic=True,
    ).to(memory_format=torch.channels_last)
    model = ConditionalEDM2.load_from_checkpoint(
        checkpoint,
        network=network,
        map_location="cpu",
        weights_only=False,
    ).to(DEVICE).eval()
    model.ema.to(DEVICE)

    with model.ema.average_parameters():
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        model.net = torch.compile(
            model.net.to(memory_format=torch.channels_last),
            mode="reduce-overhead",
            dynamic=False,
        )
        if teacher.exists():
            with h5py.File(teacher, "r") as handle:
                if handle.attrs["checkpoint"] != str(checkpoint):
                    raise ValueError(f"wrong teacher checkpoint in {teacher}")
        else:
            generate_teacher(model, checkpoint, teacher, target_time, seed)
        network = model.net._orig_mod
        model.net = network
        torch.compiler.reset()
        torch.cuda.empty_cache()

        model.net = torch.compile(
            network,
            mode="reduce-overhead",
            dynamic=False,
        )
        fit, valid = teacher_pairs(model, teacher, seed)
        result = learn_grid(model, fit, valid, NFE)
        grid, initial_fit, initial_valid, final_fit, final_valid = result
        np.savez(
            grid_file,
            sigmas=grid.cpu().numpy(),
            errors=np.asarray(
                (*initial_fit, *initial_valid, *final_fit, *final_valid)
            ),
        )
        print(f"saved {grid_file}", flush=True)

        del fit, valid
        network = model.net._orig_mod
        model.net = network
        torch.compiler.reset()
        torch.cuda.empty_cache()
        model.net = torch.compile(
            network,
            mode="reduce-overhead",
            dynamic=False,
        )
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            generate(
                model, grid, output, checkpoint, grid_file, target_time, seed
            )
            print(f"saved {output}", flush=True)

    teacher.unlink()
    print(f"removed intermediate {teacher}", flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    main()
