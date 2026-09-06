"""Train EDM-preconditioned NCSN++ Model B kernels for 10000 epochs."""

from contextlib import nullcontext
from pathlib import Path
import sys

MODEL_B = Path(__file__).resolve().parents[1]
DM = MODEL_B.parent
sys.path[:0] = [str(DM), str(MODEL_B)]

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch_ema import ExponentialMovingAverage

from cond_lightning import ModelBPairDataModule, NCSNppCond2D
from edm_lightning import trivial_std


# The single intended training run.
LS = (8, 16, 32)
KAPPA = 0.34034
LAM = 1.0
HORIZON = 10.0
CHANNELS = (16, 32, 64, 128)
BATCH_SIZE = 2048
EPOCHS = 10_000
LR = 1e-3
P_MEAN = -1.2
P_STD = 1.2
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
EMA_DECAY = 0.9999
DEVICE = "cuda:0"

DATA = MODEL_B / "trainingdata"
COND_PATHS = [
    DATA / "modelB_L8_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080601.jld2",
    DATA / "modelB_L16_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080501.jld2",
    DATA / "modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2",
]
TARGET_PATHS = [
    DATA / "modelB_L8_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080602.jld2",
    DATA / "modelB_L16_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080502.jld2",
    DATA / "modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080102.jld2",
]
RUN_DIR = (
    MODEL_B
    / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_"
    "edm_ncsnpp_fahmc_noskip"
)


class ConditionalEDM2(pl.LightningModule):
    """EDM-preconditioned residual denoiser conditioned on the current field."""

    def __init__(
        self,
        network,
        sigma_data,
        channels=CHANNELS,
        lr=LR,
        p_mean=P_MEAN,
        p_std=P_STD,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        ema_decay=EMA_DECAY,
        norm_res=None,
        norm_cond=None,
        training_Ls=LS,
        kappa=KAPPA,
        lam=LAM,
        horizon=HORIZON,
        cond_paths=(),
        target_paths=(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.net = network
        self.sigma_data = float(sigma_data)
        self.lr = lr
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.norm_res = norm_res
        self.norm_cond = norm_cond
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=ema_decay)
        self._ema_parameters = tuple(self.net.parameters())

    @staticmethod
    def project(x):
        return x - x.mean(dim=(-2, -1), keepdim=True)

    def coefficients(self, sigma):
        sigma2 = sigma.square()
        data2 = self.sigma_data**2
        denominator = torch.sqrt(sigma2 + data2)
        return (
            1.0 / denominator,
            sigma * self.sigma_data / denominator,
            data2 / (sigma2 + data2),
            0.25 * torch.log(sigma),
        )

    @staticmethod
    def broadcast(x):
        return x[:, None, None, None]

    def denoise(self, noisy_residual, condition, sigma):
        c_in, c_out, c_skip, c_noise = self.coefficients(sigma)
        network_input = torch.cat(
            [self.broadcast(c_in) * noisy_residual, condition], dim=1
        )
        prediction = self.net(network_input, c_noise)
        denoised = (
            self.broadcast(c_skip) * noisy_residual
            + self.broadcast(c_out) * prediction
        )
        return self.project(denoised.float())

    def loss_fn(self, batch):
        residual = self.project(batch[:, :1])
        condition = batch[:, 1:]
        if self.training:
            sign = torch.randint(
                0, 2, (len(batch), 1, 1, 1), device=batch.device
            ).mul_(2).sub_(1)
            residual = residual * sign
            condition = condition * sign

        log_sigma = self.p_mean + self.p_std * torch.randn(
            len(batch), device=batch.device
        )
        sigma = log_sigma.exp()
        noise = self.project(torch.randn_like(residual))
        noisy = residual + self.broadcast(sigma) * noise
        c_in, c_out, c_skip, c_noise = self.coefficients(sigma)
        prediction = self.net(
            torch.cat([self.broadcast(c_in) * noisy, condition], dim=1),
            c_noise,
        )
        target = (
            residual - self.broadcast(c_skip) * noisy
        ) / self.broadcast(c_out)
        return (prediction - target).square().mean()

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def on_fit_start(self):
        self.ema.to(self.device)

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        decay = self.ema.decay
        if self.ema.num_updates is not None:
            self.ema.num_updates += 1
            decay = min(
                decay,
                (1 + self.ema.num_updates) / (10 + self.ema.num_updates),
            )
        torch._foreach_lerp_(
            self.ema.shadow_params, self._ema_parameters, 1.0 - decay
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("state_dict", {})
        checkpoint_compiled = any("net._orig_mod." in key for key in state)
        model_compiled = any("net._orig_mod." in key for key in self.state_dict())
        if checkpoint_compiled and not model_compiled:
            state = {
                key.replace("net._orig_mod.", "net.", 1): value
                for key, value in state.items()
            }
        elif model_compiled and not checkpoint_compiled:
            state = {
                key.replace("net.", "net._orig_mod.", 1): value
                for key, value in state.items()
            }
        checkpoint["state_dict"] = state
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    @torch.no_grad()
    def sample(self, condition, num_steps=32, rho=7.0, use_ema=True):
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        condition = condition.to(
            device=self.device, dtype=torch.float32, memory_format=torch.channels_last
        )
        batch = len(condition)
        step = torch.arange(num_steps, device=self.device, dtype=torch.float32)
        inv_rho = 1.0 / rho
        sigmas = (
            self.sigma_max**inv_rho
            + step / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ).pow(rho)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        residual = self.project(torch.randn_like(condition) * sigmas[0])

        ema = self.ema.average_parameters() if use_ema else nullcontext()
        with ema, torch.autocast(self.device.type, dtype=torch.bfloat16):
            for index in range(num_steps):
                sigma = sigmas[index]
                sigma_next = sigmas[index + 1]
                sigma_batch = sigma.expand(batch)
                denoised = self.denoise(residual, condition, sigma_batch)
                derivative = self.project((residual - denoised) / sigma)
                proposal = self.project(
                    residual + (sigma_next - sigma) * derivative
                )
                if index < num_steps - 1:
                    next_batch = sigma_next.expand(batch)
                    next_denoised = self.denoise(
                        proposal, condition, next_batch
                    )
                    next_derivative = self.project(
                        (proposal - next_denoised) / sigma_next
                    )
                    proposal = self.project(
                        residual
                        + 0.5 * (sigma_next - sigma)
                        * (derivative + next_derivative)
                    )
                residual = proposal
        return self.project(residual.float())


class SaveCheckpoints(Callback):
    def __init__(self, directory):
        self.directory = directory

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % 1000 == 0:
            trainer.save_checkpoint(self.directory / f"epoch={epoch:04d}.ckpt")
            previous = self.directory / f"epoch={epoch - 1000:04d}.ckpt"
            if epoch > 2000 and previous.exists():
                previous.unlink()


def main(
    run_dir=RUN_DIR,
    cond_paths=COND_PATHS,
    target_paths=TARGET_PATHS,
    horizon=HORIZON,
):
    models = run_dir / "models"
    checkpoints = list(models.glob("epoch=*.ckpt"))
    if checkpoints:
        checkpoint = max(
            checkpoints, key=lambda path: int(path.stem.split("=")[-1])
        )
        print(f"resuming from {checkpoint}")
    else:
        if run_dir.exists():
            raise FileExistsError(f"output exists without checkpoint: {run_dir}")
        models.mkdir(parents=True)
        checkpoint = None

    data = ModelBPairDataModule(
        [str(path) for path in cond_paths],
        [str(path) for path in target_paths],
        batch_size=BATCH_SIZE,
        device=DEVICE,
        expected_L=LS,
    )
    data.setup()
    for lattice, tensor in data.data_by_L.items():
        data.data_by_L[lattice] = tensor.to(memory_format=torch.channels_last)
    sigma_data = sum(
        tensor[:, 0].float().std(correction=0).item()
        for tensor in data.data_by_L.values()
    ) / len(data.data_by_L)

    network = NCSNppCond2D(
        trivial_std,
        in_channels=2,
        channels=list(CHANNELS),
        use_attention=False,
        periodic=True,
    ).to(memory_format=torch.channels_last)
    parameters = sum(parameter.numel() for parameter in network.parameters())
    print(f"sigma_data={sigma_data:.8f}, parameters={parameters / 1e6:.3f}M")
    network = torch.compile(network, mode="reduce-overhead", dynamic=False)
    model = ConditionalEDM2(
        network,
        sigma_data,
        horizon=horizon,
        norm_res=data.A_res,
        norm_cond=data.A_cond,
        cond_paths=tuple(map(str, cond_paths)),
        target_paths=tuple(map(str, target_paths)),
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=[int(DEVICE.split(":")[-1])],
        precision="bf16-mixed",
        callbacks=[SaveCheckpoints(models)],
        default_root_dir=run_dir,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=20,
    )
    trainer.fit(model, data, ckpt_path=checkpoint)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    main()
