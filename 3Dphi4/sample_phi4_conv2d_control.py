"""Sample a direct-score Conv2d-only/native 3D diagnostic checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT3D = Path(__file__).resolve().parent
REPO = ROOT3D.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT3D))

from diffusion_lightning import DiffusionModel, marginal_prob_std  # noqa: E402
from networks_3d_conv2d import (  # noqa: E402
    NCSNpp3DConv2d,
    convert_conv2d_wrappers_to_native,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--schedule", default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    run_dir = checkpoint.parent.parent
    with (run_dir / "training_config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    backend = config["convolution_backend"]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = saved["hyper_parameters"]
    sigma = float(hp["sigma"])
    norm_min, norm_max = float(hp["norm_min"]), float(hp["norm_max"])
    channels = tuple(int(channel) for channel in config.get("channels", (4, 8, 16, 32)))
    final_bias = bool(config.get("final_convolution_bias", False))
    score_model = NCSNpp3DConv2d(
        lambda t: marginal_prob_std(t, sigma=sigma),
        channels=channels,
        final_bias=final_bias,
    )
    if backend == "native_conv3d":
        score_model = convert_conv2d_wrappers_to_native(score_model)
    elif backend != "conv2d_only":
        raise ValueError(f"unknown convolution backend {backend}")

    model = DiffusionModel.load_from_checkpoint(
        checkpoint, score_model=score_model
    )
    if not args.no_compile:
        model.score_model = torch.compile(model.score_model)
    model = model.cuda().eval()
    model.ema.to("cuda")

    torch.manual_seed(args.seed)
    start = time.time()
    samples = model.sample(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        schedule=args.schedule,
        use_ema=True,
    )
    elapsed = time.time() - start
    physical = (
        (samples.float().cpu().numpy() + 1.0)
        * (norm_max - norm_min)
        / 2.0
        + norm_min
    )[:, 0]
    physical = np.moveaxis(physical, 0, -1)

    match = re.search(r"epoch=(\d+)", checkpoint.stem)
    epoch = f"{int(match.group(1)):04d}" if match else checkpoint.stem
    output_dir = run_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"samples_em_steps{args.num_steps}_{args.schedule}_"
        f"epoch={epoch}_seed{args.seed}.npy"
    )
    np.save(output, physical)
    print(
        f"saved {physical.shape} to {output} in {elapsed:.1f}s "
        f"({backend}, EMA)",
        flush=True,
    )


if __name__ == "__main__":
    main()
