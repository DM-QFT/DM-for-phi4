"""
Minimal NCSNpp2D forward profile for H100 vs 5090 comparison.

Run from 2Dphi4/:
    python profile_ncsnpp.py --device cuda:0 --batch 1024 --L 128 --sigma 640
"""
import sys
sys.path.append("..")

import argparse
import functools
import torch
from torch.profiler import profile, ProfilerActivity

from networks import NCSNpp2D
from diffusion_lightning import marginal_prob_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--sigma", type=float, default=640.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no_autocast", action="store_true",
                   help="Disable bf16 autocast (default: enabled, matches dm.sample())")
    p.add_argument("--rows", type=int, default=25, help="Rows to show in the table")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    mps = functools.partial(marginal_prob_std, sigma=args.sigma)
    net = NCSNpp2D(mps).to(device).eval()

    x = torch.randn(args.batch, 1, args.L, args.L, device=device)
    t = torch.full((args.batch,), 0.5, device=device)

    autocast_ctx = (torch.autocast(device.type, dtype=torch.bfloat16)
                    if not args.no_autocast else torch.amp.autocast(device.type, enabled=False))

    print(f"device={device}  batch={args.batch}  L={args.L}  sigma={args.sigma}")
    print(f"autocast bf16: {not args.no_autocast}")
    print(f"warmup={args.warmup} iters / profile={args.iters} iters")
    print()

    # ── warmup ───────────────────────────────────────────────────────────
    with torch.no_grad(), autocast_ctx:
        for _ in range(args.warmup):
            _ = net(x, t)
        torch.cuda.synchronize()

    # ── timed wall-clock first ───────────────────────────────────────────
    import time
    with torch.no_grad(), autocast_ctx:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            _ = net(x, t)
        torch.cuda.synchronize()
        wall = (time.time() - t0) / args.iters * 1000
    print(f"wall-clock per forward: {wall:.2f} ms  → {1000/wall:.2f} forwards/s")
    print()

    # ── profile ──────────────────────────────────────────────────────────
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        with torch.no_grad(), autocast_ctx:
            for _ in range(args.iters):
                _ = net(x, t)
            torch.cuda.synchronize()

    sort_key = ("self_device_time_total"
                if hasattr(next(iter(prof.key_averages())), "self_device_time_total")
                else "self_cuda_time_total")
    print(prof.key_averages().table(
        sort_by=sort_key, row_limit=args.rows))

    # Rough TC vs non-TC breakdown
    tc_keys = ["convolution", "conv2d", "cudnn", "gemm", "addmm", "linear"]
    nontc_keys = ["group_norm", "silu", "add", "mul", "div",
                  "upsample", "cat", "interpolate", "to_copy", "copy"]
    cuda_tot = 0.0
    tc = 0.0
    nontc = 0.0
    for ev in prof.key_averages():
        ct = getattr(ev, "self_device_time_total",
                     getattr(ev, "self_cuda_time_total", 0.0))
        cuda_tot += ct
        name = ev.key.lower()
        if any(k in name for k in tc_keys):
            tc += ct
        elif any(k in name for k in nontc_keys):
            nontc += ct
    if cuda_tot > 0:
        print()
        print(f"GPU time breakdown (self_cuda):")
        print(f"  Tensor-Core-ish (conv/linear/gemm): {tc/cuda_tot*100:5.1f}%  ({tc/1e3:.1f} us)")
        print(f"  Non-TC (norm/act/add/upsample/...): {nontc/cuda_tot*100:5.1f}%  ({nontc/1e3:.1f} us)")
        print(f"  Other / launch / overhead         : {(cuda_tot-tc-nontc)/cuda_tot*100:5.1f}%")
        print(f"  Total self_cuda_time              : {cuda_tot/1e3:.1f} us/iter (× {args.iters} iters)")


if __name__ == "__main__":
    main()
