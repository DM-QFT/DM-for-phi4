# DSM 损失分 t 段（UV/mid/IR）与分模随训练 epoch 的走势，判断红外是否还在改善。
# 训练时的 loss_UV/mid/IR 是按随机 t 落入 [0,0.2]/[0.2,0.8]/[0.8,1] 的抽样估计；这里改成
# 每段固定 t 网格、固定噪声、固定样本，各 checkpoint 输入完全相同，消掉抽样噪声。
# 残差 r=score·std+z 的功率按模分解，给出红外各模各自的贡献。
#   python analysis/loss_bins_vs_epoch.py <run 目录名> [epoch,epoch,...]
import functools
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from diffusion_lightning import marginal_prob_std  # noqa: E402
from cond_lightning import ConditionalDiffusionModel, NCSNppCond2D  # noqa: E402

L, DEV, N = 32, torch.device("cuda"), 4096
TD = ROOT / "trainingdata"
# 留出集与训练集同一个盒 A、不同随机种子，训练只见过后者
SETS = {
    "held": ("modelB_L32_Ne51200_corrgaussboxa_l1_kappa_0.340340_t0_seed2026091901.jld2",
             "modelB_L32_Ne51200_corrgaussboxa_l1_kappa_0.340340_t10_dt0.002_seed2026091903.jld2"),
    "train": ("modelB_L32_Ne102400_corrgaussboxa2_l1_kappa_0.340340_t0_seed2026092001.jld2",
              "modelB_L32_Ne102400_corrgaussboxa2_l1_kappa_0.340340_t10_dt0.002_seed2026092003.jld2"),
}
BINS = (("UV", np.linspace(0.02, 0.20, 4)), ("mid", np.linspace(0.25, 0.75, 4)),
        ("IR", np.linspace(0.82, 0.99, 4)))
MODES = ((1, 0), (1, 1), (2, 0), (2, 2), (4, 4))
RUN = ROOT / "runs" / sys.argv[1]
EPOCHS = [int(e) for e in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100, 300, 1000, 3000, 6000, 10000]

DATA = {}
for k, (fc, ft) in SETS.items():
    with h5py.File(TD / fc) as h:
        a = np.asarray(h["cfgs"][:N], dtype=np.float32)
    with h5py.File(TD / ft) as h:
        b = np.asarray(h["cfgs"][:N], dtype=np.float32)
    DATA[k] = (a, b)


def evaluate(ck, c, tg):
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    hp = raw["hyper_parameters"]
    ch = list(hp["channels"]) if hp.get("channels") else [16, 32, 64, 128]
    sm = NCSNppCond2D(functools.partial(marginal_prob_std, sigma=hp["sigma"]),
                      in_channels=hp["in_channels"], channels=ch)
    m = ConditionalDiffusionModel.load_from_checkpoint(
        str(ck), score_model=sm, map_location=DEV, weights_only=False).to(DEV).eval()
    x = torch.from_numpy(np.stack([(tg - c) / float(hp["norm_max"]), c / float(hp["cond_norm_max"])], 1)).to(DEV)
    out, ir_modes = {}, np.zeros(len(MODES))
    with torch.inference_mode(), m.ema.average_parameters():
        for tag, grid in BINS:
            acc = 0.0
            for b, tv in enumerate(grid):
                for s in range(0, N, 512):
                    xb = x[s:s + 512]
                    torch.manual_seed(7000 + 100 * b + s)      # 各 ckpt 同噪声
                    z = torch.randn_like(xb[:, :1])
                    tt = torch.full((len(xb),), float(tv), device=DEV)
                    std = m.marginal_prob_std_fn(tt).view(-1, 1, 1, 1)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        sc = m(torch.cat([xb[:, :1] + z * std, xb[:, 1:]], 1), tt)
                    r = (sc.float() * std + z)[:, 0]
                    acc += float((r ** 2).sum())
                    if tag == "IR":
                        F = torch.fft.fft2(r.double()) / L
                        ir_modes += np.array([float((F[:, i, j].abs() ** 2).sum()) for i, j in MODES])
            out[tag] = acc / (N * len(grid))
    return out, ir_modes / (N * len(BINS[2][1]))


print(f"run={RUN.name}  n={N}/集（train=模型见过的 102400 对前 {N} 条，held=同盒不同种子）")
print("epoch |      loss_UV         loss_mid          loss_IR     | IR 残差功率(held) "
      + " ".join(f"({i},{j})" for i, j in MODES))
print("      |  train    held    train    held    train    held   |")
for ep in EPOCHS:
    ck = RUN / "models" / f"epoch={ep:04d}.ckpt"
    if not ck.exists():
        cands = sorted(int(p.stem.split("=")[1]) for p in (RUN / "models").glob("epoch=*.ckpt"))
        ep = min(cands, key=lambda q: abs(q - ep)); ck = RUN / "models" / f"epoch={ep:04d}.ckpt"
    o = {k: evaluate(ck, *DATA[k]) for k in SETS}
    print(f"{ep:5d} | " + "  ".join(f"{o['train'][0][b]:7.3f} {o['held'][0][b]:7.3f}" for b in ("UV", "mid", "IR"))
          + "  | " + " ".join(f"{v:7.4f}" for v in o["held"][1]), flush=True)
