"""Network checks in float64 on CPU: shapes, permutation, batch vs single, translation and box symmetries, locality, grid-phase sensitivity."""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ParticleUNet, ParticleGNN, collate

torch.manual_seed(0)
rng = np.random.default_rng(0)
dt = torch.float64
torch.set_default_dtype(dt)


def run(model, x0s, zs, Ls, sigma, shift):
    pos, typ, batch = collate(x0s, zs)
    return model(pos, typ, batch, Ls, torch.tensor(sigma, dtype=dt), torch.tensor(shift, dtype=dt))


def rand_event(L, N, M):
    return rng.random((N, 2)) * L, rng.random((M, 2)) * L


def randomize(model):
    # the zero-initialised residual layers would make every output exactly zero, structure tests need generic weights
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return model.to(dt)


for name, model in (("unet", randomize(ParticleUNet(C=(16, 32, 64), V=(4, 8, 16), n_blocks=1))), ("gnn", randomize(ParticleGNN(C=16, V=4, n_blocks=2)))):
    print(f"== {name}: {sum(p.numel() for p in model.parameters())} parameters")

    # shapes with mixed (N, M, L), including the smallest training box L = 8
    Ls = [12.0, 16.0, 14.0, 8.0]
    ev = [rand_event(12, 50, 60), rand_event(16, 100, 80), rand_event(14, 70, 70), rand_event(8, 30, 40)]
    x0s, zs = [e[0] for e in ev], [e[1] for e in ev]
    sig, shift = [0.3, 2.0, 20.0, 1.0], rng.random((4, 2))
    out = run(model, x0s, zs, Ls, sig, shift)
    assert out.shape == (60 + 80 + 70 + 40, 2) and torch.isfinite(out).all() and out.abs().max() > 0
    print("shapes ok, output rms", out.pow(2).mean().sqrt().item())

    # batch vs single
    off = 0
    for b in range(4):
        single = run(model, [x0s[b]], [zs[b]], [Ls[b]], [sig[b]], shift[b:b + 1])
        assert torch.allclose(single, out[off:off + len(zs[b])], atol=1e-10), (single - out[off:off + len(zs[b])]).abs().max()
        off += len(zs[b])
    print("batch vs single ok")

    # permutation within an event permutes the outputs
    p0, p1 = rng.permutation(50), rng.permutation(60)
    perm = run(model, [x0s[0][p0]], [zs[0][p1]], [Ls[0]], [sig[0]], shift[:1])
    assert torch.allclose(perm, out[:60][p1], atol=1e-10)
    print("permutation ok")

    # translation (with the grid translated along) including wrap across the boundary
    t = np.array([5.3, -2.1])
    tr = run(model, [(x0s[0] + t) % 12], [(zs[0] + t) % 12], [Ls[0]], [sig[0]], shift[:1] + t)
    assert torch.allclose(tr, out[:60], atol=1e-10), (tr - out[:60]).abs().max()
    print("translation ok")

    # box symmetries with zero grid shift: rotation by 90 degrees and reflection map the grid onto itself
    z0 = np.zeros((1, 2))
    base = run(model, [x0s[0]], [zs[0]], [Ls[0]], [sig[0]], z0)
    rot = lambda x: np.stack([12 - x[:, 1], x[:, 0]], 1) % 12
    r = run(model, [rot(x0s[0])], [rot(zs[0])], [Ls[0]], [sig[0]], z0)
    assert torch.allclose(r, torch.stack([-base[:, 1], base[:, 0]], 1), atol=1e-10)
    ref = lambda x: np.stack([12 - x[:, 0], x[:, 1]], 1) % 12
    r = run(model, [ref(x0s[0])], [ref(zs[0])], [Ls[0]], [sig[0]], z0)
    assert torch.allclose(r, torch.stack([-base[:, 0], base[:, 1]], 1), atol=1e-10)
    print("rotation and reflection ok")

    # gradients finite
    out.pow(2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print("gradients ok")

# locality of the U-Net: receptive field radius 30 (29 from radii and 2 blocks per level, 1 from the count feature), so use L = 64
model = randomize(ParticleUNet(C=(16, 32, 64), V=(4, 8, 16), n_blocks=2))
L = 64.0
x0, z = rand_event(64, 150, 150)
x0[0], z[0] = (10.0, 10.0), (10.5, 10.2)      # probed output particle z[0] and a nearby condition particle
x0[1] = (45.0, 45.0)                           # far particle: minimum-image distance to (10,10) is sqrt(2)*29 > 29
shift = rng.random((1, 2))
base = run(model, [x0], [z], [L], [1.0], shift)
x0_far = x0.copy()
x0_far[1] = (40.0, 38.0)                       # still farther than 29 from z[0]
moved = run(model, [x0_far], [z], [L], [1.0], shift)
assert torch.allclose(moved[0], base[0], atol=1e-10), (moved[0] - base[0]).abs()
x0_near = x0.copy()
x0_near[0] = (10.2, 9.7)
moved = run(model, [x0_near], [z], [L], [1.0], shift)
assert not torch.allclose(moved[0], base[0], atol=1e-6)
print("locality ok: far move changes nothing, near move changes the output")

# grid-phase sensitivity (reported, not asserted)
other = run(model, [x0], [z], [L], [1.0], rng.random((1, 2)))
print(f"grid phase sensitivity: rms change {(other - base).pow(2).mean().sqrt().item():.3e} vs output rms {base.pow(2).mean().sqrt().item():.3e}")
print("ALL MODEL TESTS PASSED")
