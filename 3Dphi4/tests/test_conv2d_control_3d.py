"""Correctness tests for the Conv2d-only 3D control network."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT3D = Path(__file__).resolve().parents[1]
REPO = ROOT3D.parent
sys.path.insert(0, str(ROOT3D))
sys.path.insert(0, str(REPO))

from diffusion_lightning import DiffusionModel, marginal_prob_std  # noqa: E402
from networks_3d_conv2d import (  # noqa: E402
    Conv3dCircularVia2d,
    NCSNpp3DConv2d,
    convert_conv2d_wrappers_to_native,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _brute_periodic_conv3d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    result = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shifted = torch.roll(x, (-i, -j, -k), dims=(2, 3, 4))
                tap = weight[:, :, i + 1, j + 1, k + 1]
                term = torch.einsum("oi,bitxy->botxy", tap, shifted)
                result = term if result is None else result + term
    if bias is not None:
        result = result + bias.view(1, -1, 1, 1, 1)
    return result


def _native_weight(module: Conv3dCircularVia2d) -> torch.Tensor:
    out_ch, stacked_ch, ky, kx = module.conv.weight.shape
    in_ch = stacked_ch // 3
    return (
        module.conv.weight.reshape(out_ch, 3, in_ch, ky, kx)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def test_conv2d_only_and_exact_forward() -> None:
    torch.manual_seed(1)
    for stride in (1, 2):
        layer = Conv3dCircularVia2d(
            2, 3, 3, stride=stride, padding=1, bias=True
        ).double().to(DEVICE)
        shape = (2, 2, 7, 8, 9) if stride == 1 else (2, 2, 8, 10, 12)
        x = torch.randn(shape, dtype=torch.float64, device=DEVICE)
        reference = _brute_periodic_conv3d(
            x, _native_weight(layer), layer.conv.bias
        )
        if stride == 2:
            reference = reference[:, :, ::2, ::2, ::2]
        error = (layer(x) - reference).abs().max().item()
        assert error < 1.0e-12, error
        assert not any(isinstance(m, nn.Conv3d) for m in layer.modules())


def test_native_conv3d_weight_mapping_and_gradients() -> None:
    torch.manual_seed(2)
    for stride in (1, 2):
        via = Conv3dCircularVia2d(
            2, 3, 3, stride=stride, padding=1, bias=True
        ).double().to(DEVICE)
        native = nn.Conv3d(
            2,
            3,
            3,
            stride=stride,
            padding=1,
            padding_mode="circular",
            bias=True,
        ).double().to(DEVICE)
        with torch.no_grad():
            native.weight.copy_(_native_weight(via))
            native.bias.copy_(via.conv.bias)
        x1 = torch.randn(
            2, 2, 8, 10, 12, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        x2 = x1.detach().clone().requires_grad_(True)
        y1, y2 = via(x1), native(x2)
        assert (y1 - y2).abs().max().item() < 1.0e-12
        probe = torch.randn_like(y1)
        (y1 * probe).sum().backward()
        (y2 * probe).sum().backward()
        assert (x1.grad - x2.grad).abs().max().item() < 1.0e-11
        mapped_native_grad = (
            native.weight.grad.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape_as(via.conv.weight.grad)
        )
        assert (
            via.conv.weight.grad - mapped_native_grad
        ).abs().max().item() < 1.0e-10
        assert (
            via.conv.bias.grad - native.bias.grad
        ).abs().max().item() < 1.0e-10


def test_axis_impulse_and_batch_isolation() -> None:
    layer = Conv3dCircularVia2d(1, 1, 3, padding=1, bias=False).double()
    with torch.no_grad():
        layer.conv.weight.zero_()
        weight = _native_weight(layer).clone()
        weight[0, 0, 0, 1, 1] = 1.0
        weight[0, 0, 1, 0, 1] = 2.0
        weight[0, 0, 1, 1, 0] = 3.0
        layer.conv.weight.copy_(
            weight.permute(0, 2, 1, 3, 4).reshape_as(layer.conv.weight)
        )
    x = torch.zeros(2, 1, 5, 6, 7, dtype=torch.float64)
    x[0, 0, 0, 0, 0] = 1.0
    y = layer(x)
    expected = _brute_periodic_conv3d(x, weight, None)
    assert torch.equal(y, expected)
    assert y[1].abs().max().item() == 0.0


def test_network_audit_and_dsm_backward() -> None:
    torch.manual_seed(3)
    std_fn = lambda t: marginal_prob_std(t, sigma=100.0)
    network = NCSNpp3DConv2d(std_fn).to(DEVICE)
    assert sum(p.numel() for p in network.parameters()) == 552_528
    assert not any(isinstance(m, nn.Conv3d) for m in network.modules())
    assert network.conv_out.conv.bias is None
    x = torch.randn(2, 1, 8, 8, 8, device=DEVICE)
    t = torch.tensor([0.2, 0.8], device=DEVICE)
    y = network(x, t)
    assert y.shape == x.shape and torch.isfinite(y).all()

    model = DiffusionModel(
        network,
        sigma=100.0,
        L=8,
        spatial_dims=3,
        zero_mode_loss_weight=0.0,
    ).to(DEVICE)
    loss, _, _ = model.loss_fn(x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_full_network_native_parity() -> None:
    torch.manual_seed(31)
    std_fn = lambda t: marginal_prob_std(t, sigma=100.0)
    via = NCSNpp3DConv2d(std_fn).double().to(DEVICE).eval()
    native = convert_conv2d_wrappers_to_native(copy.deepcopy(via)).eval()
    assert not any(isinstance(m, nn.Conv3d) for m in via.modules())
    assert not any(isinstance(m, nn.Conv2d) for m in native.modules())
    x1 = torch.randn(
        1, 1, 8, 8, 8, dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    x2 = x1.detach().clone().requires_grad_(True)
    t = torch.tensor([0.37], dtype=torch.float64, device=DEVICE)
    y1, y2 = via(x1, t), native(x2, t)
    scale = y1.abs().max().clamp_min(1.0)
    assert ((y1 - y2).abs().max() / scale).item() < 1.0e-11
    probe = torch.randn_like(y1)
    (y1 * probe).sum().backward()
    (y2 * probe).sum().backward()
    grad_scale = x1.grad.abs().max().clamp_min(1.0)
    assert ((x1.grad - x2.grad).abs().max() / grad_scale).item() < 1.0e-10


def test_reference_width_configuration() -> None:
    torch.manual_seed(32)
    std_fn = lambda t: marginal_prob_std(t, sigma=100.0)
    via = NCSNpp3DConv2d(
        std_fn,
        channels=(10, 20, 40, 80),
        final_bias=True,
    ).double().to(DEVICE).eval()
    assert sum(parameter.numel() for parameter in via.parameters()) == 2_836_021
    assert via.conv_out.conv.bias is not None
    assert not any(isinstance(module, nn.Conv3d) for module in via.modules())

    native = convert_conv2d_wrappers_to_native(copy.deepcopy(via)).eval()
    assert sum(parameter.numel() for parameter in native.parameters()) == 2_836_021
    x = torch.randn(1, 1, 8, 8, 8, dtype=torch.float64, device=DEVICE)
    t = torch.tensor([0.41], dtype=torch.float64, device=DEVICE)
    with torch.no_grad():
        y_via, y_native = via(x, t), native(x, t)
    scale = y_native.abs().max().clamp_min(1.0)
    assert ((y_via - y_native).abs().max() / scale).item() < 1.0e-11


def test_mod8_translation_equivariance() -> None:
    torch.manual_seed(4)
    std_fn = lambda t: marginal_prob_std(t, sigma=100.0)
    network = NCSNpp3DConv2d(std_fn).double().to(DEVICE).eval()
    x = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64, device=DEVICE)
    t = torch.tensor([0.4], dtype=torch.float64, device=DEVICE)
    shift = (8, 0, 8)
    with torch.no_grad():
        y = network(x, t)
        shifted_y = network(torch.roll(x, shift, dims=(2, 3, 4)), t)
    expected = torch.roll(y, shift, dims=(2, 3, 4))
    relative = (shifted_y - expected).abs().max() / y.abs().max().clamp_min(1e-12)
    assert relative.item() < 1.0e-10, relative.item()


if __name__ == "__main__":
    tests = [
        test_conv2d_only_and_exact_forward,
        test_native_conv3d_weight_mapping_and_gradients,
        test_axis_impulse_and_batch_isolation,
        test_network_audit_and_dsm_backward,
        test_full_network_native_parity,
        test_reference_width_configuration,
        test_mod8_translation_equivariance,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
    print("ALL CONV2D-ONLY 3D TESTS PASSED")
