# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Verification and ablation script for QYOLO26 (Quantum-Inspired Shared Channel Mixing on YOLO26)."""

from __future__ import annotations

import argparse
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import QMixBlock, SharedQuantumMixer
from ultralytics.utils.torch_utils import get_flops, get_num_params


def compute_gflops(model: torch.nn.Module, imgsz: int = 640) -> float:
    """Compute model GFLOPs at given imgsz, falling back to forward hooks if ultralytics-thop is unavailable."""
    flops = get_flops(model, imgsz=imgsz)
    if flops > 0:
        return flops

    total_macs = 0
    hooks = []

    def conv_hook(module: torch.nn.Conv2d, inp, out):
        nonlocal total_macs
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        _, c_out, h_out, w_out = out_t.shape
        k_h, k_w = module.kernel_size
        c_in = module.in_channels // module.groups
        total_macs += c_out * h_out * w_out * c_in * k_h * k_w

    def linear_hook(module: torch.nn.Linear, inp, out):
        nonlocal total_macs
        total_macs += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
        model(dummy)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return (total_macs * 2) / 1e9


def count_unique_params(modules: list[torch.nn.Module]) -> int:
    """Count unique parameters across a list of modules (accounting for shared parameter tensors)."""
    seen = set()
    total = 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def verify_entanglement_and_gradients(model: YOLO, scale_name: str) -> None:
    """Assert that layers 6 and 8 share the exact same SharedQuantumMixer and accumulate gradients properly."""
    seq = model.model.model
    layer6, layer8 = seq[6], seq[8]

    assert isinstance(layer6, QMixBlock), f"[{scale_name}] Layer 6 must be QMixBlock, got {type(layer6)}"
    assert isinstance(layer8, QMixBlock), f"[{scale_name}] Layer 8 must be QMixBlock, got {type(layer8)}"
    assert isinstance(layer6.shared_mixer, SharedQuantumMixer), f"[{scale_name}] Invalid shared_mixer type"
    assert layer6.shared_mixer is layer8.shared_mixer, (
        f"[{scale_name}] Layer 6 and Layer 8 do NOT share the same SharedQuantumMixer instance in memory!"
    )
    assert layer6.shared_mixer.w is layer8.shared_mixer.w, f"[{scale_name}] Shared parameter 'w' mismatch!"
    assert layer6.shared_mixer.theta is layer8.shared_mixer.theta, f"[{scale_name}] Shared parameter 'theta' mismatch!"

    # Verify forward + backward gradient accumulation on shared quantum parameters and dual-pool alpha
    model.model.zero_grad(set_to_none=True)
    model.model.train()
    dummy = torch.randn(2, 3, 256, 256, requires_grad=True)
    preds = model.model(dummy)

    # Sum all tensor outputs in the raw prediction dict/tuple to produce a scalar loss
    def _sum_tensors(obj) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj.sum()
        if isinstance(obj, dict):
            return sum((_sum_tensors(v) for v in obj.values()), torch.tensor(0.0))
        if isinstance(obj, (list, tuple)):
            return sum((_sum_tensors(v) for v in obj), torch.tensor(0.0))
        return torch.tensor(0.0)

    loss = _sum_tensors(preds)
    loss.backward()

    mixer = layer6.shared_mixer
    assert mixer.w.grad is not None, f"[{scale_name}] No gradient accumulated on shared_mixer.w"
    assert mixer.theta.grad is not None, f"[{scale_name}] No gradient accumulated on shared_mixer.theta"

    # Active dimensions up to max(layer6.r, layer8.r) must have non-zero gradients, while unused tail stays zero
    max_r = max(layer6.r, layer8.r)
    assert mixer.w.grad[:, :max_r].abs().sum().item() > 0, (
        f"[{scale_name}] Active slice [:, :{max_r}] of shared_mixer.w.grad is zero!"
    )
    assert mixer.theta.grad[:, :max_r].abs().sum().item() > 0, (
        f"[{scale_name}] Active slice [:, :{max_r}] of shared_mixer.theta.grad is zero!"
    )
    if max_r < mixer.max_dim:
        assert mixer.w.grad[:, max_r:].abs().sum().item() == 0.0, (
            f"[{scale_name}] Inactive slice [:, {max_r}:] of shared_mixer.w.grad should be zero!"
        )
    assert layer6.pool_alpha.grad is not None and layer8.pool_alpha.grad is not None, (
        f"[{scale_name}] Dual-pooling learnable alpha did not receive gradients!"
    )
    print(
        f"[OK] {scale_name}: SharedQuantumMixer entanglement verified "
        f"(id={id(mixer)}, r_P4={layer6.r}, r_P5={layer8.r}, "
        f"||grad_w[:{max_r}]||_1={mixer.w.grad[:, :max_r].abs().sum().item():.4e}, "
        f"||grad_theta[:{max_r}]||_1={mixer.theta.grad[:, :max_r].abs().sum().item():.4e})"
    )


def run_ablation() -> None:
    """Load YOLO26 and QYOLO26 (n/s), verify entanglement, and print the comparative ablation table."""
    pairs = [
        ("n", "yolo26n.yaml", "qyolo26n.yaml"),
        ("s", "yolo26s.yaml", "qyolo26s.yaml"),
    ]

    rows = []
    for scale, base_cfg, q_cfg in pairs:
        base_model = YOLO(base_cfg)
        q_model = YOLO(q_cfg)

        verify_entanglement_and_gradients(q_model, f"QYOLO26{scale}")

        base_total = get_num_params(base_model.model)
        q_total = get_num_params(q_model.model)

        base_l68 = count_unique_params([base_model.model.model[6], base_model.model.model[8]])
        q_l68 = count_unique_params([q_model.model.model[6], q_model.model.model[8]])

        base_flops = compute_gflops(base_model.model, imgsz=640)
        q_flops = compute_gflops(q_model.model, imgsz=640)

        total_red_pct = (1.0 - q_total / base_total) * 100.0
        l68_red_pct = (1.0 - q_l68 / base_l68) * 100.0
        flops_red_pct = (1.0 - q_flops / base_flops) * 100.0 if base_flops > 0 else 0.0

        rows.append(
            {
                "scale": scale,
                "base_cfg": f"YOLO26{scale}",
                "q_cfg": f"QYOLO26{scale}",
                "base_total": base_total,
                "q_total": q_total,
                "total_red_pct": total_red_pct,
                "base_l68": base_l68,
                "q_l68": q_l68,
                "l68_red_pct": l68_red_pct,
                "base_flops": base_flops,
                "q_flops": q_flops,
                "flops_red_pct": flops_red_pct,
            }
        )

    print("\n" + "=" * 112)
    print(f"{'QYOLO26 vs YOLO26 ABLATION & COMPLEXITY COMPARISON (imgsz=640)':^112}")
    print("=" * 112)
    header = (
        f"{'Model':<12} | {'Total Params':>14} | {'Delta Total':>11} | "
        f"{'Layers 6&8 Params':>17} | {'Delta L6&8':>11} | {'GFLOPs':>10} | {'Delta GFLOPs':>12}"
    )
    print(header)
    print("-" * 112)
    for r in rows:
        print(
            f"{r['base_cfg']:<12} | {r['base_total']:>14,d} | {'baseline':>11} | "
            f"{r['base_l68']:>17,d} | {'baseline':>11} | {r['base_flops']:>10.2f} | {'baseline':>12}"
        )
        print(
            f"{r['q_cfg']:<12} | {r['q_total']:>14,d} | {-r['total_red_pct']:>+10.2f}% | "
            f"{r['q_l68']:>17,d} | {-r['l68_red_pct']:>+10.2f}% | {r['q_flops']:>10.2f} | {-r['flops_red_pct']:>+11.2f}%"
        )
        print("-" * 112)


def main() -> None:
    """Entry point for QYOLO26 verification and optional short training test."""
    parser = argparse.ArgumentParser(description="Verify QYOLO26 architecture and optionally run a short training test.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run a short test training session after verification.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qyolo26n.yaml",
        help="Model YAML to use when --train is specified (default: qyolo26n.yaml).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco8.yaml",
        help="Dataset YAML for test training, e.g., coco8.yaml or VisDrone.yaml (default: coco8.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs when --train is enabled (default: 3).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640).",
    )
    args = parser.parse_args()

    run_ablation()

    if args.train:
        print(f"\n[TRAIN] Launching short training test: model={args.model}, data={args.data}, epochs={args.epochs}")
        model = YOLO(args.model)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
