"""Run an actual CUDA forward/backward pass and print a JSON report."""

from __future__ import annotations

import json
import time

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    x = torch.randn(2048, 2048, device=device, requires_grad=True)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    loss = (x @ x.transpose(0, 1)).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    duration_ms = (time.perf_counter() - start) * 1000.0

    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": properties.name,
        "total_vram_gib": round(properties.total_memory / 1024**3, 2),
        "forward_backward_ms": round(duration_ms, 3),
        "loss": float(loss.detach().cpu()),
        "gradient_finite": bool(torch.isfinite(x.grad).all().item()),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
    }
    if not report["gradient_finite"]:
        raise FloatingPointError("CUDA smoke test produced a non-finite gradient")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
