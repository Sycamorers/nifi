#!/usr/bin/env python3
"""Garden GPU environment sanity check."""

from __future__ import annotations

import json
import time

import torch


def main() -> None:
    t0 = time.perf_counter()
    cuda_ok = bool(torch.cuda.is_available())
    payload = {
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_ok,
        "torch_cuda_version": str(torch.version.cuda),
    }

    if not cuda_ok:
        payload["status"] = "fail"
        payload["reason"] = "CUDA is unavailable"
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(1024, 1024, device=device)
    y = (x @ x.t()).mean()
    _ = float(y.item())
    torch.cuda.synchronize(device)

    payload.update(
        {
            "status": "ok",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "memory_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_sec": float(time.perf_counter() - t0),
        }
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
