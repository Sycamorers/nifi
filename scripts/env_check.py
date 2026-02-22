#!/usr/bin/env python3
"""Environment and GPU sanity check for NiFi runs."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict

import torch


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))


def _collect() -> Dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "torch_cuda_version": str(torch.version.cuda),
        "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu_name": gpu_name,
        "diffusers_version": _module_version("diffusers"),
        "transformers_version": _module_version("transformers"),
        "xformers_version": _module_version("xformers"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print and persist environment metadata")
    parser.add_argument("--out", type=str, default="logs/env_check.txt", help="Output text log path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _collect()

    lines = [
        f"python_version: {payload['python_version']}",
        f"python_executable: {payload['python_executable']}",
        f"platform: {payload['platform']}",
        f"torch_version: {payload['torch_version']}",
        f"cuda_available: {payload['cuda_available']}",
        f"torch_cuda_version: {payload['torch_cuda_version']}",
        f"gpu_count: {payload['gpu_count']}",
        f"gpu_name: {payload['gpu_name']}",
        f"diffusers_version: {payload['diffusers_version']}",
        f"transformers_version: {payload['transformers_version']}",
        f"xformers_version: {payload['xformers_version']}",
    ]

    text = "\n".join(lines) + "\n\njson:\n" + json.dumps(payload, indent=2) + "\n"
    print(text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()

