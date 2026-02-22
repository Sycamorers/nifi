#!/usr/bin/env python3
"""Copy a stable subset of qualitative example mosaics into docs/assets for README embedding."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export qualitative example mosaics to docs/assets")
    parser.add_argument("--src_dir", type=str, required=True, help="Source mosaic directory")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination docs/assets directory")
    parser.add_argument("--max_images", type=int, default=4)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"--src_dir does not exist or is not a directory: {src_dir}")

    mosaics = sorted(src_dir.glob("*.png"))
    if not mosaics:
        raise RuntimeError(f"No PNG mosaics found in {src_dir}")

    selected = mosaics[: max(1, int(args.max_images))]

    dst_dir.mkdir(parents=True, exist_ok=True)
    for existing in dst_dir.glob("*.png"):
        existing.unlink()

    for src in selected:
        shutil.copy2(src, dst_dir / src.name)

    print(f"[done] copied {len(selected)} image(s) to {dst_dir}")
    for path in selected:
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
