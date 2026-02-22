#!/usr/bin/env python3
"""Export demo triptychs into docs/assets/qualitative for README embedding."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export demo triptych PNGs to docs/assets/qualitative")
    parser.add_argument("--dataset", type=str, default="mipnerf360")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--src", type=str, default=None, help="Source triptych dir. Default: outputs/demo/<dataset>/<scene>/triptych")
    parser.add_argument("--dst", type=str, default=None, help="Destination dir. Default: docs/assets/qualitative/<dataset>/<scene>")
    parser.add_argument("--max_images", type=int, default=4)
    return parser.parse_args()


def _list_pngs(src: Path) -> List[Path]:
    files = sorted(src.glob("view_*.png"))
    if not files:
        files = sorted(src.glob("*.png"))
    if not files:
        raise RuntimeError(f"No PNG files found in {src}")
    return files


def main() -> None:
    args = parse_args()

    src = Path(args.src) if args.src else (ROOT / "outputs" / "demo" / args.dataset / args.scene / "triptych")
    dst = Path(args.dst) if args.dst else (ROOT / "docs" / "assets" / "qualitative" / args.dataset / args.scene)

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source directory missing: {src}")

    images = _list_pngs(src)[: max(1, int(args.max_images))]

    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.png"):
        old.unlink()

    for idx, image in enumerate(images):
        shutil.copy2(image, dst / f"view_{idx:03d}.png")

    print(f"[done] exported {len(images)} image(s)")
    print(f" - source: {src}")
    print(f" - dest:   {dst}")


if __name__ == "__main__":
    main()
