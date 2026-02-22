#!/usr/bin/env python3
"""Create labeled [HQ | Compressed | Restored] triptych PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from nifi.utils.triptych import save_labeled_triptych


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labeled HQ/Compressed/Restored triptych images")
    parser.add_argument("--hq_dir", type=str, required=True, help="Directory containing HQ PNG files")
    parser.add_argument("--lq_dir", type=str, required=True, help="Directory containing compressed/LQ PNG files")
    parser.add_argument("--restored_dir", type=str, required=True, help="Directory containing restored PNG files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for triptych PNG files")
    parser.add_argument("--scene", type=str, required=True, help="Scene label for title bar")
    parser.add_argument("--rate", type=str, required=True, help="Rate label for title bar")
    parser.add_argument("--pattern", type=str, default="view_*.png", help="File name glob pattern (default: view_*.png)")
    return parser.parse_args()


def _sorted_pngs(path: Path, pattern: str) -> List[Path]:
    files = sorted(path.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched pattern '{pattern}' in {path}")
    return files


def main() -> None:
    args = parse_args()

    hq_dir = Path(args.hq_dir)
    lq_dir = Path(args.lq_dir)
    restored_dir = Path(args.restored_dir)
    out_dir = Path(args.out_dir)

    if not hq_dir.exists() or not lq_dir.exists() or not restored_dir.exists():
        raise FileNotFoundError("One or more input directories do not exist")

    hq_files = _sorted_pngs(hq_dir, args.pattern)
    out_dir.mkdir(parents=True, exist_ok=True)

    for hq_path in hq_files:
        name = hq_path.name
        lq_path = lq_dir / name
        restored_path = restored_dir / name
        if not lq_path.exists():
            raise FileNotFoundError(f"Missing LQ image for {name}: {lq_path}")
        if not restored_path.exists():
            raise FileNotFoundError(f"Missing restored image for {name}: {restored_path}")

        title = f"Scene: {args.scene} | Rate: {args.rate} | View: {name}"
        save_labeled_triptych(
            hq_path=hq_path,
            lq_path=lq_path,
            restored_path=restored_path,
            out_path=out_dir / name,
            title=title,
        )

    print(f"[done] wrote {len(hq_files)} triptych image(s) to {out_dir}")


if __name__ == "__main__":
    main()
