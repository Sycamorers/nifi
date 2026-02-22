#!/usr/bin/env python3
"""Fetch one real benchmark scene for qualitative demo reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.benchmark.download import download_mipnerf360, download_tandt_deepblending_bundle
from nifi.benchmark.registry import DEEPBLENDING_SCENES, MIPNERF360_SCENES, TANKS_AND_TEMPLES_SCENES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a real dataset scene used in NiFi paper evaluation")
    parser.add_argument("--dataset", type=str, default="mipnerf360", choices=["mipnerf360", "deepblending", "tanks_temples"])
    parser.add_argument("--scene", type=str, default=None, help="Scene name (dataset default if omitted)")
    parser.add_argument("--out", type=str, default="data", help="Data root")
    parser.add_argument("--remove_zip", action="store_true", help="Delete zip archives after extraction")
    return parser.parse_args()


def _has_images(scene_dir: Path) -> bool:
    for d in ("images", "images_4", "images_2", "rgb", "train"):
        p = scene_dir / d
        if p.exists() and any(p.glob("*")):
            return True
    return False


def _resolve_bundle_scene(bundle_root: Path, scene: str) -> Path:
    direct = bundle_root / scene
    if direct.exists() and direct.is_dir():
        return direct

    for p in bundle_root.rglob(scene):
        if p.is_dir() and _has_images(p):
            return p

    raise FileNotFoundError(f"Scene '{scene}' not found under extracted bundle: {bundle_root}")


def _print_structure(scene_dir: Path) -> None:
    print(f"[done] scene ready: {scene_dir}")
    for d in ("images", "images_4", "images_2", "rgb", "train", "sparse"):
        p = scene_dir / d
        if p.exists():
            print(f" - {p}")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mipnerf360":
        scene = args.scene or "garden"
        if scene not in MIPNERF360_SCENES:
            raise ValueError(f"Unknown Mip-NeRF360 scene '{scene}'. Allowed: {MIPNERF360_SCENES}")

        scene_dir = out_root / "mipnerf360" / scene
        if _has_images(scene_dir):
            print(f"[info] already exists, skipping download: {scene_dir}")
            _print_structure(scene_dir)
            return

        download_mipnerf360(out_root=out_root, scenes=[scene], remove_zip=bool(args.remove_zip))
        if not _has_images(scene_dir):
            raise RuntimeError(f"Download finished but expected images were not found in {scene_dir}")
        _print_structure(scene_dir)
        return

    scene_choices = DEEPBLENDING_SCENES if args.dataset == "deepblending" else TANKS_AND_TEMPLES_SCENES
    scene = args.scene or scene_choices[0]
    if scene not in scene_choices:
        raise ValueError(f"Unknown scene '{scene}' for dataset '{args.dataset}'. Allowed: {scene_choices}")

    target_scene_dir = out_root / args.dataset / scene
    if _has_images(target_scene_dir):
        print(f"[info] already exists, skipping download: {target_scene_dir}")
        _print_structure(target_scene_dir)
        return

    bundle_root = download_tandt_deepblending_bundle(out_root=out_root, remove_zip=bool(args.remove_zip))
    source_scene_dir = _resolve_bundle_scene(bundle_root, scene)

    target_scene_dir.parent.mkdir(parents=True, exist_ok=True)
    if not target_scene_dir.exists():
        # copytree here is intentionally explicit to keep idempotence and stable output path
        import shutil

        shutil.copytree(source_scene_dir, target_scene_dir)

    _print_structure(target_scene_dir)


if __name__ == "__main__":
    main()
