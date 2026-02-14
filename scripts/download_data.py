#!/usr/bin/env python3
import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MIPNERF360_BASE = "https://storage.googleapis.com/gresearch/refraw360"
DEFAULT_MIPNERF360_SCENES = ["garden"]
ALL_MIPNERF360_SCENES = [
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
]



def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with out_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))



def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)



def make_tiny_subset(scene_dir: Path, max_images: int) -> None:
    candidates = [scene_dir / "images_4", scene_dir / "images_2", scene_dir / "images"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print(f"[warn] Could not find image folder for tiny subset in {scene_dir}")
        return

    dst = scene_dir / "images_tiny"
    dst.mkdir(parents=True, exist_ok=True)

    patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    images = []
    for pat in patterns:
        images.extend(sorted(src.glob(pat)))
    for img in images[:max_images]:
        shutil.copy2(img, dst / img.name)



def handle_mipnerf360(args: argparse.Namespace) -> None:
    out_root = Path(args.out) / "mipnerf360"
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = args.scenes
    if not scenes:
        scenes = DEFAULT_MIPNERF360_SCENES
    if "all" in scenes:
        scenes = ALL_MIPNERF360_SCENES

    for scene in scenes:
        if scene not in ALL_MIPNERF360_SCENES:
            raise ValueError(f"Unknown Mip-NeRF360 scene: {scene}")

        url = f"{MIPNERF360_BASE}/{scene}.zip"
        zip_path = out_root / f"{scene}.zip"
        print(f"[info] downloading {scene} from {url}")
        download_file(url, zip_path)

        print(f"[info] extracting {zip_path}")
        extract_zip(zip_path, out_root)

        if args.remove_zip:
            zip_path.unlink(missing_ok=True)

        if args.max_images is not None:
            make_tiny_subset(out_root / scene, args.max_images)

    print(f"[done] Mip-NeRF360 data prepared at: {out_root}")



def handle_dl3dv(args: argparse.Namespace) -> None:
    out_root = Path(args.out) / "dl3dv"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dl3dv_source is None:
        print(
            "[info] DL3DV requires manual access. Download a subset locally, then rerun with "
            "--dl3dv_source /path/to/DL3DV_subset"
        )
        return

    src = Path(args.dl3dv_source)
    if not src.exists():
        raise FileNotFoundError(f"DL3DV source path does not exist: {src}")

    dst = out_root / src.name
    if dst.exists() and args.skip_existing:
        print(f"[info] destination exists, skipping: {dst}")
    else:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    print(f"[done] DL3DV subset copied to: {dst}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download datasets for NiFi repro-lite")
    p.add_argument("--dataset", type=str, required=True, choices=["mipnerf360", "dl3dv"])
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--scenes", nargs="*", default=None, help="Mip-NeRF360 scene names or 'all'")
    p.add_argument("--max_images", type=int, default=None, help="Create images_tiny with first N images")
    p.add_argument("--remove_zip", action="store_true", help="Delete zip files after extraction")

    p.add_argument("--dl3dv_source", type=str, default=None, help="Local DL3DV subset path")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    if args.dataset == "mipnerf360":
        handle_mipnerf360(args)
    elif args.dataset == "dl3dv":
        handle_dl3dv(args)


if __name__ == "__main__":
    main()
