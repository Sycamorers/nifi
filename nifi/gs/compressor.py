import json
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter


SUPPORTED_IMAGE_DIRS = ["images_4", "images_2", "images", "rgb", "train"]
IMAGE_PATTERNS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")


@dataclass
class CompressionConfig:
    jpeg_quality_min: int = 8
    jpeg_quality_max: int = 55
    downsample_min: int = 2
    downsample_max: int = 8



def list_scene_images(scene_dir: Path) -> List[Path]:
    scene_dir = Path(scene_dir)

    candidates = []
    for d in SUPPORTED_IMAGE_DIRS:
        p = scene_dir / d
        if p.exists() and p.is_dir():
            candidates.append(p)

    if not candidates:
        # fallback: all png/jpg under scene root
        imgs = []
        for pat in IMAGE_PATTERNS:
            imgs.extend(sorted(scene_dir.glob(pat)))
        if imgs:
            return imgs
        raise FileNotFoundError(
            f"Could not find image folders in {scene_dir}. "
            f"Checked {SUPPORTED_IMAGE_DIRS}."
        )

    img_dir = candidates[0]
    imgs = []
    for pat in IMAGE_PATTERNS:
        imgs.extend(sorted(img_dir.glob(pat)))
    if not imgs:
        raise RuntimeError(f"No images found under {img_dir}")
    return imgs


class Proxy3DGSCompressor:
    """
    Repro-lite artifact synthesizer that emulates extreme 3DGS compression artifacts.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def degrade_image(self, img: Image.Image, rate: float) -> Image.Image:
        img = self._ensure_rgb(img)

        # map user rate (higher => less compression) to severity [0, 1]
        rate = float(rate)
        clamped = max(0.01, min(1.0, rate))
        severity = 1.0 - clamped

        # spatial decimation (proxy for gaussian pruning/quantization)
        ds_factor = int(round(
            self.config.downsample_min + severity * (self.config.downsample_max - self.config.downsample_min)
        ))
        ds_factor = max(1, ds_factor)

        w, h = img.size
        low_w = max(8, w // ds_factor)
        low_h = max(8, h // ds_factor)

        img_ds = img.resize((low_w, low_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BICUBIC)

        if severity > 0.35:
            img_ds = img_ds.filter(ImageFilter.GaussianBlur(radius=0.6 + 2.2 * (severity - 0.35)))

        # color quantization to mimic entropy-constrained coding effects
        n_colors = int(round(256 - severity * 224))
        n_colors = max(16, min(256, n_colors))
        img_q = img_ds.quantize(colors=n_colors, method=Image.MEDIANCUT).convert("RGB")

        # JPEG roundtrip to inject blocking/ringing
        quality = int(round(
            self.config.jpeg_quality_min + clamped * (self.config.jpeg_quality_max - self.config.jpeg_quality_min)
        ))
        quality = max(5, min(95, quality))
        buf = BytesIO()
        img_q.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        return out

    def build_scene_artifacts(
        self,
        scene_dir: Path,
        rates: Sequence[float],
        out_dir: Path,
        holdout_every: int = 8,
        max_images: Optional[int] = None,
    ) -> Dict[str, object]:
        scene_dir = Path(scene_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images = list_scene_images(scene_dir)
        if max_images is not None:
            images = images[: max_images]

        clean_dir = out_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)

        names: List[str] = []
        for i, src in enumerate(images):
            name = f"{i:05d}.png"
            names.append(name)
            with Image.open(src) as img:
                self._ensure_rgb(img).save(clean_dir / name)

        for rate in rates:
            rate_name = f"rate_{float(rate):.3f}"
            rate_dir = out_dir / rate_name
            rate_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                with Image.open(clean_dir / name) as clean:
                    degraded = self.degrade_image(clean, float(rate))
                    degraded.save(rate_dir / name)

        splits = {"train": [], "test": []}
        for i, name in enumerate(names):
            if holdout_every > 0 and i % holdout_every == 0:
                splits["test"].append(name)
            else:
                splits["train"].append(name)

        meta = {
            "scene_dir": str(scene_dir),
            "num_images": len(names),
            "rates": [float(r) for r in rates],
            "splits": splits,
            "compression": {
                "backend": "proxy",
                "jpeg_quality_min": self.config.jpeg_quality_min,
                "jpeg_quality_max": self.config.jpeg_quality_max,
                "downsample_min": self.config.downsample_min,
                "downsample_max": self.config.downsample_max,
            },
        }

        with (out_dir / "metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)

        return meta


class HACPPWrapper:
    """
    Optional hook into external HAC++ repo.

    This wrapper intentionally keeps integration minimal and requires user to provide
    both HAC++ repository path and scene preparation scripts.
    """

    def __init__(self, hacpp_repo: Path):
        self.hacpp_repo = Path(hacpp_repo)
        if not self.hacpp_repo.exists():
            raise FileNotFoundError(f"HAC++ repo path not found: {hacpp_repo}")

    def run(
        self,
        scene_dir: Path,
        out_dir: Path,
        rates: Sequence[float],
        train_cmd_template: str,
        compress_cmd_template: str,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for rate in rates:
            env = {
                "SCENE": str(scene_dir),
                "OUT": str(out_dir),
                "RATE": str(rate),
            }
            train_cmd = train_cmd_template.format(**env)
            compress_cmd = compress_cmd_template.format(**env)

            subprocess.run(train_cmd, shell=True, check=True, cwd=self.hacpp_repo)
            subprocess.run(compress_cmd, shell=True, check=True, cwd=self.hacpp_repo)
