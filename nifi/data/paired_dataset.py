from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class PairSample:
    scene: str
    rate: str
    name: str
    clean_path: Path
    degraded_path: Path
    prompt: str


class PairedImageDataset(Dataset):
    """
    Expected layout:
      data_root/<scene>/<rate>/<split>/clean/*.png
      data_root/<scene>/<rate>/<split>/degraded/*.png
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        allowed_rates: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.allowed_rates = set(allowed_rates) if allowed_rates else None
        self.samples = self._scan_samples()

        if max_samples is not None:
            self.samples = self.samples[: max_samples]

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _scan_samples(self) -> List[PairSample]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        pairs: List[PairSample] = []
        for scene_dir in sorted([p for p in self.data_root.iterdir() if p.is_dir()]):
            for rate_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
                if self.allowed_rates and rate_dir.name not in self.allowed_rates:
                    continue
                split_dir = rate_dir / self.split
                clean_dir = split_dir / "clean"
                degraded_dir = split_dir / "degraded"
                if not clean_dir.exists() or not degraded_dir.exists():
                    continue

                for clean_path in sorted(clean_dir.glob("*.png")):
                    degraded_path = degraded_dir / clean_path.name
                    if not degraded_path.exists():
                        continue
                    pairs.append(
                        PairSample(
                            scene=scene_dir.name,
                            rate=rate_dir.name,
                            name=clean_path.stem,
                            clean_path=clean_path,
                            degraded_path=degraded_path,
                            prompt="",
                        )
                    )

        if not pairs:
            raise RuntimeError(
                f"No paired samples found in {self.data_root}. "
                "Run scripts/render_pairs.py first."
            )

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        clean = self.tf(self._load_rgb(sample.clean_path))
        degraded = self.tf(self._load_rgb(sample.degraded_path))

        return {
            "clean": clean,
            "degraded": degraded,
            "prompt": sample.prompt,
            "scene": sample.scene,
            "rate": sample.rate,
            "name": sample.name,
        }
