from typing import List, Optional

from torch.utils.data import DataLoader

from nifi.data.paired_dataset import PairedImageDataset


def build_paired_dataloader(
    data_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: Optional[int] = None,
    allowed_rates: Optional[List[str]] = None,
) -> DataLoader:
    ds = PairedImageDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        max_samples=max_samples,
        allowed_rates=allowed_rates,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
