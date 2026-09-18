"""
DataLoader Factory
==================
"""

from pathlib import Path

from torch.utils.data import DataLoader

from src.ml.dataset import BrachyceraDataset
from src.ml.transforms import get_train_transforms, get_eval_transforms
from src.utils.config import CONFIG


def create_dataloader(
    csv_file: str | Path,
    batch_size: int,
    train: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    config = CONFIG.project()
    image_size = int(config["preprocessing"]["image_size"])

    transform = (
        get_train_transforms(image_size)
        if train
        else get_eval_transforms(image_size)
    )

    dataset = BrachyceraDataset(
        csv_file=csv_file,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
    )
