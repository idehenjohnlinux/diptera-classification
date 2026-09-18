"""
Integration test for the Brachycera CNN data pipeline.

This script verifies that:

1. The fold CSV is loaded correctly.
2. Training and validation datasets are created.
3. Images are transformed into tensors.
4. DataLoader batches have valid dimensions.
5. Labels are valid integer class indices.
6. A batch passes successfully through a CNN model.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.ml.dataset import create_datasets
from src.ml.models import create_model
from src.ml.transforms import (
    get_training_transforms,
    get_validation_transforms,
)


ARCHITECTURES = (
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
)

LEVELS = ("family", "genus")
VIEWS = ("FDT", "FFF", "FLP", "FLT")


def inspect_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> None:
    """
    Validate the dimensions and values of one DataLoader batch.
    """

    if not isinstance(images, torch.Tensor):
        raise TypeError(
            f"Images must be a torch.Tensor, received {type(images)}."
        )

    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"Labels must be a torch.Tensor, received {type(labels)}."
        )

    if images.ndim != 4:
        raise ValueError(
            "Images must have shape [batch, channels, height, width]. "
            f"Received {tuple(images.shape)}."
        )

    if images.shape[1] != 3:
        raise ValueError(
            "The model expects RGB images with three channels. "
            f"Received {images.shape[1]} channels."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Labels must have shape [batch]. "
            f"Received {tuple(labels.shape)}."
        )

    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "The image and label batch sizes do not match."
        )

    minimum_label = int(labels.min().item())
    maximum_label = int(labels.max().item())

    if minimum_label < 0:
        raise ValueError(
            f"Negative class index detected: {minimum_label}."
        )

    if maximum_label >= num_classes:
        raise ValueError(
            f"Class index {maximum_label} is invalid for "
            f"{num_classes} classes."
        )


def run_pipeline_test(
    level: str,
    view: str,
    fold: int,
    architecture: str,
    batch_size: int,
    num_workers: int,
    pretrained: bool,
) -> None:
    """
    Run one complete dataset-to-model integration test.
    """

    print("=" * 65)
    print("BRACHYCERA DATA PIPELINE TEST")
    print("=" * 65)

    print(f"Taxonomic level : {level}")
    print(f"View             : {view}")
    print(f"Validation fold  : {fold}")
    print(f"Architecture     : {architecture}")
    print(f"Batch size       : {batch_size}")
    print(f"Pretrained       : {pretrained}")

    train_transform = get_training_transforms()
    validation_transform = get_validation_transforms()

    train_dataset, validation_dataset = create_datasets(
        level=level,
        view=view,
        validation_fold=fold,
        train_transform=train_transform,
        validation_transform=validation_transform,
    )

    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    if len(validation_dataset) == 0:
        raise ValueError("The validation dataset is empty.")

    if not hasattr(train_dataset, "classes"):
        raise AttributeError(
            "The dataset must provide a 'classes' attribute."
        )

    classes = train_dataset.classes
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError(
            "At least two classes are required for classification."
        )

    print("-" * 65)
    print(f"Training images   : {len(train_dataset)}")
    print(f"Validation images : {len(validation_dataset)}")
    print(f"Number of classes : {num_classes}")
    print(f"Classes           : {classes}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_images, train_labels = next(iter(train_loader))
    validation_images, validation_labels = next(
        iter(validation_loader)
    )

    inspect_batch(
        images=train_images,
        labels=train_labels,
        num_classes=num_classes,
    )

    inspect_batch(
        images=validation_images,
        labels=validation_labels,
        num_classes=num_classes,
    )

    print("-" * 65)
    print(f"Training batch images : {tuple(train_images.shape)}")
    print(f"Training batch labels : {tuple(train_labels.shape)}")
    print(f"Training image dtype  : {train_images.dtype}")
    print(f"Training label dtype  : {train_labels.dtype}")

    print(f"Validation images     : {tuple(validation_images.shape)}")
    print(f"Validation labels     : {tuple(validation_labels.shape)}")

    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(train_images)

    expected_shape = (
        train_images.shape[0],
        num_classes,
    )

    if tuple(outputs.shape) != expected_shape:
        raise ValueError(
            f"Unexpected model output shape: {tuple(outputs.shape)}. "
            f"Expected {expected_shape}."
        )

    predicted_labels = outputs.argmax(dim=1)

    print("-" * 65)
    print(f"Device               : {device}")
    print(f"Model output shape   : {tuple(outputs.shape)}")
    print(f"Prediction shape     : {tuple(predicted_labels.shape)}")
    print(f"Example labels       : {train_labels[:5].tolist()}")
    print(f"Example predictions  : {predicted_labels[:5].tolist()}")

    print("-" * 65)
    print("STATUS: COMPLETE DATA PIPELINE IS VALID")
    print("=" * 65)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test the complete Brachycera dataset and CNN pipeline."
        )
    )

    parser.add_argument(
        "--level",
        choices=LEVELS,
        required=True,
        help="Taxonomic classification level.",
    )

    parser.add_argument(
        "--view",
        choices=VIEWS,
        required=True,
        help="Morphological image view.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Validation fold number.",
    )

    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="resnet18",
        help="CNN architecture.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images per batch.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet pretrained weights.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_parser()
    args = parser.parse_args()

    if args.fold < 1 or args.fold > 5:
        parser.error("--fold must be between 1 and 5.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    run_pipeline_test(
        level=args.level,
        view=args.view,
        fold=args.fold,
        architecture=args.architecture,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
